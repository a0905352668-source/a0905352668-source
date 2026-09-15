"""Guarded services-only cutover. Invoke the candidate wrapper as root.

CA arguments name immutable, already-installed trust files. Only current, the
live VLM configuration and processes.services are published or rolled back.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from live_operator.cli import LifecycleError, RuntimeHooks
from live_operator.config import LiveConfig
from live_operator.processes import ProcessIdentity, StateStore, capture_identity, is_same_process
from live_operator.vlm_review import DEFAULT_VLM_REVIEW_CONFIG, VLMReviewConfig
from live_operator.watchdog import probe_port


@dataclass(frozen=True)
class ReloadPaths:
    candidate_release: Path
    rollback_release: Path
    config: Path
    candidate_vlm_config: Path
    rollback_vlm_config: Path
    candidate_ca: Path
    rollback_ca: Path
    watchdog_unit: str
    current: Path | None = None
    live_vlm_config: Path | None = None


@dataclass(frozen=True)
class LaunchContext:
    environment: dict[str, str]
    python: str
    cwd: str
    uid: int
    gid: int
    groups: tuple[int, ...]


def _regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LifecycleError("expected a regular non-symlink file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_file(path: Path, content: bytes, metadata: os.stat_result) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode))
        if os.geteuid() == 0:
            os.fchown(descriptor, metadata.st_uid, metadata.st_gid)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_link(path: Path, target: str | Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        temporary.symlink_to(target)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ReloadHooks(RuntimeHooks):
    """External process and systemd boundary; never exposes component selection."""

    def __init__(self):
        super().__init__()
        self.children: dict[int, subprocess.Popen] = {}

    @staticmethod
    def _systemctl(*arguments: str) -> str:
        result = subprocess.run(["systemctl", *arguments],
                                capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise LifecycleError("watchdog systemctl operation failed")
        return result.stdout

    def watchdog_state(self, unit: str) -> dict[str, str]:
        output = self._systemctl("show", unit, "--property=KillMode,ExecStop,ActiveState,LoadState")
        values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        if values.get("LoadState") != "loaded":
            raise LifecycleError("watchdog unit unavailable")
        return values

    def suspend_watchdog(self, unit: str) -> None:
        self._systemctl("stop", unit)
        if self.watchdog_state(unit).get("ActiveState") != "inactive":
            raise LifecycleError("watchdog did not stop")

    def restore_watchdog(self, unit: str, was_active: bool) -> None:
        self._systemctl("start" if was_active else "stop", unit)
        wanted = "active" if was_active else "inactive"
        if self.watchdog_state(unit).get("ActiveState") != wanted:
            raise LifecycleError("watchdog restoration unproven")

    def alive(self, identity: ProcessIdentity) -> bool:
        child = self.children.get(identity.pid)
        if child is not None and child.poll() is not None:
            child.wait(timeout=0)
            return False
        if not is_same_process(identity):
            return False
        try:
            raw = Path(f"/proc/{identity.pid}/stat").read_text()
            return raw[raw.rfind(")") + 2:].split()[0] not in {"Z", "X"}
        except (OSError, IndexError):
            return False

    def launch_context(self, identity: ProcessIdentity, config: Path, run_dir: Path) -> LaunchContext:
        proc = Path(f"/proc/{identity.pid}")
        if os.geteuid() != 0:
            raise LifecycleError("reload requires root for the narrow publication boundary")
        details = proc.stat()
        status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line)
        groups = tuple(int(group) for group in status["Groups"].split())
        if len(set(status["Uid"].split())) != 1 or len(set(status["Gid"].split())) != 1:
            raise LifecycleError("services has mixed real/effective credentials")
        if details.st_uid == 0:
            raise LifecycleError("replacement services must run as an unprivileged owner")
        environment = dict(item.split("=", 1) for item in
                           (proc / "environ").read_bytes().decode().split("\0") if "=" in item)
        if environment.get("JIAN_KONG_OWNER_TOKEN") != identity.owner_token:
            raise LifecycleError("services environment ownership mismatch")
        command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        expected = ["-m", "live_operator.cli", "_services", "--run-dir", str(run_dir), "--config", str(config)]
        if command[1:] != expected:
            raise LifecycleError("services launch command does not match requested run/config")
        LiveConfig.load(config)
        if not self.alive(identity):
            raise LifecycleError("services identity changed while capturing environment")
        return LaunchContext(environment, os.readlink(proc / "exe"), os.readlink(proc / "cwd"),
                             details.st_uid, details.st_gid, groups)

    def start_services(self, release: Path, run_dir: Path, config: Path,
                       context: LaunchContext) -> ProcessIdentity:
        environment = dict(context.environment)
        environment["PYTHONPATH"] = str(release)
        owner = uuid.uuid4().hex
        environment["JIAN_KONG_OWNER_TOKEN"] = owner
        logs = run_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        if os.geteuid() == 0:
            os.chown(logs, context.uid, context.gid)
        with (logs / "services.log").open("ab", buffering=0) as log:
            if os.geteuid() == 0:
                os.fchown(log.fileno(), context.uid, context.gid)
            child = subprocess.Popen(
                [context.python, "-m", "live_operator.cli", "_services", "--run-dir", str(run_dir),
                 "--config", str(config)], stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, env=environment, cwd=context.cwd, start_new_session=True,
                user=context.uid, group=context.gid, extra_groups=context.groups)
        self.children[child.pid] = child
        try:
            identity = capture_identity(child, owner_token=owner)
            if not self.alive(identity):
                raise LifecycleError("services exited during identity capture")
            return identity
        except BaseException:
            # A capture failure still leaves a child we own and must reap.
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait(timeout=5)
            raise

    def stop_services(self, identity: ProcessIdentity) -> None:
        child = self.children.get(identity.pid)
        if child is not None:
            child.poll()  # Reap before the inherited start-token check sees a zombie.
        self.stop_component("services", identity)
        if child is not None:
            child.wait(timeout=5)
        deadline = time.monotonic() + 10
        while self.alive(identity) or self.has_owned_processes("services", identity):
            if time.monotonic() >= deadline:
                raise LifecycleError("owned services did not stop")
            time.sleep(0.1)

    def wait_port(self, verify: Callable[[], None]) -> None:
        deadline = time.monotonic() + 75.0
        while True:
            verify()
            if probe_port(8767):  # Deliberately no SO_REUSEADDR: also wait out TIME_WAIT.
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LifecycleError("port 8767 not reusable within 75 seconds")
            time.sleep(min(0.25, remaining))


class ServicesReload:
    def __init__(self, store: StateStore, paths: ReloadPaths, hooks: ReloadHooks | None = None):
        self.store = store
        self.paths = paths
        self.hooks = hooks or ReloadHooks()

    def _preserved(self) -> None:
        state = self.store.load()
        if state is None:
            raise LifecycleError("state disappeared during reload")
        for name in ("mediamtx", "deepstream"):
            if state.get("processes", {}).get(name) != self.before[name].to_dict() or not self.hooks.alive(self.before[name]):
                raise LifecycleError("preserved process identity changed")

    def _all_alive(self) -> None:
        self._preserved()
        if not self.hooks.alive(self.service):
            raise LifecycleError("services identity is not alive")

    def _publish(self, release: Path, content: bytes, target: str | Path) -> None:
        self._preserved()
        # Sources and CA contents are captured before blocking; CA files are immutable.
        self._verify_ca(release)
        _atomic_file(self.live_vlm, content, self.live_metadata)
        _atomic_link(self.current, target)

    def _verify_ca(self, release: Path) -> None:
        # A broken candidate trust file must not prevent restoring the old trust domain.
        paths = self.ca_contents if release == self.paths.candidate_release else (self.paths.rollback_ca,)
        for path in paths:
            if _regular(path) != self.ca_contents[path]:
                raise LifecycleError("immutable CA changed")

    def _verify_domain(self, release: Path) -> None:
        expected = self.candidate_bytes if release == self.paths.candidate_release else self.rollback_bytes
        if self.current.resolve() != release.resolve() or _regular(self.live_vlm) != expected:
            raise LifecycleError("published release/configuration changed")
        if _regular(self.paths.config) != self.live_config_bytes:
            raise LifecycleError("live operator configuration changed")
        self._verify_ca(release)

    def _start_ready_save(self, release: Path) -> None:
        self.hooks.wait_port(self._preserved)
        self._preserved()
        self.service = self.hooks.start_services(release, self.run_dir, self.paths.config, self.context)
        self._all_alive()
        self.hooks.http_self_check()
        self._all_alive()
        self._verify_domain(release)
        updated = dict(self.original, processes=dict(self.original["processes"], services=self.service.to_dict()))
        self.store.save(updated)
        # StateStore's atomic replacement is still inside its lock; restore the
        # original file owner before the watchdog can read it as that Unix user.
        if os.geteuid() == 0:
            os.chown(self.store.path, self.state_metadata.st_uid, self.state_metadata.st_gid)
        if self.store.load() != updated:
            raise LifecycleError("services identity persistence unproven")

    def _drain(self, requested_at: float) -> None:
        try:
            self.hooks.drain_clips(8.0, self.service.owner_token, requested_at)
        except LifecycleError as error:
            if "clip drain handshake timed out" not in str(error):
                raise
            # Existing 8-second handshake can race a worker finishing its clip.
            # Only this explicit timeout earns a further bounded fresh-ack wait.
            deadline = time.monotonic() + 30
            while True:
                self._all_alive()
                path = self.hooks._storage_for_run(self.run_dir).metadata_dir / "worker_status.json"
                try:
                    value = json.loads(path.read_text())
                    updated = value["updated_at"]
                    active = value["active"]
                    if (value["service_instance"] == self.service.owner_token and value["accepting"] is False
                            and type(updated) in (float, int) and math.isfinite(updated)
                            and updated >= requested_at and 0 <= time.time() - updated <= 2.0
                            and type(active) is int and active == 0):
                        return
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LifecycleError("extended drain acknowledgement unproven")
                time.sleep(min(0.1, remaining))
        self._all_alive()

    def _preflight(self) -> bool:
        p = self.paths
        self.original = self.store.load()
        if not self.original or self.original.get("state") != "running":
            raise LifecycleError("operator state must be running")
        self.state_metadata = self.store.path.stat()
        self.before = {name: ProcessIdentity.from_value(self.original["processes"][name])
                       for name in ("mediamtx", "deepstream", "services")}
        for identity in self.before.values():
            if (identity.pid <= 0 or identity.pgid != identity.pid or
                    not re.fullmatch(r"[0-9a-f]{32}", identity.owner_token or "") or
                    not self.hooks.alive(identity)):
                raise LifecycleError("all three owned identities must be alive")
        self.service = self.before["services"]
        for release in (p.candidate_release, p.rollback_release):
            if not release.is_absolute() or release.is_symlink() or not release.is_dir():
                raise LifecycleError("release must be a real non-symlink directory")
            _regular(release / "live_operator" / "cli.py")
        if p.candidate_release.parent.resolve() != p.rollback_release.parent.resolve():
            raise LifecycleError("releases must share their releases parent")
        if p.candidate_release.resolve() == p.rollback_release.resolve():
            raise LifecycleError("candidate and rollback must differ")
        canonical_current = p.rollback_release.parent.parent / "current"
        self.current = p.current or canonical_current
        if self.current.absolute() != canonical_current.absolute() or not self.current.is_symlink():
            raise LifecycleError("current must be the sibling release selector symlink")
        self.old_target = os.readlink(self.current)
        if self.current.resolve() != p.rollback_release.resolve():
            raise LifecycleError("current does not select rollback release")
        self.run_dir = Path(self.original["run_dir"])
        self.hooks.run_dir = self.run_dir
        self.context = self.hooks.launch_context(self.service, p.config, self.run_dir)
        self.live_config_bytes = _regular(p.config)
        configured = self.context.environment.get("JIAN_KONG_VLM_REVIEW_CONFIG", str(DEFAULT_VLM_REVIEW_CONFIG))
        self.live_vlm = Path(configured)
        if not self.live_vlm.is_absolute() or (p.live_vlm_config is not None and p.live_vlm_config != self.live_vlm):
            raise LifecycleError("live VLM path disagrees with owned services environment/default")
        self.candidate_bytes = _regular(p.candidate_vlm_config)
        self.rollback_bytes = _regular(p.rollback_vlm_config)
        if _regular(self.live_vlm) != self.rollback_bytes:
            raise LifecycleError("rollback configuration is not the current live configuration")
        self.live_metadata = self.live_vlm.stat()
        if stat.S_IMODE(self.live_metadata.st_mode) != 0o600:
            raise LifecycleError("live VLM configuration must retain private 0600 mode")
        if self.live_vlm in {p.candidate_vlm_config, p.rollback_vlm_config, p.candidate_ca, p.rollback_ca}:
            raise LifecycleError("live VLM target must not alias immutable sources")
        for config_path, ca_path in ((p.candidate_vlm_config, p.candidate_ca), (p.rollback_vlm_config, p.rollback_ca)):
            config = VLMReviewConfig.load(config_path)
            if not ca_path.is_absolute() or config.tls_ca_file != ca_path:
                raise LifecycleError("VLM configuration must reference its exact immutable CA argument")
        self.ca_contents = {path: _regular(path) for path in (p.candidate_ca, p.rollback_ca)}
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", p.watchdog_unit):
            raise LifecycleError("watchdog unit must be a service unit name")
        watchdog = self.hooks.watchdog_state(p.watchdog_unit)
        if watchdog.get("KillMode") != "process" or watchdog.get("ExecStop") != "":
            raise LifecycleError("watchdog requires KillMode=process and empty ExecStop")
        if watchdog.get("ActiveState") not in {"active", "inactive"}:
            raise LifecycleError("watchdog must be active or inactive")
        self._all_alive()
        return watchdog["ActiveState"] == "active"

    def execute(self) -> dict:
        receipt = {"ok": False, "rolled_back": False, "before": {}, "after": {}, "watchdog_restored": False}
        restore = False
        blocked = False
        stage = "preflight"
        try:
            with self.store.lock():
                was_active = self._preflight()
                receipt["before"] = {name: identity.to_dict() for name, identity in self.before.items()}
                restore = True  # Even a partially failed systemctl stop needs finally restoration.
                try:
                    stage = "watchdog_suspend"
                    self.hooks.suspend_watchdog(self.paths.watchdog_unit)
                    self._all_alive()
                    stage = "drain"
                    blocked = True  # A marker write failure can occur after its atomic replace.
                    requested_at = self.hooks.block_new_events(self.service.owner_token)
                    self._drain(requested_at)
                    stage = "candidate"
                    self.hooks.stop_services(self.service)
                    self._publish(self.paths.candidate_release, self.candidate_bytes, self.paths.candidate_release)
                    self._start_ready_save(self.paths.candidate_release)
                    receipt["ok"] = True
                except BaseException as error:
                    receipt["error"] = {"stage": stage, "type": type(error).__name__}
                    if blocked:
                        try:
                            self.hooks.stop_services(self.service)
                            self._publish(self.paths.rollback_release, self.rollback_bytes, self.old_target)
                            self._start_ready_save(self.paths.rollback_release)
                            receipt["rolled_back"] = True
                        except BaseException as rollback_error:
                            receipt["rollback_error"] = {"type": type(rollback_error).__name__}
                    receipt["ok"] = False
                finally:
                    receipt["after"] = (self.store.load() or {}).get("processes", {})
        except BaseException as error:
            receipt["ok"] = False
            receipt["error"] = {"stage": stage, "type": type(error).__name__}
        finally:
            if restore:
                try:
                    self.hooks.restore_watchdog(self.paths.watchdog_unit, was_active)
                    receipt["watchdog_restored"] = True
                except BaseException as error:
                    receipt["ok"] = False
                    receipt["watchdog_error"] = {"type": type(error).__name__}
        # Never include config, environment, command lines or arbitrary exception text.
        return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("candidate-release", "rollback-release", "config", "candidate-vlm-config",
                   "rollback-vlm-config", "candidate-ca", "rollback-ca", "state"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--watchdog-unit", required=True)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--live-vlm-config", type=Path)
    values = vars(parser.parse_args(argv))
    store = StateStore(values.pop("state"))
    previous = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupted)
        receipt = ServicesReload(store, ReloadPaths(**values)).execute()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
