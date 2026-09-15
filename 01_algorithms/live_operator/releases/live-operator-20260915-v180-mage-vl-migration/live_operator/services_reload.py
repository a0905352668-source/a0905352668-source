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
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from live_operator.cli import LifecycleError, RuntimeHooks
from live_operator.config import LiveConfig
from live_operator.processes import ProcessIdentity, StateStore, capture_identity, is_same_process
from live_operator.vlm_review import DEFAULT_VLM_REVIEW_CONFIG, VLMReviewConfig
from live_operator.watchdog import probe_port
from live_operator.reload_files import (
    PinnedDirectory, PinnedFile, ReloadStateStore, ReloadTransactionLock, identity as file_identity, regular,
)


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


def _absolute(path: Path, description: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise LifecycleError(f"{description} must be absolute without traversal")
    return value


def _normalize_legacy_path(spelling: Path, canonical: Path, project_root: Path,
                           description: str) -> Path:
    """Resolve an untrusted legacy spelling once, then return its authority."""

    spelling = _absolute(spelling, description)
    canonical = _absolute(canonical, f"canonical {description}")
    project_root = _absolute(project_root, "trusted project root")
    try:
        if canonical.resolve(strict=True) != canonical or project_root.resolve(strict=True) != project_root:
            raise LifecycleError(f"canonical {description} must not contain aliases")
        canonical.relative_to(project_root)
        resolved = spelling.resolve(strict=True)
        resolved.relative_to(project_root)
    except ValueError as error:
        raise LifecycleError(f"{description} is outside the trusted project") from error
    except OSError as error:
        raise LifecycleError(f"{description} is unavailable") from error
    if resolved != canonical:
        raise LifecycleError(f"{description} does not match its canonical authority")
    return canonical


def _regular(path: Path) -> bytes:
    file = PinnedFile(path)
    try:
        return file.content
    finally:
        file.close()


@contextmanager
def _service_user(context: LaunchContext):
    """Runtime helpers may write only with the existing unprivileged credentials.

    Effective-ID switching is limited to this single-threaded reload process.
    Restoring root occurs before any publication or systemd operation.
    """
    if os.geteuid() == context.uid and os.getegid() == context.gid:
        yield
        return
    if os.geteuid() != 0 or context.uid == 0 or threading.active_count() != 1:
        raise LifecycleError("runtime write requires a single-threaded service-owner context")
    old_gid, old_groups = os.getegid(), os.getgroups()
    try:
        os.setgroups(context.groups)
        os.setegid(context.gid)
        os.seteuid(context.uid)
        yield
    finally:
        os.seteuid(0)
        os.setegid(old_gid)
        os.setgroups(old_groups)


class ReloadHooks(RuntimeHooks):
    """External process and systemd boundary; never exposes component selection."""

    trusted_uid = 0

    def __init__(self):
        super().__init__()
        self.children: dict[int, subprocess.Popen] = {}

    @contextmanager
    def transaction_lock(self):
        if os.geteuid() != 0:
            raise LifecycleError("reload requires root")
        # /run is a canonical, root-controlled directory; no caller-selected lock.
        with PinnedDirectory(Path("/run"), trusted_owner=0) as run:
            try:
                os.mkdir("jiankong-services-reload", mode=0o700, dir_fd=run.fd)
            except FileExistsError:
                pass
        with ReloadTransactionLock(Path("/run/jiankong-services-reload/transaction.lock")):
            yield

    @staticmethod
    def _systemctl(*arguments: str) -> str:
        result = subprocess.run(["/usr/bin/systemctl", *arguments],
                                capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise LifecycleError("watchdog systemctl operation failed")
        return result.stdout

    def watchdog_state(self, unit: str) -> dict[str, str]:
        output = self._systemctl("show", unit, "--property=KillMode,ActiveState,LoadState")
        values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        values["ExecStop"] = self._systemctl(
            "show", unit, "--property=ExecStop", "--value"
        ).rstrip("\n")
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
            proc = Path(f"/proc/{identity.pid}")
            raw = (proc / "stat").read_text()
            entries = (proc / "environ").read_bytes().split(b"\0")
            owners = [entry for entry in entries if entry.startswith(b"JIAN_KONG_OWNER_TOKEN=")]
            return (raw[raw.rfind(")") + 2:].split()[0] not in {"Z", "X"}
                    and identity.pgid == os.getpgid(identity.pid)
                    and owners == [f"JIAN_KONG_OWNER_TOKEN={identity.owner_token}".encode()]
                    and is_same_process(identity))
        except (OSError, IndexError):
            return False

    def launch_context(self, identity: ProcessIdentity, config: Path, run_dir: Path,
                       project_root: Path | None = None) -> LaunchContext:
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
        actual = command[1:]
        if actual and actual[0] == "-P":
            actual = actual[1:]
        expected_prefix = ["-m", "live_operator.cli", "_services", "--run-dir"]
        if (len(actual) != 7 or actual[:4] != expected_prefix or actual[5] != "--config"):
            raise LifecycleError("services launch command does not match requested run/config")
        root = project_root or Path(os.path.commonpath((str(run_dir), str(config))))
        _normalize_legacy_path(Path(actual[4]), run_dir, root, "services run path")
        _normalize_legacy_path(Path(actual[6]), config, root, "services config path")
        LiveConfig.load(config)
        if not self.alive(identity):
            raise LifecycleError("services identity changed while capturing environment")
        self.context = LaunchContext(environment, os.readlink(proc / "exe"), os.readlink(proc / "cwd"),
                                     details.st_uid, details.st_gid, groups)
        with self.open_services_log(run_dir, self.context):
            pass
        return self.context

    @staticmethod
    def _environment(release, context):
        return dict(context.environment, PYTHONPATH=str(release), PYTHONSAFEPATH="1")

    def validate_resolution(self, release, context):
        result = subprocess.run(
            [context.python, "-P", "-c", "import importlib.util,json; print(json.dumps(importlib.util.find_spec('live_operator.cli').origin))"],
            env=self._environment(release, context), cwd=context.cwd,
            user=context.uid, group=context.gid, extra_groups=context.groups,
            capture_output=True, text=True, timeout=15, check=False)
        if result.returncode:
            raise LifecycleError("safe-path module resolution failed")
        origin = Path(json.loads(result.stdout))
        if origin != release / "live_operator" / "cli.py":
            raise LifecycleError("services module resolved outside the selected release")
        return str(origin)

    def block_new_events(self, owner):
        if not re.fullmatch(r"[0-9a-f]{32}", owner):
            raise LifecycleError("invalid service owner")
        requested_at = time.time()
        metadata = self._storage_for_run(self.run_dir).metadata_dir
        # Never reuse the old predictable .stop_events.tmp. Even a malicious
        # runtime layout cannot turn this write into a privileged root write.
        with _service_user(self.context), PinnedDirectory(metadata) as directory:
            name = f".stop_events.{uuid.uuid4().hex}.tmp"
            fd = -1
            try:
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory.fd)
                with os.fdopen(fd, "w") as stream:
                    fd = -1
                    json.dump({"service_instance": owner, "requested_at": requested_at}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory.verify()
                os.replace(name, ".stop_events", src_dir_fd=directory.fd, dst_dir_fd=directory.fd)
            finally:
                if fd >= 0:
                    os.close(fd)
                try:
                    os.unlink(name, dir_fd=directory.fd)
                except FileNotFoundError:
                    pass
        return requested_at

    def drain_clips(self, *args):
        with _service_user(self.context):
            return super().drain_clips(*args)

    def http_self_check(self):
        # The inherited Range self-check creates a runtime clip fixture.
        with _service_user(self.context):
            return super().http_self_check()

    def open_services_log(self, run_dir, context):
        with PinnedDirectory(run_dir / "logs") as logs:
            fd = os.open("services.log", os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=logs.fd)
            try:
                details = os.fstat(fd)
                regular(details)
                if (details.st_uid != context.uid or details.st_gid != context.gid
                        or stat.S_IMODE(details.st_mode) & 0o022):
                    raise LifecycleError("services log must retain its exact service owner and safe mode")
                logs.verify()
                return os.fdopen(fd, "ab", buffering=0)
            except BaseException:
                os.close(fd)
                raise

    def start_services(self, release: Path, run_dir: Path, config: Path,
                       context: LaunchContext) -> ProcessIdentity:
        environment = self._environment(release, context)
        owner = uuid.uuid4().hex
        environment["JIAN_KONG_OWNER_TOKEN"] = owner
        self.context = context
        with self.open_services_log(run_dir, context) as log:
            child = subprocess.Popen(
                [context.python, "-P", "-m", "live_operator.cli", "_services", "--run-dir", str(run_dir),
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
        if is_same_process(identity) and not self.alive(identity):
            raise LifecycleError("refusing to signal services with changed live ownership")
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
        self._pins = []

    def _pin(self, value):
        self._pins.append(value)
        return value

    def _verify_releases(self, release):
        self.release_parent.verify()
        self.selector_parent.verify()
        for pin in self.release_pins[release]:
            pin.verify()

    def _pin_release(self, release):
        directory = self._pin(PinnedDirectory(release, trusted_owner=self.hooks.trusted_uid))
        pins = [directory]
        # Imported package directories and files must also be immutable to the
        # unprivileged services owner, including pre-existing bytecode caches.
        package = release / "live_operator"
        for root, dirs, files in os.walk(package, followlinks=False):
            parent = self._pin(PinnedDirectory(Path(root), trusted_owner=self.hooks.trusted_uid))
            pins.append(parent)
            for name in dirs:
                if (Path(root) / name).is_symlink():
                    raise LifecycleError("release package contains a directory symlink")
            for name in files:
                file = self._pin(PinnedFile(Path(root) / name, parent=parent))
                if file.metadata.st_uid != self.hooks.trusted_uid or stat.S_IMODE(file.metadata.st_mode) & 0o022:
                    raise LifecycleError("release package is not trusted and immutable")
                pins.append(file)
        if not any(isinstance(pin, PinnedFile) and pin.path == package / "cli.py" for pin in pins):
            raise LifecycleError("release package is missing cli.py")
        return pins

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

    def _stop_services(self):
        for name in ("mediamtx", "deepstream"):
            preserved = self.before[name]
            if (self.service.pid == preserved.pid or self.service.pgid == preserved.pgid
                    or self.service.owner_token == preserved.owner_token):
                raise LifecycleError("services identity aliases a preserved component")
        self.hooks.stop_services(self.service)
        if self.hooks.alive(self.service):
            raise LifecycleError("services stop unproven")

    def _publish(self, release: Path, content: bytes, target: str | Path) -> None:
        self._preserved()
        self._verify_releases(release)
        self._verify_ca(release)
        self.live_file.replace(content)
        self.selector.replace_link(target)

    def _verify_ca(self, release: Path) -> None:
        # A broken candidate trust file must not prevent restoring the old trust domain.
        paths = self.ca_files if release == self.paths.candidate_release else (self.paths.rollback_ca,)
        for path in paths:
            self.ca_files[path].verify()

    def _verify_domain(self, release: Path) -> None:
        expected = self.candidate_bytes if release == self.paths.candidate_release else self.rollback_bytes
        self._verify_releases(release)
        self.selector.verify()
        if self.current.resolve() != release or self.live_file.verify() != expected:
            raise LifecycleError("published release/configuration changed")
        if self.operator_file.verify() != self.live_config_bytes:
            raise LifecycleError("live operator configuration changed")
        self._verify_ca(release)

    def _start_ready_save(self, release: Path) -> None:
        self._verify_releases(release)
        self.hooks.wait_port(self._preserved)
        self._preserved()
        self._verify_releases(release)
        self.hooks.validate_resolution(release, self.context)
        self._verify_releases(release)
        self.service = self.hooks.start_services(release, self.run_dir, self.paths.config, self.context)
        self._all_alive()
        self.hooks.http_self_check()
        self._all_alive()
        self._verify_domain(release)
        updated = dict(self.original, processes=dict(self.original["processes"], services=self.service.to_dict()))
        self.store.save(updated)
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
                    value = json.loads(_regular(path))
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
        fixed_paths = (
            p.candidate_release, p.rollback_release, p.config,
            p.candidate_vlm_config, p.rollback_vlm_config,
            p.candidate_ca, p.rollback_ca, self.store.path,
        )
        for path in fixed_paths:
            _normalize_legacy_path(path, path, path.parent, "canonical reload input")
        self.project_root = Path(os.path.commonpath(str(path) for path in fixed_paths))
        if self.project_root == Path("/") or self.project_root.resolve(strict=True) != self.project_root:
            raise LifecycleError("reload inputs do not share a canonical trusted project")
        self.original = self.store.load()
        if not self.original or self.original.get("state") != "running":
            raise LifecycleError("operator state must be running")
        self.before = {name: ProcessIdentity.from_value(self.original["processes"][name])
                       for name in ("mediamtx", "deepstream", "services")}
        if len({item.pid for item in self.before.values()}) != 3 or len({item.owner_token for item in self.before.values()}) != 3:
            raise LifecycleError("component identities and owner tokens must be distinct")
        for identity in self.before.values():
            if (identity.pid <= 0 or identity.pgid != identity.pid or
                    not re.fullmatch(r"[0-9a-f]{32}", identity.owner_token or "") or
                    not self.hooks.alive(identity)):
                raise LifecycleError("all three owned identities must be alive")
        self.service = self.before["services"]
        if p.candidate_release.parent != p.rollback_release.parent:
            raise LifecycleError("releases must share their releases parent")
        if p.candidate_release == p.rollback_release:
            raise LifecycleError("candidate and rollback must differ")
        canonical_current = p.rollback_release.parent.parent / "current"
        self.current = p.current or canonical_current
        if self.current != canonical_current:
            raise LifecycleError("current must be the sibling release selector symlink")
        self.selector_parent = self._pin(PinnedDirectory(self.current.parent, trusted_owner=self.hooks.trusted_uid))
        self.release_parent = self._pin(PinnedDirectory(p.rollback_release.parent, trusted_owner=self.hooks.trusted_uid))
        self.release_pins = {release: self._pin_release(release) for release in (p.candidate_release, p.rollback_release)}
        self.selector = self._pin(PinnedFile(self.current, symlink=True))
        self.old_target = self.selector.content
        if self.current.resolve() != p.rollback_release:
            raise LifecycleError("current does not select rollback release")
        run_spelling = Path(self.original["run_dir"])
        try:
            canonical_run = run_spelling.resolve(strict=True)
        except OSError as error:
            raise LifecycleError("stored run path is unavailable") from error
        self.run_dir = _normalize_legacy_path(
            run_spelling, canonical_run, self.project_root, "stored run path"
        )
        self.hooks.run_dir = self.run_dir
        self.context = self.hooks.launch_context(
            self.service, p.config, self.run_dir, self.project_root
        )
        self.operator_file = self._pin(PinnedFile(p.config))
        self.live_config_bytes = self.operator_file.content
        configured = self.context.environment.get("JIAN_KONG_VLM_REVIEW_CONFIG", str(DEFAULT_VLM_REVIEW_CONFIG))
        live_spelling = Path(configured)
        try:
            canonical_live = (
                p.live_vlm_config
                if p.live_vlm_config is not None
                else live_spelling.resolve(strict=True)
            )
        except OSError as error:
            raise LifecycleError("live VLM path is unavailable") from error
        self.live_vlm = _normalize_legacy_path(
            live_spelling, canonical_live, self.project_root, "live VLM path"
        )
        self.context = LaunchContext(
            dict(self.context.environment, JIAN_KONG_VLM_REVIEW_CONFIG=str(self.live_vlm)),
            self.context.python, self.context.cwd, self.context.uid,
            self.context.gid, self.context.groups,
        )
        self.candidate_file = self._pin(PinnedFile(p.candidate_vlm_config))
        self.rollback_file = self._pin(PinnedFile(p.rollback_vlm_config))
        self.live_file = self._pin(PinnedFile(self.live_vlm))
        self.candidate_bytes = self.candidate_file.content
        self.rollback_bytes = self.rollback_file.content
        if self.live_file.content != self.rollback_bytes:
            raise LifecycleError("rollback configuration is not the current live configuration")
        self.live_metadata = self.live_file.metadata
        if stat.S_IMODE(self.live_metadata.st_mode) != 0o600:
            raise LifecycleError("live VLM configuration must retain private 0600 mode")
        self.ca_files = {path: self._pin(PinnedFile(path)) for path in (p.candidate_ca, p.rollback_ca)}
        files = [self.candidate_file, self.rollback_file, self.live_file, self.operator_file,
                 self.store.file, *self.ca_files.values()]
        if len(self.ca_files) != 2 or len({file_identity(file.metadata) for file in files}) != len(files):
            raise LifecycleError("CA files, sources, state and live targets must be distinct file authorities")
        for file, ca_path in ((self.candidate_file, p.candidate_ca), (self.rollback_file, p.rollback_ca)):
            if stat.S_IMODE(file.metadata.st_mode) != 0o600:
                raise LifecycleError("VLM source configuration must be private 0600")
            config = VLMReviewConfig.from_payload(json.loads(file.content))
            if file is self.candidate_file and config.tls_ca_file != ca_path:
                raise LifecycleError("candidate VLM configuration must use its canonical CA path")
            _normalize_legacy_path(
                config.tls_ca_file, ca_path, self.project_root,
                "VLM configuration CA path",
            )
        for file in files:
            file.verify()
        for release in (p.candidate_release, p.rollback_release):
            self._verify_releases(release)
            self.hooks.validate_resolution(release, self.context)
            self._verify_releases(release)
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", p.watchdog_unit):
            raise LifecycleError("watchdog unit must be a service unit name")
        watchdog = self.hooks.watchdog_state(p.watchdog_unit)
        if watchdog.get("KillMode") != "process" or watchdog.get("ExecStop") != "":
            raise LifecycleError("watchdog requires KillMode=process and empty ExecStop")
        if watchdog.get("ActiveState") not in {"active", "inactive"}:
            raise LifecycleError("watchdog must be active or inactive")
        self._all_alive()
        return watchdog["ActiveState"] == "active"

    def _rollback(self, receipt):
        errors = {}
        def attempt(stage, operation):
            try:
                operation()
                return True
            except BaseException as error:
                errors[stage] = type(error).__name__
                return False
        stopped = attempt("stop", self._stop_services)
        # These attempts are deliberately independent of stop and of the other
        # publication. A lost preserved identity cannot strand candidate files.
        config_restored = attempt("config", lambda: self.live_file.replace(self.rollback_bytes, restore=True))
        def restore_current():
            self._verify_releases(self.paths.rollback_release)
            self.selector.replace_link(self.old_target, restore=True)
        current_restored = attempt("current", restore_current)
        if stopped and config_restored and current_restored:
            receipt["rolled_back"] = attempt("readiness", lambda: self._start_ready_save(self.paths.rollback_release))
        if errors:
            receipt["rollback_errors"] = errors
            receipt["rollback_error"] = {"type": next(iter(errors.values()))}

    def execute(self) -> dict:
        receipt = {"ok": False, "rolled_back": False, "before": {}, "after": {}, "watchdog_restored": False}
        self._watchdog_restore = None
        original_store = self.store
        try:
            with self.hooks.transaction_lock():
                # Hold the root transaction lock across state lock release and
                # watchdog restoration; state lock is always acquired second.
                self.store = ReloadStateStore(original_store.path)
                try:
                    with self.store.lock():
                        self._execute_locked(receipt)
                finally:
                    if self._watchdog_restore is not None:
                        try:
                            self.hooks.restore_watchdog(self.paths.watchdog_unit, self._watchdog_restore)
                            receipt["watchdog_restored"] = True
                        except BaseException as error:
                            receipt["ok"] = False
                            receipt["watchdog_error"] = {"type": type(error).__name__}
        except BaseException as error:
            receipt["ok"] = False
            receipt["error"] = {"stage": "preflight", "type": type(error).__name__}
        finally:
            if self.store is not original_store:
                self.store.close()
                self.store = original_store
            for pin in reversed(self._pins):
                pin.close()
            self._pins.clear()
        return receipt

    def _execute_locked(self, receipt):
        blocked = False
        stage = "preflight"
        self._watchdog_restore = None
        try:
            was_active = self._preflight()
            receipt["before"] = {name: identity.to_dict() for name, identity in self.before.items()}
            self._watchdog_restore = was_active
            stage = "watchdog_suspend"
            self.hooks.suspend_watchdog(self.paths.watchdog_unit)
            self._all_alive()
            stage = "drain"
            blocked = True  # A marker write failure can occur after its atomic replace.
            requested_at = self.hooks.block_new_events(self.service.owner_token)
            self._drain(requested_at)
            stage = "candidate"
            self._verify_releases(self.paths.candidate_release)
            self._stop_services()
            self._publish(self.paths.candidate_release, self.candidate_bytes, self.paths.candidate_release)
            self._start_ready_save(self.paths.candidate_release)
            receipt["ok"] = True
        except BaseException as error:
            receipt["ok"] = False
            receipt["error"] = {"stage": stage, "type": type(error).__name__}
            if blocked:
                self._rollback(receipt)
        finally:
            try:
                receipt["after"] = (self.store.load() or {}).get("processes", {})
            except BaseException as error:
                receipt["ok"] = False
                receipt["state_error"] = {"type": type(error).__name__}


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
