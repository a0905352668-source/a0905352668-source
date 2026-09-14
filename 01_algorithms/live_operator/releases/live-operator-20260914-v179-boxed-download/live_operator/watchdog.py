"""Conservative runtime probes and recovery for the live operator."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal
from live_operator.storage_paths import RunStorage, resolve_run_storage

from live_operator.processes import (
    ProcessIdentity,
    StateStore,
    capture_identity,
    is_same_process,
    owned_process_group_exists,
    stop_process,
)


CommandResult = subprocess.CompletedProcess[str]
CommandRunner = Callable[[list[str]], CommandResult]
ProcessScan = dict[str, tuple[int, ...]]
GPU_LOCK_PATH = Path("/tmp/jiankong_gpu0.lock")
CHECK_INTERVAL_SECONDS = 15
UNKNOWN_PROCESS_CONFIRMATIONS = 2
PROGRESS_STALL_SECONDS = 45.0
BACKOFF_SECONDS = (30, 60, 120, 300)
HEALTH_PATH = Path(
    "/media/boshi/Data/JianKong/02_configs/runtime/live_operator_health.json"
)
STATE_PATH = Path(
    "/media/boshi/Data/JianKong/02_configs/runtime/live_operator_state.json"
)
WATCHDOG_LOCK_PATH = Path("/run/lock/jiankong-live-watchdog.lock")
SERVICES_READY_TIMEOUT_SECONDS = 30
WORKER_STALE_SECONDS = 120.0
HEALTH_PUBLISH_WARNING = (
    "jiankong-watchdog: health publication failed; recovery paused"
)


@dataclass(frozen=True)
class ComponentSnapshot:
    identity_alive: bool
    owned_processes_alive: bool = False
    unknown_processes: tuple[int, ...] = ()
    probe_confident: bool = True


@dataclass(frozen=True)
class RuntimeSnapshot:
    mediamtx: ComponentSnapshot
    deepstream: ComponentSnapshot
    services: ComponentSnapshot
    gpu_lock_free: bool | None
    gpu_owned_by_current: bool
    gpu_compute_pids: tuple[int, ...]
    docker_ready: bool
    network_ready: bool
    port_8767_free: bool
    gpu_owned_compute_pids: tuple[int, ...] = ()
    inference_stalled_streams: tuple[int, ...] = ()
    runtime_identity_key: str = ""
    worker_health: str = "healthy"
    runtime_run_key: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryDecision:
    action: Literal[
        "none", "start_full", "restart_full", "restart_services", "blocked"
    ]
    state: Literal["healthy", "recovering", "degraded", "blocked", "offline"]
    reason_code: str
    message: str


class ProcessedProgressMonitor:
    """Detect only sustained per-stream zero progress in validated live stats."""

    def __init__(
        self,
        stall_seconds: float = PROGRESS_STALL_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(stall_seconds, bool)
            or not isinstance(stall_seconds, (int, float))
            or not math.isfinite(float(stall_seconds))
            or float(stall_seconds) <= 0
        ):
            raise ValueError("stall_seconds must be positive and finite")
        self._stall_seconds = float(stall_seconds)
        self._monotonic = monotonic
        self._run_key: tuple[str, ...] | None = None
        self._processed: tuple[int, ...] | None = None
        self._last_progress_at: tuple[float, ...] = ()

    def observe(
        self,
        run_key: tuple[str, ...] | None,
        payload: dict[str, Any] | None,
        expected_streams: int,
    ) -> tuple[int, ...]:
        counters = _processed_counters(payload, expected_streams)
        if run_key is None or counters is None:
            self.reset()
            return ()
        try:
            now = float(self._monotonic())
        except (TypeError, ValueError):
            self.reset()
            return ()
        if not math.isfinite(now):
            self.reset()
            return ()
        if self._run_key != run_key or self._processed is None:
            self._set_baseline(run_key, counters, now)
            return ()
        if len(counters) != len(self._processed) or any(
            current < previous
            for current, previous in zip(counters, self._processed)
        ) or any(now < previous for previous in self._last_progress_at):
            self._set_baseline(run_key, counters, now)
            return ()

        self._last_progress_at = tuple(
            now if current > previous else last_progress_at
            for current, previous, last_progress_at in zip(
                counters, self._processed, self._last_progress_at
            )
        )
        self._processed = counters
        return tuple(
            index
            for index, last_progress_at in enumerate(self._last_progress_at)
            if now - last_progress_at >= self._stall_seconds
        )

    def reset(self) -> None:
        self._run_key = None
        self._processed = None
        self._last_progress_at = ()

    def _set_baseline(
        self,
        run_key: tuple[str, ...],
        counters: tuple[int, ...],
        now: float,
    ) -> None:
        self._run_key = run_key
        self._processed = counters
        self._last_progress_at = tuple(now for _ in counters)


def _bound_run_storage(run_dir: str | Path, bindings: dict[Path, RunStorage]) -> RunStorage:
    storage = resolve_run_storage(run_dir)
    known = bindings.setdefault(storage.run_dir, storage)
    if known != storage:
        raise ValueError("watchdog metadata authority changed")
    return storage


class WatchdogProbe:
    """Collect a conservative runtime snapshot without retaining command lines."""

    def __init__(
        self,
        *,
        state_store: StateStore,
        same_process: Callable[[ProcessIdentity], bool] = is_same_process,
        owned_group: Callable[
            [ProcessIdentity], bool
        ] = owned_process_group_exists,
        owned_group_pids: Callable[
            [ProcessIdentity], tuple[int, ...] | None
        ] = lambda identity: owned_process_group_pids(identity),
        process_scanner: Callable[[], ProcessScan] = lambda: scan_runtime_processes(),
        gpu_lock_probe: Callable[[], bool | None] = lambda: probe_gpu_lock(),
        gpu_runner: CommandRunner = lambda command: _run_command(command),
        docker_runner: CommandRunner = lambda command: _run_command(command),
        network_probe: Callable[[], bool] = lambda: probe_network(),
        port_probe: Callable[[int], bool] = lambda port: probe_port(port),
        progress_monitor: ProcessedProgressMonitor | None = None,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self._state_store = state_store
        self._storage_bindings: dict[Path, RunStorage] = {}
        self._same_process = same_process
        self._owned_group = owned_group
        self._owned_group_pids = owned_group_pids
        self._process_scanner = process_scanner
        self._gpu_lock_probe = gpu_lock_probe
        self._gpu_runner = gpu_runner
        self._docker_runner = docker_runner
        self._network_probe = network_probe
        self._port_probe = port_probe
        self._progress_monitor = progress_monitor or ProcessedProgressMonitor()
        self._wall_time = wall_time

    def snapshot(self) -> RuntimeSnapshot:
        identities, state_confident, runtime_run_key = self._load_identities()
        identity_alive: dict[str, bool] = {}
        owned_alive: dict[str, bool] = {}
        owned_pids: dict[str, tuple[int, ...]] = {}
        component_confident = {
            name: state_confident for name in ("mediamtx", "deepstream", "services")
        }

        for name in component_confident:
            identity = identities.get(name)
            if identity is None:
                identity_alive[name] = False
                owned_alive[name] = False
                owned_pids[name] = ()
                continue
            try:
                identity_alive[name] = bool(self._same_process(identity))
                owned_alive[name] = bool(self._owned_group(identity))
                group_pids = self._owned_group_pids(identity)
                if group_pids is None:
                    raise RuntimeError("owner process group could not be inspected")
                owned_pids[name] = tuple(
                    sorted(
                        {
                            pid
                            for pid in group_pids
                            if isinstance(pid, int) and pid > 0
                        }
                    )
                )
                owned_alive[name] = owned_alive[name] or bool(owned_pids[name])
            except Exception:
                identity_alive[name] = False
                owned_alive[name] = False
                owned_pids[name] = ()
                component_confident[name] = False

        (
            docker_ready,
            docker_owned,
            docker_confident,
            docker_owned_pids,
        ) = self._probe_docker(
            identities.get("deepstream")
        )
        owned_pids["deepstream"] = tuple(
            sorted(set(owned_pids["deepstream"]) | set(docker_owned_pids))
        )
        owned_alive["deepstream"] = owned_alive["deepstream"] or docker_owned
        component_confident["deepstream"] = (
            component_confident["deepstream"] and docker_confident
        )

        try:
            scanned = self._process_scanner()
            if not isinstance(scanned, dict):
                raise TypeError("process scanner must return a mapping")
            unknown = self._unknown_processes(
                scanned, identities, identity_alive, owned_pids
            )
        except Exception:
            scanned = {}
            unknown = {
                name: () for name in ("mediamtx", "deepstream", "services")
            }
            for name in component_confident:
                component_confident[name] = False

        components = {
            name: ComponentSnapshot(
                identity_alive=identity_alive[name],
                owned_processes_alive=owned_alive[name],
                unknown_processes=unknown[name],
                probe_confident=component_confident[name],
            )
            for name in component_confident
        }

        gpu_lock_free = self._safe_gpu_lock_probe()
        gpu_compute_pids = self._probe_gpu_compute_pids()
        if gpu_compute_pids is None:
            gpu_lock_free = None
            gpu_compute_pids = ()

        try:
            network_ready = bool(self._network_probe())
        except Exception:
            network_ready = False
        try:
            port_free = bool(self._port_probe(8767))
        except Exception:
            port_free = False
            services = components["services"]
            components["services"] = ComponentSnapshot(
                identity_alive=services.identity_alive,
                owned_processes_alive=services.owned_processes_alive,
                unknown_processes=services.unknown_processes,
                probe_confident=False,
            )

        return RuntimeSnapshot(
            mediamtx=components["mediamtx"],
            deepstream=components["deepstream"],
            services=components["services"],
            gpu_lock_free=gpu_lock_free,
            gpu_owned_by_current=(
                gpu_lock_free is False
                and (
                    components["deepstream"].identity_alive
                    or components["deepstream"].owned_processes_alive
                )
            ),
            gpu_compute_pids=gpu_compute_pids,
            docker_ready=docker_ready,
            network_ready=network_ready,
            port_8767_free=port_free,
            gpu_owned_compute_pids=owned_gpu_compute_pids(
                gpu_compute_pids, owned_pids
            ),
            inference_stalled_streams=self._probe_inference_progress(),
            runtime_identity_key=_runtime_identity_key(identities),
            worker_health=self._probe_worker_health(identities.get("services")),
            runtime_run_key=runtime_run_key,
        )

    def _probe_worker_health(self, identity: ProcessIdentity | None) -> str:
        try:
            state = self._state_store.load()
            if not isinstance(state, dict) or state.get("state") != "running" or identity is None:
                return "unknown"
            run_dir = state.get("run_dir")
            if not isinstance(run_dir, str) or not Path(run_dir).is_absolute():
                return "unknown"
            return _worker_health(
                _bound_run_storage(run_dir, self._storage_bindings).metadata_dir / "worker_status.json",
                identity.owner_token, now=self._wall_time(),
            )
        except (OSError, TypeError, ValueError, KeyError):
            return "unknown"

    def _probe_inference_progress(self) -> tuple[int, ...]:
        try:
            state = self._state_store.load()
            if not isinstance(state, dict) or state.get("state") != "running":
                return self._progress_monitor.observe(None, None, 0)
            run_dir = state.get("run_dir")
            relays = state.get("relays")
            generation_id = state.get("generation_id", "")
            if (
                not isinstance(run_dir, str)
                or not Path(run_dir).is_absolute()
                or not isinstance(relays, list)
                or not relays
                or any(not isinstance(relay, str) or not relay for relay in relays)
                or len(set(relays)) != len(relays)
                or not isinstance(generation_id, str)
            ):
                return self._progress_monitor.observe(None, None, 0)
            run_key = (run_dir, generation_id, *relays)
            payload = _read_latest_progress_json(
                Path(run_dir) / "inference" / "live_stats.jsonl"
            )
            return self._progress_monitor.observe(run_key, payload, len(relays))
        except (OSError, TypeError, ValueError):
            return self._progress_monitor.observe(None, None, 0)

    def _load_identities(
        self,
    ) -> tuple[dict[str, ProcessIdentity], bool, tuple[str, ...]]:
        try:
            state = self._state_store.load() or {}
            raw_processes = state.get("processes", {})
            if not isinstance(raw_processes, dict):
                return {}, False, ()
            identities: dict[str, ProcessIdentity] = {}
            for name in ("mediamtx", "deepstream", "services"):
                if name not in raw_processes:
                    continue
                value = raw_processes[name]
                if not isinstance(value, dict):
                    return {}, False, ()
                identities[name] = ProcessIdentity.from_value(value)
            return identities, True, _runtime_run_key(state)
        except (OSError, TypeError, ValueError):
            return {}, False, ()

    def _probe_docker(
        self, deepstream: ProcessIdentity | None
    ) -> tuple[bool, bool, bool, tuple[int, ...]]:
        owner_token = deepstream.owner_token if deepstream is not None else None
        all_ready, all_confident, all_container_ids = self._query_docker_ids(
            ["docker", "ps", "-q", "--filter", "label=jiankong.owner"]
        )
        if not all_confident:
            return all_ready, False, False, ()
        if not owner_token:
            return (
                all_ready,
                False,
                not all_container_ids,
                (),
            )

        current_ready, current_confident, current_container_ids = (
            self._query_docker_ids(
                [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"label=jiankong.owner={owner_token}",
                ]
            )
        )
        docker_ready = all_ready and current_ready
        if not current_confident:
            return docker_ready, False, False, ()
        if set(all_container_ids) != set(current_container_ids):
            return docker_ready, False, False, ()
        if not current_container_ids:
            return docker_ready, False, True, ()

        try:
            inspect_result = self._docker_runner(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Pid}}",
                    *current_container_ids,
                ]
            )
            if inspect_result.returncode != 0:
                return docker_ready, False, False, ()
        except Exception:
            return docker_ready, False, False, ()
        try:
            raw_pids = tuple((inspect_result.stdout or "").splitlines())
            if len(raw_pids) != len(current_container_ids):
                return docker_ready, False, False, ()
            owned_pids = tuple(
                int(line.strip())
                for line in raw_pids
                if line.strip().isdigit() and int(line.strip()) > 0
            )
            if (
                len(owned_pids) != len(current_container_ids)
                or len(set(owned_pids)) != len(owned_pids)
            ):
                return docker_ready, False, False, ()
        except (AttributeError, TypeError, ValueError):
            return docker_ready, False, False, ()
        return docker_ready, True, True, tuple(sorted(owned_pids))

    def _query_docker_ids(
        self, command: list[str]
    ) -> tuple[bool, bool, tuple[str, ...]]:
        try:
            result = self._docker_runner(command)
            if result.returncode != 0:
                return False, False, ()
            container_ids = tuple(
                line.strip()
                for line in (result.stdout or "").splitlines()
                if line.strip()
            )
        except Exception:
            return False, False, ()
        if len(set(container_ids)) != len(container_ids) or any(
            re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None
            for container_id in container_ids
        ):
            return True, False, ()
        return True, True, container_ids

    def _safe_gpu_lock_probe(self) -> bool | None:
        try:
            value = self._gpu_lock_probe()
        except Exception:
            return None
        return value if value in (True, False, None) else None

    def _probe_gpu_compute_pids(self) -> tuple[int, ...] | None:
        try:
            result = self._gpu_runner(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ]
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        try:
            return tuple(
                sorted(
                    {
                        int(line.strip())
                        for line in (result.stdout or "").splitlines()
                        if line.strip()
                    }
                )
            )
        except ValueError:
            return None

    @staticmethod
    def _unknown_processes(
        scanned: ProcessScan,
        identities: dict[str, ProcessIdentity],
        identity_alive: dict[str, bool],
        owned_pids: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        categories = {
            "mediamtx": ("mediamtx",),
            "deepstream": ("pipeline", "launcher"),
            "services": ("services",),
        }
        result: dict[str, tuple[int, ...]] = {}
        for component, component_categories in categories.items():
            found = {
                pid
                for category in component_categories
                for pid in scanned.get(category, ())
                if isinstance(pid, int) and pid > 0
            }
            identity = identities.get(component)
            if identity is not None and identity_alive[component]:
                found.discard(identity.pid)
            found.difference_update(owned_pids[component])
            result[component] = tuple(sorted(found))
        return result


class WatchdogController:
    """Execute only the recovery action allowed by the latest safe snapshot."""

    def __init__(
        self,
        *,
        state_store: StateStore,
        probe: WatchdogProbe,
        release_root: Path,
        health_path: Path = HEALTH_PATH,
        watchdog_lock_path: Path = WATCHDOG_LOCK_PATH,
        check_interval: float = CHECK_INTERVAL_SECONDS,
        full_start: Callable[[], None] | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
        popen: Callable[..., Any] = subprocess.Popen,
        capture_identity: Callable[..., ProcessIdentity] = capture_identity,
        services_readiness: Callable[
            [Any, ProcessIdentity, Path, str], bool
        ]
        | None = None,
        stop_identity: Callable[[ProcessIdentity], None] = stop_process,
        same_process: Callable[[ProcessIdentity], bool] = is_same_process,
        owned_group_pids: Callable[
            [ProcessIdentity], tuple[int, ...] | None
        ] = lambda identity: owned_process_group_pids(identity),
        process_scanner: Callable[[], ProcessScan] = lambda: scan_runtime_processes(),
        port_probe: Callable[[int], bool] = lambda port: probe_port(port),
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.state_store = state_store
        self._storage_bindings: dict[Path, RunStorage] = {}
        self._probe = probe
        self._release_root = Path(release_root)
        self._health_path = Path(health_path)
        self._watchdog_lock_path = Path(watchdog_lock_path)
        if (
            isinstance(check_interval, bool)
            or not isinstance(check_interval, (int, float))
            or not math.isfinite(float(check_interval))
            or float(check_interval) <= 0
        ):
            raise ValueError("check_interval must be positive and finite")
        self._check_interval = float(check_interval)
        self._command_runner = command_runner
        self._full_start = full_start or self._run_full_start
        self._popen = popen
        self._capture_identity = capture_identity
        self._services_readiness = (
            services_readiness or self._wait_for_services_ready
        )
        self._stop_identity = stop_identity
        self._same_process = same_process
        self._owned_group_pids = owned_group_pids
        self._process_scanner = process_scanner
        self._port_probe = port_probe
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._sleep = sleep
        self._consecutive_failures = 0
        self._healthy_checks = 0
        self._next_retry_at = 0.0
        self._exit_event = threading.Event()
        self._health_publish_failed = False
        self._unknown_process_signature: tuple[tuple[str, tuple[int, ...]], ...] = ()
        self._unknown_process_observations = 0
        self._health: dict[str, Any] = {
            "state": "offline",
            "reason_code": "not_checked",
            "message": "The watchdog has not completed a check.",
            "last_action": "none",
            "retry_after": 0,
            "consecutive_failures": 0,
            "healthy_checks": 0,
            "updated_at": self._wall_time(),
        }

    def health(self) -> dict[str, Any]:
        """Return a detached, non-secret health summary."""
        return dict(self._health)

    def _confirm_unknown_processes(
        self, snapshot: RuntimeSnapshot
    ) -> tuple[RuntimeSnapshot, bool]:
        """Require a repeated identical scan before surfacing an unknown process.

        A process may disappear while /proc is being scanned (or be a short-lived
        diagnostic helper).  It must not create a visible production incident on
        one isolated watchdog pass.  A persistent, unchanged runtime process is
        still blocked before any recovery action is allowed.
        """
        components = (
            ("mediamtx", snapshot.mediamtx),
            ("deepstream", snapshot.deepstream),
            ("services", snapshot.services),
        )
        signature = tuple(
            (name, component.unknown_processes)
            for name, component in components
            if component.unknown_processes
        )
        if not signature:
            self._unknown_process_signature = ()
            self._unknown_process_observations = 0
            return snapshot, False
        if signature == self._unknown_process_signature:
            self._unknown_process_observations += 1
        else:
            self._unknown_process_signature = signature
            self._unknown_process_observations = 1
        if self._unknown_process_observations >= UNKNOWN_PROCESS_CONFIRMATIONS:
            return snapshot, True
        return (
            replace(
                snapshot,
                mediamtx=replace(snapshot.mediamtx, unknown_processes=()),
                deepstream=replace(snapshot.deepstream, unknown_processes=()),
                services=replace(snapshot.services, unknown_processes=()),
            ),
            False,
        )

    @property
    def exit_requested(self) -> bool:
        return self._exit_event.is_set()

    def request_exit(self, _signum: int, _frame: Any) -> None:
        """Request a clean exit without interrupting an active recovery finally."""
        self._exit_event.set()

    def check_once(self) -> RecoveryDecision:
        if self._health_publish_failed:
            paused = _blocked(
                "health_publish_failed",
                "Health publication is unavailable; recovery remains paused.",
            )
            self._publish_health(paused)
            return paused

        now = self._monotonic()
        retry_allowed = now >= self._next_retry_at
        raw_snapshot = self._probe.snapshot()
        snapshot, unknown_confirmed = self._confirm_unknown_processes(raw_snapshot)
        decision = decide_recovery(snapshot, retry_allowed=retry_allowed)
        if (
            not unknown_confirmed
            and _has_unknown_process(raw_snapshot)
            and decision.action != "none"
        ):
            decision = _blocked(
                "unknown_process_observing",
                "An unowned runtime process is awaiting confirmation; recovery is paused.",
            )
        if decision.reason_code in {"inference_progress_stalled", "worker_stalled"} and decision.action in {"restart_full", "restart_services"}:
            fresh_snapshot = self._probe.snapshot()
            reprobed = decide_recovery(fresh_snapshot, retry_allowed=retry_allowed)
            if (
                not snapshot.runtime_identity_key
                or snapshot.runtime_identity_key
                != fresh_snapshot.runtime_identity_key
                or (
                    decision.reason_code == "worker_stalled"
                    and (
                        not snapshot.runtime_run_key
                        or snapshot.runtime_run_key != fresh_snapshot.runtime_run_key
                    )
                )
            ):
                decision = _blocked(
                    "runtime_identity_changed_during_progress_reprobe",
                    "Runtime identity changed during inference progress re-probe; recovery is deferred.",
                )
            elif (
                reprobed.action == decision.action
                and reprobed.reason_code == decision.reason_code
            ):
                decision = reprobed
            elif reprobed.action in {"none", "blocked"}:
                decision = reprobed
            else:
                decision = _blocked(
                    "runtime_changed_during_progress_reprobe",
                    "Runtime changed during inference progress re-probe; recovery is deferred.",
                )
        if decision.action == "none":
            self._healthy_checks += 1
            if self._healthy_checks >= 2:
                self._consecutive_failures = 0
                self._next_retry_at = 0.0
            self._publish_health(decision, last_action="none")
            return decision

        self._healthy_checks = 0
        if decision.action == "blocked":
            self._publish_health(decision)
            return decision

        try:
            if decision.action == "start_full":
                self._full_start()
            elif decision.action == "restart_full":
                self._restart_full()
            else:
                self._restart_services(
                    restart_unresponsive=decision.reason_code == "worker_stalled",
                    expected_runtime_identity_key=snapshot.runtime_identity_key,
                    expected_runtime_run_key=snapshot.runtime_run_key,
                )
        except Exception:
            self._record_failure(decision.action)
        else:
            self._publish_health(
                decision, last_action=f"{decision.action}_succeeded"
            )
        return decision

    def run_forever(self) -> None:
        """Run checks while holding the singleton watchdog lock."""
        import fcntl

        self._watchdog_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._watchdog_lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another live watchdog is already running") from error
            previous_handlers = {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGTERM, signal.SIGINT)
            }
            try:
                for signum in previous_handlers:
                    signal.signal(signum, self.request_exit)
                while not self.exit_requested:
                    self.check_once()
                    if not self.exit_requested:
                        self._exit_event.wait(self._check_interval)
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _run_full_start(self) -> None:
        self._command_runner(
            [str(self._release_root / "scripts" / "jiankong-start")],
            check=True,
            timeout=180,
        )

    def _restart_full(self) -> None:
        """Clean known residual components before a single guarded full start."""
        self._command_runner(
            [str(self._release_root / "scripts" / "jiankong-stop")],
            check=True,
            timeout=180,
        )
        self._full_start()

    def _restart_services(
        self, *, restart_unresponsive: bool = False,
        expected_runtime_identity_key: str = "",
        expected_runtime_run_key: tuple[str, ...] = (),
    ) -> None:
        new_identity: ProcessIdentity | None = None
        committed = False
        with self.state_store.lock():
            before = self._require_running_state()
            if restart_unresponsive:
                current_identities = {
                    name: ProcessIdentity.from_value(before["processes"][name])
                    for name in ("mediamtx", "deepstream", "services")
                }
                if (
                    not expected_runtime_identity_key
                    or _runtime_identity_key(current_identities) != expected_runtime_identity_key
                    or not expected_runtime_run_key
                    or _runtime_run_key(before) != expected_runtime_run_key
                ):
                    raise RuntimeError("runtime identity or run changed before services recovery")
                identity = ProcessIdentity.from_value(before["processes"]["services"])
                # Recheck under the lifecycle lock: the worker may have recovered
                # since the snapshot. Never stop a new owner or a healthy worker.
                if not self._same_process(identity) or _worker_health(
                    _bound_run_storage(str(before["run_dir"]), self._storage_bindings).metadata_dir / "worker_status.json",
                    identity.owner_token, now=self._wall_time(),
                ) != "stale":
                    raise RuntimeError("worker recovered or service identity changed")
                self._stop_identity(identity)
            self._require_services_start_safe(before)
            before_hash = _state_hash_without_services(before)
            run_dir = Path(str(before["run_dir"]))
            owner_token = secrets.token_hex(16)
            environment = os.environ.copy()
            inherited_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = str(self._release_root) + (
                os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
            )
            environment["JIAN_KONG_OWNER_TOKEN"] = owner_token
            process = self._popen(
                [
                    sys.executable,
                    "-m",
                    "live_operator.cli",
                    "_services",
                    "--run-dir",
                    str(run_dir),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            try:
                new_identity = self._capture_identity(
                    process, owner_token=owner_token
                )
                if self.exit_requested:
                    raise RuntimeError("watchdog exit requested")
                if not self._services_readiness(
                    process, new_identity, run_dir, owner_token
                ):
                    raise RuntimeError("services readiness timed out")
                if (
                    self.exit_requested
                    or not _process_is_running(process)
                    or not self._same_process(new_identity)
                ):
                    raise RuntimeError("new services identity is no longer running")
                current = self._require_running_state()
                if _state_hash_without_services(current) != before_hash:
                    raise RuntimeError("operator state changed during services recovery")
                current["processes"]["services"] = new_identity.to_dict()
                if _state_hash_without_services(current) != before_hash:
                    raise RuntimeError("non-services state changed before commit")
                if (
                    self.exit_requested
                    or not _process_is_running(process)
                    or not self._same_process(new_identity)
                ):
                    raise RuntimeError("new services identity changed before commit")
                self.state_store.save(current)
                committed = True
            finally:
                if new_identity is not None and not committed:
                    self._stop_identity(new_identity)

    def _require_running_state(self) -> dict[str, Any]:
        state = self.state_store.load()
        if not isinstance(state, dict):
            raise RuntimeError("operator state is unavailable")
        if state.get("state") != "running" or not state.get("run_dir"):
            raise RuntimeError("operator state is not running")
        processes = state.get("processes")
        if not isinstance(processes, dict):
            raise RuntimeError("operator process state is invalid")
        for name in ("mediamtx", "deepstream"):
            raw_identity = processes.get(name)
            if not isinstance(raw_identity, dict):
                raise RuntimeError(f"{name} identity is unavailable")
            if not self._same_process(ProcessIdentity.from_value(raw_identity)):
                raise RuntimeError(f"{name} identity changed")
        return json.loads(json.dumps(state))

    def _require_services_start_safe(self, state: dict[str, Any]) -> None:
        processes = state["processes"]
        raw_services = processes.get("services")
        if raw_services is not None:
            if not isinstance(raw_services, dict):
                raise RuntimeError("services identity is invalid")
            services = ProcessIdentity.from_value(raw_services)
            if self._same_process(services):
                raise RuntimeError("services identity is already running")
            group_pids = self._owned_group_pids(services)
            if not isinstance(group_pids, (tuple, list)) or any(
                not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
                for pid in group_pids
            ):
                raise RuntimeError("services owner process group is uncertain")
            if group_pids:
                raise RuntimeError("services owner process group is still running")

        scanned = self._process_scanner()
        if not isinstance(scanned, dict):
            raise RuntimeError("runtime process scan is unavailable")
        raw_service_pids = scanned.get("services", ())
        if not isinstance(raw_service_pids, (tuple, list)) or any(
            not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            for pid in raw_service_pids
        ):
            raise RuntimeError("services process scan is uncertain")
        if raw_service_pids:
            raise RuntimeError("an unknown services process is already running")

        if self._port_probe(8767) is not True:
            raise RuntimeError("services port is not available")

    def _wait_for_services_ready(
        self,
        process: Any,
        identity: ProcessIdentity,
        run_dir: Path,
        owner_token: str,
    ) -> bool:
        from urllib.request import ProxyHandler, Request, build_opener

        opener = build_opener(ProxyHandler({}))
        deadline = self._monotonic() + SERVICES_READY_TIMEOUT_SECONDS
        while self._monotonic() < deadline:
            if (
                self.exit_requested
                or not _process_is_running(process)
                or not self._same_process(identity)
            ):
                return False
            worker_ready = _matching_worker_status(
                _bound_run_storage(run_dir, self._storage_bindings).metadata_dir / "worker_status.json",
                owner_token,
                now=self._wall_time(),
            )
            http_ready = False
            if not probe_port(8767):
                try:
                    with opener.open(
                        Request("http://127.0.0.1:8767/api/status"), timeout=2
                    ) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                        http_ready = response.status == 200 and isinstance(
                            payload, dict
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    http_ready = False
            if worker_ready and http_ready:
                return True
            self._sleep(0.25)
        return False

    def _record_failure(self, action: str) -> None:
        self._consecutive_failures += 1
        delay = BACKOFF_SECONDS[
            min(self._consecutive_failures - 1, len(BACKOFF_SECONDS) - 1)
        ]
        self._next_retry_at = self._monotonic() + delay
        self._publish_health(
            RecoveryDecision(
                "blocked",
                "degraded",
                f"{action}_failed",
                "Recovery action failed; manual inspection may be required.",
            ),
            last_action=f"{action}_failed",
        )

    def _publish_health(
        self, decision: RecoveryDecision, *, last_action: str | None = None
    ) -> None:
        updated_at = self._wall_time()
        if decision.state in {"blocked", "degraded", "recovering"}:
            previous = self._health.get("last_incident")
            observed_at = updated_at
            if (
                isinstance(previous, dict)
                and previous.get("active") is True
                and previous.get("reason_code") == decision.reason_code
                and type(previous.get("observed_at")) in (int, float)
                and math.isfinite(previous["observed_at"])
                and previous["observed_at"] <= updated_at
            ):
                observed_at = previous["observed_at"]
            self._health["last_incident"] = {
                "active": True,
                "message": decision.message,
                "observed_at": observed_at,
                "reason_code": decision.reason_code,
            }
        elif decision.state == "healthy":
            incident = self._health.get("last_incident")
            if isinstance(incident, dict) and incident.get("active") is True:
                self._health["last_incident"] = {
                    **incident,
                    "active": False,
                    "resolved_at": updated_at,
                }
        self._health.update(
            {
                "state": decision.state,
                "reason_code": decision.reason_code,
                "message": decision.message,
                "retry_after": self._retry_after(),
                "consecutive_failures": self._consecutive_failures,
                "healthy_checks": self._healthy_checks,
                "updated_at": updated_at,
            }
        )
        if last_action is not None:
            self._health["last_action"] = last_action
        try:
            _atomic_json_write(self._health_path, self._health)
        except Exception:
            self._health_publish_failed = True
            print(HEALTH_PUBLISH_WARNING, file=sys.stderr, flush=True)
        else:
            self._health_publish_failed = False

    def _retry_after(self) -> int:
        if self._next_retry_at <= 0:
            return 0
        return max(0, math.ceil(self._next_retry_at - self._monotonic()))


def _read_latest_progress_json(
    path: Path, *, tail_bytes: int = 64 * 1024
) -> dict[str, Any] | None:
    """Read only the newest complete stats record; uncertainty resets progress."""
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - tail_bytes))
            payload = handle.read()
    except OSError:
        return None
    if not payload or not payload.endswith(b"\n"):
        return None
    lines = payload.splitlines()
    if not lines:
        return None
    try:
        value = json.loads(lines[-1].decode("utf-8"))
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _processed_counters(
    payload: dict[str, Any] | None, expected_streams: int
) -> tuple[int, ...] | None:
    if expected_streams < 1 or not isinstance(payload, dict):
        return None
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return None
    indexed: dict[int, int] = {}
    for stream in streams:
        if not isinstance(stream, dict):
            return None
        stream_index = stream.get("stream_index")
        processed = stream.get("processed")
        if (
            type(stream_index) is not int
            or type(processed) is not int
            or processed < 0
            or stream_index in indexed
        ):
            return None
        indexed[stream_index] = processed
    if set(indexed) != set(range(expected_streams)):
        return None
    return tuple(indexed[index] for index in range(expected_streams))


def decide_recovery(
    snapshot: RuntimeSnapshot, retry_allowed: bool = True
) -> RecoveryDecision:
    """Return the safest recovery action supported by a runtime snapshot."""
    if _has_unknown_process(snapshot):
        return _blocked("unknown_process", "An unowned runtime process was detected.")

    if _has_uncertain_component_probe(snapshot):
        return _blocked(
            "process_probe_unknown", "Runtime process status could not be determined."
        )

    if set(snapshot.gpu_compute_pids) - set(snapshot.gpu_owned_compute_pids):
        return _blocked("unknown_gpu_compute", "GPU compute activity is present.")

    if snapshot.gpu_lock_free is None:
        return _blocked("gpu_lock_unknown", "GPU lock status could not be determined.")

    if snapshot.inference_stalled_streams and _all_alive(snapshot):
        if not retry_allowed:
            return _blocked("retry_not_allowed", "Recovery retry is not currently allowed.")
        if _stalled_inference_restart_gates_safe(snapshot):
            return RecoveryDecision(
                "restart_full",
                "recovering",
                "inference_progress_stalled",
                "Inference stream progress stalled; restarting the owned runtime.",
            )
        return _blocked(
            "inference_stall_restart_gates",
            "Inference progress stalled but safe full-restart gates are not satisfied.",
        )

    if _all_alive(snapshot):
        if snapshot.worker_health == "stale":
            if not retry_allowed:
                return _blocked("retry_not_allowed", "Recovery retry is not currently allowed.")
            return RecoveryDecision(
                "restart_services", "recovering", "worker_stalled",
                "Event coordination heartbeat stalled; recovering supporting services only.",
            )
        if snapshot.worker_health != "healthy":
            return _blocked(
                f"worker_{snapshot.worker_health}",
                "Event coordination is not healthy; realtime inference is left running.",
            )
        return RecoveryDecision("none", "healthy", "healthy", "All components are healthy.")

    if _only_services_dead(snapshot) and snapshot.port_8767_free:
        if not retry_allowed:
            return _blocked("retry_not_allowed", "Recovery retry is not currently allowed.")
        return RecoveryDecision(
            "restart_services",
            "recovering",
            "services_only",
            "Only supporting services will be restarted.",
        )

    if _only_deepstream_dead(snapshot):
        if not retry_allowed:
            return _blocked("retry_not_allowed", "Recovery retry is not currently allowed.")
        if _missing_deepstream_restart_gates_safe(snapshot):
            return RecoveryDecision(
                "restart_full",
                "recovering",
                "deepstream_missing",
                "Inference stopped while owned support services remained; restarting the runtime.",
            )
        return _blocked(
            "deepstream_restart_gates",
            "Inference is missing and safe full-restart gates are not satisfied.",
        )

    if _all_dead(snapshot):
        if not retry_allowed:
            return _blocked("retry_not_allowed", "Recovery retry is not currently allowed.")
        if _full_start_gates_safe(snapshot):
            return RecoveryDecision(
                "start_full",
                "recovering",
                "full_start",
                "All components are down and full-start gates are safe.",
            )
        return _blocked("full_start_gates", "Full-start safety gates are not satisfied.")

    return RecoveryDecision(
        "blocked",
        "degraded",
        "partial_runtime",
        "The runtime is only partially available; recovery is alert-only.",
    )


def owned_gpu_compute_pids(
    gpu_compute_pids: tuple[int, ...],
    owned_pids: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    """Return GPU PIDs proven to belong to inference or support services."""
    trusted = set(owned_pids.get("deepstream", ()))
    trusted.update(owned_pids.get("services", ()))
    return tuple(pid for pid in gpu_compute_pids if pid in trusted)


def _runtime_run_key(state: dict[str, Any]) -> tuple[str, ...]:
    run_dir = state.get("run_dir")
    generation_id = state.get("generation_id", "")
    if (
        state.get("state") != "running"
        or not isinstance(run_dir, str)
        or not Path(run_dir).is_absolute()
        or not isinstance(generation_id, str)
    ):
        return ()
    return (run_dir, generation_id)


def _runtime_identity_key(identities: dict[str, ProcessIdentity]) -> str:
    if set(identities) != {"mediamtx", "deepstream", "services"}:
        return ""
    encoded = json.dumps(
        {
            name: identities[name].to_dict()
            for name in ("mediamtx", "deepstream", "services")
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_unknown_process(snapshot: RuntimeSnapshot) -> bool:
    return any(
        component.unknown_processes
        for component in (snapshot.mediamtx, snapshot.deepstream, snapshot.services)
    )


def _all_alive(snapshot: RuntimeSnapshot) -> bool:
    return all(
        component.identity_alive
        for component in (snapshot.mediamtx, snapshot.deepstream, snapshot.services)
    ) and snapshot.docker_ready and snapshot.network_ready


def _only_services_dead(snapshot: RuntimeSnapshot) -> bool:
    return (
        snapshot.mediamtx.identity_alive
        and snapshot.deepstream.identity_alive
        and not snapshot.services.identity_alive
        and not snapshot.services.owned_processes_alive
    )


def _only_deepstream_dead(snapshot: RuntimeSnapshot) -> bool:
    return (
        snapshot.mediamtx.identity_alive
        and not snapshot.deepstream.identity_alive
        and not snapshot.deepstream.owned_processes_alive
        and snapshot.services.identity_alive
    )


def _missing_deepstream_restart_gates_safe(snapshot: RuntimeSnapshot) -> bool:
    return (
        snapshot.gpu_lock_free is True
        and not snapshot.gpu_owned_by_current
        and not snapshot.gpu_compute_pids
        and snapshot.docker_ready
        and snapshot.network_ready
    )


def _stalled_inference_restart_gates_safe(snapshot: RuntimeSnapshot) -> bool:
    return (
        _all_alive(snapshot)
        and snapshot.gpu_lock_free is False
        and snapshot.gpu_owned_by_current
        and bool(snapshot.gpu_compute_pids)
        and set(snapshot.gpu_compute_pids).issubset(
            snapshot.gpu_owned_compute_pids
        )
    )


def _all_dead(snapshot: RuntimeSnapshot) -> bool:
    return not any(
        component.identity_alive or component.owned_processes_alive
        for component in (snapshot.mediamtx, snapshot.deepstream, snapshot.services)
    )


def _full_start_gates_safe(snapshot: RuntimeSnapshot) -> bool:
    return (
        snapshot.gpu_lock_free is True
        and not snapshot.gpu_owned_by_current
        and not snapshot.gpu_compute_pids
        and snapshot.docker_ready
        and snapshot.network_ready
        and snapshot.port_8767_free
    )


def _blocked(reason_code: str, message: str) -> RecoveryDecision:
    return RecoveryDecision("blocked", "blocked", reason_code, message)


def _has_uncertain_component_probe(snapshot: RuntimeSnapshot) -> bool:
    return not all(
        component.probe_confident
        for component in (snapshot.mediamtx, snapshot.deepstream, snapshot.services)
    )


def scan_runtime_processes(proc_root: Path = Path("/proc")) -> ProcessScan:
    """Classify production processes while discarding command-line bytes."""
    found: dict[str, list[int]] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        raise
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        category = _process_category(command)
        if category is not None:
            found.setdefault(category, []).append(int(entry.name))
    return {
        category: tuple(sorted(set(pids)))
        for category, pids in found.items()
    }


def _process_category(command: bytes) -> str | None:
    parts = tuple(part for part in command.split(b"\0") if part)
    if not parts:
        return None
    executable = parts[0].rsplit(b"/", 1)[-1]

    # Match real runtime entry points, not arbitrary text in another process's
    # arguments.  Tools such as ``pgrep -f`` include their search expression in
    # their own command line and previously appeared to be a second, unowned
    # JianKong process for one watchdog cycle.
    if executable == b"jiankong_custom_pipeline":
        return "pipeline"
    if executable in {b"docker", b"podman"} and any(
        part.rsplit(b"/", 1)[-1]
        in {b"jiankong_custom_pipeline", b"run_7x8.sh"}
        for part in parts[1:]
    ):
        return "pipeline"
    if (
        executable.startswith(b"python")
        and len(parts) >= 4
        and parts[1:4] == (b"-m", b"live_operator.cli", b"_services")
    ):
        return "services"
    if executable in {b"bash", b"sh"} and any(
        part.rsplit(b"/", 1)[-1]
        in {b"run_container_50p2.sh", b"run_7x8.sh"}
        for part in parts[1:]
    ):
        return "launcher"
    if executable == b"mediamtx":
        return "mediamtx"
    return None


def owned_process_group_pids(
    identity: ProcessIdentity, proc_root: Path = Path("/proc")
) -> tuple[int, ...] | None:
    """Return PIDs proven to share the identity's owner marker and process group."""
    if os.name != "posix" or identity.pgid is None or not identity.owner_token:
        return ()
    marker = f"JIAN_KONG_OWNER_TOKEN={identity.owner_token}".encode()
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return None
    result: list[int] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != identity.pgid:
                continue
            environment = (entry / "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return None
        if marker in environment.split(b"\0"):
            result.append(pid)
    return tuple(sorted(set(result)))


def probe_gpu_lock(lock_path: Path = GPU_LOCK_PATH) -> bool | None:
    """Try the production GPU lock without blocking and release it if acquired."""
    import fcntl

    try:
        handle = lock_path.open("a+b")
    except OSError:
        return None
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            return True
        except BlockingIOError:
            return False
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                return False
            return None
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def probe_port(port: int) -> bool:
    """Return whether the loopback TCP port can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def probe_network() -> bool:
    """Ask NetworkManager whether the host is currently online."""
    result = _run_command(["nm-online", "-q", "--timeout=0"])
    return result.returncode == 0


def _run_command(command: list[str]) -> CommandResult:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _state_hash_without_services(state: dict[str, Any]) -> str:
    value = json.loads(json.dumps(state))
    processes = value.get("processes")
    if isinstance(processes, dict):
        processes.pop("services", None)
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _process_is_running(process: Any) -> bool:
    try:
        return process.poll() is None
    except Exception:
        return False


def _worker_health(
    path: Path, owner_token: str, *, now: float, stale_after: float = WORKER_STALE_SECONDS
) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return "unknown"
        updated_at = value.get("updated_at")
        if (
            not isinstance(owner_token, str) or not re.fullmatch(r"[0-9a-f]{32}", owner_token)
            or value.get("service_instance") != owner_token
            or type(value.get("accepting")) is not bool
            or type(value.get("active")) is not int or value["active"] < 0
            or type(value.get("healthy", True)) is not bool
            or type(updated_at) not in (int, float)
            or not math.isfinite(updated_at) or not math.isfinite(now)
            or now < updated_at
        ):
            return "unknown"
        if now - updated_at > stale_after:
            return "stale"
        if value["accepting"] is False:
            return "draining"
        return "healthy" if value.get("healthy", True) else "degraded"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "unknown"


def _matching_worker_status(path: Path, owner_token: str, *, now: float) -> bool:
    return _worker_health(path, owner_token, now=now, stale_after=2.0) == "healthy"


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _positive_finite_interval(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "interval must be a positive finite number"
        ) from error
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError(
            "interval must be a positive finite number"
        )
    return interval


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jiankong-watchdog")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--health", type=Path, default=HEALTH_PATH)
    parser.add_argument("--lock", type=Path, default=WATCHDOG_LOCK_PATH)
    parser.add_argument(
        "--interval", type=_positive_finite_interval, default=CHECK_INTERVAL_SECONDS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = StateStore(args.state)
    controller = WatchdogController(
        state_store=store,
        probe=WatchdogProbe(state_store=store),
        release_root=args.release_root,
        health_path=args.health,
        watchdog_lock_path=args.lock,
        check_interval=args.interval,
    )
    controller.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
