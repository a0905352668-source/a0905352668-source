"""One-command operator lifecycle orchestration."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from live_operator.config import LiveConfig, make_legacy_config, redact_text
from live_operator.clips import (
    POST_EVENT_SECONDS,
    ClipQueueFull,
    ClipRemuxWorker,
    ClipRequest,
    refresh_clip_overlay,
)
from live_operator.events import EventCoordinator, write_source_generation
from live_operator.storage_paths import RunStorage, initialize_run_storage, resolve_run_storage
from live_operator.inference import resolve_current_inference_version
from live_operator.mediamtx import render_mediamtx_config
from live_operator.processes import (
    ProcessIdentity,
    StateStore,
    capture_identity,
    is_same_process,
    owned_process_group_exists,
    stop_process,
)
from live_operator.vlm_review import (
    DEFAULT_VLM_REVIEW_CONFIG,
    VLMReviewClient,
    VLMReviewConfig,
    VLMReviewWorker,
    evidence_request_id,
)
from live_operator.vlm_state import (
    DEFAULT_VLM_STATE_FILENAME,
    VLMReviewStateStore,
)


class LifecycleError(RuntimeError):
    pass


_LEGACY_STATUS_SLOTS = (
    ("camera01", "dianqi1"),
    ("camera02", "dianqi2"),
    ("camera03", "jixie1"),
    ("camera04", "jixie2"),
    ("camera05", "ruanjian1"),
    ("camera06", "ruanjian2"),
    ("camera07", "zoulang"),
)


class LifecycleManager:
    COMPONENTS = ("mediamtx", "services", "deepstream")

    def __init__(
        self,
        store: StateStore,
        hooks: Any,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        generation_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.store = store
        self.hooks = hooks
        self.now = now
        self.generation_id = generation_id
        self._storage_bindings: dict[Path, RunStorage] = {}

    def _storage_for_run(self, run_dir: Path) -> RunStorage:
        storage = resolve_run_storage(run_dir)
        known = self._storage_bindings.setdefault(storage.run_dir, storage)
        if known != storage:
            raise ValueError("lifecycle metadata authority changed")
        return storage

    def start(self, config_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
        with self.store.lock():
            return self._start_locked(config_path, run_dir)

    def _start_locked(self, config_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
        existing = self.store.load()
        if existing and existing.get("state") in {"starting", "running"}:
            records = self._records(existing)
            if any(self.hooks.is_same_process(identity) for identity in records.values()):
                raise LifecycleError("live operator is already running")

        run_path = Path(run_dir)
        started: dict[str, ProcessIdentity] = {}
        state: dict[str, Any] = {
            "state": "starting",
            "run_dir": str(run_path),
            "processes": {},
        }
        self.store.save(state)
        try:
            config = self.hooks.validate_config(Path(config_path))
            sources = self._camera_sequence(self.hooks.probe_sources(config), "source")
            self.hooks.check_ports()
            run_path.mkdir(parents=True, exist_ok=True)
            storage = resolve_run_storage(run_path)
            if os.environ.get("JIANKONG_HOT_METADATA_ENABLED") == "1" and not storage.hot:
                # A failed preflight may leave only the initializer's lock.
                if any(path.name != ".storage_layout.lock" for path in run_path.iterdir()):
                    raise LifecycleError("hot metadata opt-in requires a new empty run")
                storage = initialize_run_storage(run_path)
            self._storage_for_run(run_path)
            started["mediamtx"] = self.hooks.start_component(
                "mediamtx", run_path, config
            )
            relays = self._camera_sequence(self.hooks.probe_relays(), "relay")
            self.hooks.gpu_lock_available()
            generation_id = self.generation_id()
            started_at = self.now()
            if started_at.tzinfo is None or started_at.utcoffset() is None:
                raise LifecycleError("generation started_at must be timezone-aware")
            generation_file = run_path / "inference" / "source_generation.json"
            self.hooks.write_generation(
                generation_file, generation_id, started_at
            )
            started["deepstream"] = self.hooks.start_component(
                "deepstream", run_path, config
            )
            state["processes"] = self._serialize(started)
            self.store.save(state)
            self.hooks.verify_runtime(started["deepstream"], run_path)
            started["services"] = self.hooks.start_component(
                "services", run_path, config
            )
            state["processes"] = self._serialize(started)
            self.store.save(state)
            self.hooks.http_self_check()
            state.update(
                {
                    "state": "running",
                    "started_at": started_at.isoformat(),
                    "generation_id": generation_id,
                    "generation_file": str(generation_file),
                    "sources": sources,
                    "relays": relays,
                    "dashboard_url": "http://192.168.50.2:8767/",
                    "target_fps_per_stream": getattr(self.hooks, "infer_fps", 8),
                    "processes": self._serialize(started),
                }
            )
            self.store.save(state)
            return self._public(state)
        except Exception as error:
            for name in reversed(self.COMPONENTS):
                identity = started.get(name)
                if identity is not None and (
                    self.hooks.is_same_process(identity)
                    or self.hooks.has_owned_processes(name, identity)
                ):
                    self.hooks.stop_component(name, identity)
            state.update(
                {
                    "state": "failed",
                    "error": redact_text(str(error)),
                    "processes": self._serialize(started),
                }
            )
            if "generation_id" not in state and "generation_id" in locals():
                state["generation_id"] = generation_id
                state["generation_file"] = str(generation_file)
            self.store.save(state)
            raise LifecycleError(redact_text(str(error))) from error

    def status(self) -> dict[str, Any]:
        with self.store.lock():
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        state = self.store.load()
        if state is None:
            return {"state": "stopped", "processes": {}}
        records = self._records(state)
        live = {
            name: self.hooks.is_same_process(identity)
            for name, identity in records.items()
        }
        public = self._public(state)
        public["process_alive"] = live
        if state.get("state") == "running" and not all(live.values()):
            public["state"] = "stale"
        return public

    def stop(self) -> dict[str, Any]:
        with self.store.lock():
            return self._stop_locked()

    def _stop_locked(self) -> dict[str, Any]:
        state = self.store.load()
        if state is None:
            return {"state": "stopped"}
        if hasattr(self.hooks, "run_dir") and state.get("run_dir"):
            self.hooks.run_dir = Path(state["run_dir"])
        records = self._records(state)
        warnings: list[str] = []
        fatal_errors: list[str] = []
        stop_storage = None
        if state.get("run_dir"):
            try:
                # Capture before the handshake, even without service identity.
                stop_storage = self._storage_for_run(Path(state["run_dir"]))
            except (OSError, ValueError) as error:
                fatal_errors.append(f"stop metadata unavailable: {redact_text(str(error))}")
        forced = False
        services = records.get("services")
        if services is None or not services.owner_token:
            forced = True
            warnings.append("service identity is missing; forcing stop")
        else:
            try:
                requested_at = self.hooks.block_new_events(services.owner_token)
                self.hooks.drain_clips(8.0, services.owner_token, requested_at)
            except Exception as error:
                forced = True
                warnings.append(redact_text(str(error)))
        state.update({"state": "stopping", "forced": forced, "stop_errors": warnings})
        self.store.save(state)
        for name in reversed(self.COMPONENTS):
            identity = records.get(name)
            if identity is None:
                continue
            try:
                parent_alive = self.hooks.is_same_process(identity)
                owned_alive = self.hooks.has_owned_processes(name, identity)
                if parent_alive or owned_alive:
                    self.hooks.stop_component(name, identity)
            except Exception as error:
                fatal_errors.append(f"{name}: {redact_text(str(error))}")
        lingering: list[str] = []
        for name, identity in records.items():
            try:
                parent_alive = self.hooks.is_same_process(identity)
                owned_alive = self.hooks.has_owned_processes(name, identity)
                if parent_alive or owned_alive:
                    lingering.append(name)
            except Exception as error:
                lingering.append(name)
                fatal_errors.append(f"{name} verification: {redact_text(str(error))}")
        all_errors = warnings + fatal_errors
        if lingering or fatal_errors:
            state["state"] = "stop_failed"
            details = []
            if lingering:
                details.append(f"owned processes still running: {','.join(sorted(lingering))}")
            details.extend(all_errors)
            state["error"] = "; ".join(details)
            state["stop_errors"] = all_errors
            self.store.save(state)
            if stop_storage is not None:
                try:
                    self._record_stop(
                        state, forced=forced, errors=all_errors, lingering=lingering,
                        failed=True, storage=stop_storage,
                    )
                except (OSError, ValueError) as error:
                    message = f"stop record failed: {redact_text(str(error))}"
                    state["error"] += f"; {message}"
                    state["stop_errors"].append(message)
                    self.store.save(state)
            raise LifecycleError(state["error"])
        result = {"state": "stopped", "forced": forced}
        if warnings:
            result["warnings"] = warnings
        try:
            self._record_stop(
                state, forced=forced, errors=warnings, lingering=[], failed=False,
                storage=stop_storage,
            )
        except (OSError, ValueError) as error:
            message = f"stop record failed: {redact_text(str(error))}"
            state.update(state="stop_failed", error=message, stop_errors=[*warnings, message])
            self.store.save(state)
            raise LifecycleError(message) from error
        self.store.remove()
        return result

    @staticmethod
    def _record_stop(
        state: dict[str, Any], *, forced: bool, errors: list[str], lingering: list[str],
        failed: bool,
        storage: RunStorage | None = None,
    ) -> None:
        run_dir = state.get("run_dir")
        if not run_dir:
            return
        current = resolve_run_storage(str(run_dir))
        if storage is not None and current != storage:
            raise ValueError("stop metadata authority changed")
        destination = current.metadata_dir / "operator_stop.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "state": "stop_failed" if failed else "stopped",
                    "forced": forced,
                    "errors": errors,
                    "lingering": sorted(lingering),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    @staticmethod
    def _camera_sequence(values: Any, label: str) -> list[str]:
        result = [str(value) for value in values]
        if not result or len(set(result)) != len(result):
            raise LifecycleError(f"invalid {label} camera list")
        return result

    @staticmethod
    def _serialize(values: dict[str, ProcessIdentity]) -> dict[str, Any]:
        return {name: identity.to_dict() for name, identity in values.items()}

    @staticmethod
    def _records(state: dict[str, Any]) -> dict[str, ProcessIdentity]:
        raw = state.get("processes", {})
        return {
            str(name): ProcessIdentity.from_value(value)
            for name, value in raw.items()
            if isinstance(value, dict)
        }

    @staticmethod
    def _public(state: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "state", "run_dir", "started_at", "generation_id", "sources",
            "relays", "dashboard_url", "target_fps_per_stream", "processes", "error",
        }
        return {key: state[key] for key in allowed if key in state}


def configure(path: str | Path, *, input_fn=input, password_fn=getpass.getpass) -> Path:
    username = input_fn("Camera username: ").strip()
    password = password_fn("Camera password: ")
    camera_ips: dict[str, str] = {}
    for _relay, view, _calibration, _has_screen in (
        ("camera01", "dianqi1", "camera_01_screen_calibration_v21.json", False),
        ("camera02", "dianqi2", "camera_02_screen_calibration_v21.json", True),
        ("camera03", "jixie1", "camera_mechanical_01_screen_calibration_v21.json", True),
        ("camera04", "jixie2", "camera_mechanical_02_screen_calibration_v21.json", True),
        ("camera05", "ruanjian1", "camera_software_01_screen_calibration_v21.json", True),
        ("camera06", "ruanjian2", "camera_software_02_screen_calibration_v21.json", True),
        ("camera07", "zoulang", "camera_corridor_screen_calibration_v21.json", True),
    ):
        ip = input_fn(f"{view} IP: ").strip()
        camera_ips[view] = ip
    config = make_legacy_config(username, password, camera_ips)
    config.save(path)
    return Path(path)


_OWNER_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")


def _require_owner_token(token: str | None) -> str:
    if token is None or _OWNER_TOKEN_RE.fullmatch(token) is None:
        raise LifecycleError("invalid owner token")
    return token


def _has_stream_runtime_progress(
    payload: dict[str, Any] | None, expected_streams: int
) -> bool:
    if expected_streams < 1:
        return False
    if not isinstance(payload, dict):
        return False
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return False
    indexed: dict[int, dict[str, Any]] = {}
    for stream in streams:
        if not isinstance(stream, dict) or type(stream.get("stream_index")) is not int:
            continue
        indexed[int(stream["stream_index"])] = stream
    if set(indexed) != set(range(expected_streams)):
        return False
    try:
        return all(
            float(indexed[index].get("processed", 0)) > 0
            for index in range(expected_streams)
        )
    except (TypeError, ValueError):
        return False


def _has_seven_stream_runtime_progress(payload: dict[str, Any] | None) -> bool:
    """Compatibility wrapper for existing seven-camera diagnostics/tests."""

    return _has_stream_runtime_progress(payload, 7)


class RuntimeHooks:
    """50.2 runtime adapter; child argv and state contain no camera credentials."""

    def __init__(
        self,
        *,
        probe_runner: Callable[..., Any] = subprocess.run,
        probe_timeout: float = 12.0,
        relay_ready_timeout: float = 45.0,
        runtime_ready_timeout: float = 60.0,
        http_ready_timeout: float = 30.0,
    ) -> None:
        self.config: LiveConfig | None = None
        self.config_path: Path | None = None
        self._active_relays: tuple[str, ...] = ()
        self.run_dir: Path | None = None
        self._storage_bindings: dict[Path, RunStorage] = {}
        self.probe_runner = probe_runner
        self.probe_timeout = probe_timeout
        self.relay_ready_timeout = relay_ready_timeout
        self.runtime_ready_timeout = runtime_ready_timeout
        self.http_ready_timeout = http_ready_timeout
        self.infer_fps = 8

    def validate_config(self, path: Path) -> LiveConfig:
        self.config = LiveConfig.load(path)
        self.config_path = Path(path)
        self.infer_fps = resolve_current_inference_version().infer_fps
        return self.config

    def _probe(self, url: str) -> bool:
        descriptor = -1
        temporary: Path | None = None
        try:
            if "\n" in url or "\r" in url or "'" in url:
                return False
            content = f"ffconcat version 1.0\nfile '{url}'\n".encode("utf-8")
            if os.name == "posix" and hasattr(os, "memfd_create"):
                descriptor = os.memfd_create("jiankong-ffprobe", flags=0)
                os.fchmod(descriptor, 0o600)
                os.write(descriptor, content)
                os.lseek(descriptor, 0, os.SEEK_SET)
                input_path = f"/proc/self/fd/{descriptor}"
            else:
                descriptor, name = tempfile.mkstemp(prefix=".jiankong-probe-", suffix=".ffconcat")
                temporary = Path(name)
                os.write(descriptor, content)
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = -1
                os.chmod(temporary, 0o600)
                input_path = str(temporary)
            command = [
                "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
                "-f", "concat", "-safe", "0",
                "-protocol_whitelist", "file,pipe,rtsp,tcp,udp,rtp",
                "-i", input_path,
                "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate",
                "-of", "json",
            ]
            kwargs = {
                "capture_output": True,
                "text": True,
                "timeout": self.probe_timeout,
                "check": False,
            }
            if descriptor >= 0:
                kwargs["pass_fds"] = (descriptor,)
            completed = self.probe_runner(command, **kwargs)
            if completed.returncode != 0:
                return False
            try:
                payload = json.loads(completed.stdout)
                streams = payload.get("streams", [])
                video = next(item for item in streams if item.get("codec_type") == "video")
                rate = Fraction(str(video["r_frame_rate"]))
                return (
                    str(video.get("codec_name", "")).lower() in {"h264", "hevc"}
                    and int(video.get("width", 0)) == 2560
                    and int(video.get("height", 0)) == 1440
                    and rate > 0
                )
            except (StopIteration, KeyError, ValueError, ZeroDivisionError, TypeError):
                return False
        except (OSError, subprocess.SubprocessError):
            return False
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def probe_sources(self, config: LiveConfig) -> list[str]:
        cameras = config.enabled_cameras
        failed = [camera.relay for camera in cameras if not self._probe(camera.rtsp_url())]
        if failed:
            raise LifecycleError(f"source preflight failed: {','.join(failed)}")
        self._active_relays = tuple(camera.relay for camera in cameras)
        return list(self._active_relays)

    @staticmethod
    def check_ports() -> None:
        for port in (8554, 9997, 8767):
            with socket.socket() as listener:
                try:
                    listener.bind(("127.0.0.1", port))
                except OSError as error:
                    raise LifecycleError(f"port conflict: {port}") from error

    def start_component(self, name: str, run_dir: Path, config: LiveConfig) -> ProcessIdentity:
        self._storage_for_run(run_dir)
        self.run_dir = run_dir
        logs = run_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        release_root = Path(__file__).resolve().parents[1]
        if name == "mediamtx":
            config_path = render_mediamtx_config(config, run_dir)
            command = ["mediamtx", str(config_path)]
        elif name == "services":
            if self.config_path is None:
                raise LifecycleError("runtime configuration path unavailable")
            command = [
                sys.executable, "-m", "live_operator.cli", "_services",
                "--run-dir", str(run_dir), "--config", str(self.config_path),
            ]
        elif name == "deepstream":
            source_dir = release_root / "deepstream" / "custom_pipeline"
            launcher = source_dir / "scripts" / "run_container_50p2.sh"
            if not launcher.is_file():
                raise LifecycleError(f"DeepStream launcher not found: {launcher}")
            if not os.access(launcher, os.X_OK):
                raise LifecycleError(f"DeepStream launcher is not executable: {launcher}")
            version = resolve_current_inference_version()
            command = [str(launcher)]
        else:
            raise LifecycleError(f"unknown component: {name}")
        log = (logs / f"{name}.log").open("ab", buffering=0)
        environment = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH") if key in os.environ}
        if name == "services":
            environment["PYTHONPATH"] = str(release_root)
            if "JIAN_KONG_VLM_REVIEW_CONFIG" in os.environ:
                environment["JIAN_KONG_VLM_REVIEW_CONFIG"] = os.environ[
                    "JIAN_KONG_VLM_REVIEW_CONFIG"
                ]
        owner_token = _require_owner_token(uuid.uuid4().hex)
        environment["JIAN_KONG_OWNER_TOKEN"] = owner_token
        if name == "deepstream":
            cameras = config.enabled_cameras
            if not cameras:
                raise LifecycleError("no enabled cameras")
            camera_manifest = run_dir / "private" / "camera_manifest.json"
            camera_manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary = camera_manifest.with_name(f".{camera_manifest.name}.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_width": config.source_width,
                        "source_height": config.source_height,
                        "cameras": [
                            {
                                "relay": camera.relay,
                                "view": camera.view,
                                "calibration": camera.calibration,
                                "has_screen": camera.has_screen,
                            }
                            for camera in cameras
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, camera_manifest)
            os.chmod(camera_manifest, 0o600)
            environment.update(
                {
                    "SOURCE_DIR": str(source_dir),
                    "LIVE_OPERATOR_PYTHONPATH": str(release_root),
                    "LOCAL_RELAY_BASE": "rtsp://127.0.0.1:8554",
                    "OUTPUT_DIR": str(run_dir / "inference"),
                    "DEEPSTREAM_BINARY": str(version.deepstream_binary),
                    "DEEPSTREAM_BINARY_SHA256": version.deepstream_binary_sha256,
                    "POSE_ENGINE": str(version.pose_plan),
                    "PHONE_ENGINE": str(version.phone_engine),
                    "CALIB_DIR": str(version.calibration_dir),
                    "CAMERA_MANIFEST_FILE": str(camera_manifest),
                    "INFER_FPS": str(getattr(version, "infer_fps", 8)),
                    # Keep the production process long-lived. The watchdog
                    # still recovers an early exit, while this avoids a
                    # deliberate daily shutdown after the original 24-hour
                    # acceptance run.
                    "DURATION_SEC": "31536000",
                }
            )
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log if name != "mediamtx" else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        return capture_identity(process, owner_token=owner_token)

    def probe_relays(self) -> list[str]:
        relays = list(self._active_relays)
        if not relays:
            raise LifecycleError("no active relay configuration")
        pending = list(relays)
        deadline = time.monotonic() + self.relay_ready_timeout
        first_round = True
        while pending and (first_round or time.monotonic() < deadline):
            first_round = False
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                results = executor.map(
                    self._probe,
                    [f"rtsp://127.0.0.1:8554/{relay}" for relay in pending],
                )
                pending = [relay for relay, ready in zip(pending, results) if not ready]
            if not pending or time.monotonic() >= deadline:
                break
            remaining = deadline - time.monotonic()
            if remaining < 0.5:
                break
            time.sleep(0.5)
        if pending:
            raise LifecycleError(f"relay preflight failed: {','.join(pending)}")
        return relays

    @staticmethod
    def gpu_lock_available() -> None:
        if os.name != "posix":
            return
        import fcntl

        with open("/tmp/jiankong_gpu0.lock", "a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise LifecycleError("GPU lock busy") from error
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def write_generation(path: Path, generation_id: str, started_at: datetime) -> None:
        write_source_generation(path, generation_id, started_at)

    def http_self_check(self) -> None:
        from urllib.request import ProxyHandler, Request, build_opener

        opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + self.http_ready_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._http_self_check_once(opener, Request)
                return
            except (OSError, LifecycleError) as error:
                last_error = error
                time.sleep(0.25)
        raise LifecycleError("HTTP self-check failed") from last_error

    def _http_self_check_once(self, opener: Any, request_type: Any) -> None:
        try:
            for path in ("/", "/api/status"):
                request = request_type(
                    f"http://127.0.0.1:8767{path}",
                    headers={"Host": "192.168.50.2:8767"},
                )
                with opener.open(request, timeout=5) as response:
                    if response.status != 200:
                        raise LifecycleError(f"HTTP self-check returned {response.status}")
            run_dir = self.run_dir
            if run_dir is not None:
                fixture = Path(run_dir) / "dashboard" / "clips" / f".selfcheck-{uuid.uuid4().hex}.mp4"
                fixture.parent.mkdir(parents=True, exist_ok=True)
                fixture.write_bytes(b"range-self-check")
                try:
                    request = request_type(
                        f"http://127.0.0.1:8767/clips/{fixture.name}",
                        headers={"Host": "192.168.50.2:8767", "Range": "bytes=0-0"},
                    )
                    with opener.open(request, timeout=5) as response:
                        if response.status != 206:
                            raise LifecycleError("dashboard Range self-check failed")
                finally:
                    fixture.unlink(missing_ok=True)
        except OSError:
            raise

    def verify_runtime(self, identity: ProcessIdentity, run_dir: Path) -> None:
        source = run_dir / "inference" / "live_stats.jsonl"
        initial_size = source.stat().st_size if source.exists() else 0
        deadline = time.monotonic() + self.runtime_ready_timeout
        while time.monotonic() < deadline:
            if not self.is_same_process(identity):
                raise LifecycleError("DeepStream parent exited during startup")
            try:
                # A restarted pipeline may truncate and rewrite the existing
                # stats file, so any size change can represent fresh progress.
                if source.stat().st_size != initial_size:
                    if _has_stream_runtime_progress(
                        _read_latest_json_line(source), len(self._active_relays)
                    ):
                        return
            except FileNotFoundError:
                pass
            time.sleep(0.25)
        raise LifecycleError("live_stats.jsonl made no configured-stream startup progress")

    @staticmethod
    def _owned_docker_ids(owner_token: str) -> list[str]:
        token = _require_owner_token(owner_token)
        completed = subprocess.run(
            [
                "docker", "ps", "-aq", "--filter",
                f"label=jiankong.owner={token}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            raise LifecycleError("docker ownership query failed")
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    @classmethod
    def stop_component(cls, name: str, identity: ProcessIdentity) -> None:
        stop_process(identity)
        if name == "deepstream" and identity.owner_token:
            container_ids = cls._owned_docker_ids(identity.owner_token)
            if container_ids:
                completed = subprocess.run(
                    ["docker", "rm", "-f", *container_ids],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if completed.returncode != 0:
                    raise LifecycleError("docker rm failed for owned container")

    @staticmethod
    def is_same_process(identity: ProcessIdentity) -> bool:
        return is_same_process(identity)

    @classmethod
    def has_owned_processes(cls, name: str, identity: ProcessIdentity) -> bool:
        group_owned = owned_process_group_exists(identity)
        if name != "deepstream" or not identity.owner_token:
            return group_owned
        docker_owned = bool(cls._owned_docker_ids(identity.owner_token))
        return group_owned or docker_owned

    def _storage_for_run(self, run_dir: Path) -> RunStorage:
        storage = resolve_run_storage(run_dir)
        known = self._storage_bindings.setdefault(storage.run_dir, storage)
        if known != storage:
            raise ValueError("runtime metadata authority changed")
        return storage

    def block_new_events(self, service_instance: str) -> float:
        instance = _require_owner_token(service_instance)
        if self.run_dir is None:
            raise LifecycleError("run directory unavailable for stop handshake")
        requested_at = time.time()
        marker = self._storage_for_run(self.run_dir).metadata_dir / ".stop_events"
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_name(".stop_events.tmp")
        temporary.write_text(
            json.dumps({"service_instance": instance, "requested_at": requested_at}),
            encoding="utf-8",
        )
        os.replace(temporary, marker)
        return requested_at

    def drain_clips(
        self, timeout: float, service_instance: str, requested_at: float
    ) -> None:
        if self.run_dir is None:
            raise LifecycleError("run directory unavailable; forcing stop")
        instance = _require_owner_token(service_instance)
        deadline = time.monotonic() + min(timeout, 8.0)
        while time.monotonic() < deadline:
            status_path = self._storage_for_run(self.run_dir).metadata_dir / "worker_status.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                updated_at = float(status["updated_at"])
                active = int(status["active"])
                valid = (
                    status["service_instance"] == instance
                    and status["accepting"] is False
                    and updated_at >= requested_at
                    and 0 <= time.time() - updated_at <= 2.0
                    and active >= 0
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                valid = False
                active = -1
            if valid and active == 0:
                return
            time.sleep(0.1)
        raise LifecycleError(
            f"clip drain handshake timed out for service {instance}; forcing stop"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jiankong-live")
    parser.add_argument("command", choices=("config", "start", "status", "stop", "_services"))
    parser.add_argument("--config", default=str(Path("/media/boshi/Data/JianKong/02_configs/runtime/live_operator.json")))
    parser.add_argument("--state", default="/media/boshi/Data/JianKong/02_configs/runtime/live_operator_state.json")
    parser.add_argument("--run-dir")
    return parser


def _write_worker_status(
    path: Path,
    service_instance: str,
    *,
    accepting: bool,
    active: int,
    healthy: bool = True,
) -> None:
    instance = _require_owner_token(service_instance)
    if type(accepting) is not bool or type(active) is not int or active < 0 or type(healthy) is not bool:
        raise ValueError("invalid worker status")
    status = {
        "service_instance": instance,
        "updated_at": time.time(),
        "accepting": accepting,
        "active": active,
        "healthy": healthy,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(status), encoding="utf-8")
    os.replace(temporary, path)


class WorkerStatusReporter:
    """A status-file I/O fault must not terminate the business loop."""

    def __init__(self, path: Path, service_instance: str, *, run_dir: Path | None = None) -> None:
        self.path = path
        self.storage_run_dir = run_dir
        self.service_instance = _require_owner_token(service_instance)
        self._next_warning_at = 0.0

    def warn(self, message: str, error: Exception) -> None:
        now = time.monotonic()
        if now >= self._next_warning_at:
            self._next_warning_at = now + 30.0
            try:
                print(f"{message}: {redact_text(str(error))}", file=sys.stderr, flush=True)
            except (OSError, ValueError):
                # stderr may share the failed/full disk or already be closed.
                pass

    def publish(self, *, accepting: bool, active: int, healthy: bool) -> bool:
        try:
            if self.storage_run_dir is not None:
                if resolve_run_storage(self.storage_run_dir).metadata_dir != self.path.absolute().parent:
                    raise ValueError("worker metadata authority changed")
            _write_worker_status(
                self.path, self.service_instance,
                accepting=accepting, active=active, healthy=healthy,
            )
        except (OSError, ValueError) as error:
            # Leave the previous heartbeat unchanged. The watchdog then sees
            # stale progress instead of mistaking HTTP liveness for business health.
            self.warn("Worker status publication failed", error)
            return False
        return True


def _supervise_coordinator(
    coordinate: Callable[[], None], shutdown: Callable[[], None], failed: threading.Event
) -> None:
    try:
        coordinate()
    except BaseException as error:
        failed.set()
        try:
            print(f"Event coordinator terminated: {redact_text(str(error))}", file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass
        finally:
            # Run in the coordinator thread, never the serve_forever thread.
            # Recovery must not depend on the failed process's log filesystem.
            shutdown()


def _queue_collecting_clips(events, worker, futures) -> None:
    """Defer overflow on disk, without skipping the caller's completion drain."""
    for event in events:
        event_id = str(event.get("event_id", ""))
        if event.get("status") != "collecting" or event_id in futures:
            continue
        clip_post_seconds = event.get("clip_post_seconds", POST_EVENT_SECONDS)
        if not isinstance(clip_post_seconds, (int, float)) or clip_post_seconds < 0:
            clip_post_seconds = POST_EVENT_SECONDS
        try:
            futures[event_id] = worker.submit(
                ClipRequest(
                    event_id=event_id,
                    relay=str(event["camera"]),
                    occurred_at=datetime.fromisoformat(str(event["occurred_at"])),
                    event_stream_time_sec=float(event["event_stream_time_sec"]),
                    overlay=event,
                    post_event_seconds=float(clip_post_seconds),
                )
            )
        except ClipQueueFull:
            break


def _read_latest_json_lines(
    path: Path,
    *,
    limit: int = 2,
    tail_bytes: int = 64 * 1024,
) -> list[dict[str, Any]]:
    """Read newest complete JSON objects without scanning a growing stats file."""

    if limit <= 0:
        return []

    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - tail_bytes))
            payload = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    values: list[dict[str, Any]] = []
    for line in reversed(payload.splitlines()):
        try:
            value = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            values.append(value)
            if len(values) >= limit:
                break
    return values


def _read_latest_json_line(path: Path, *, tail_bytes: int = 64 * 1024) -> dict[str, Any] | None:
    """Read the newest complete JSON object without scanning a growing stats file."""

    values = _read_latest_json_lines(path, limit=1, tail_bytes=tail_bytes)
    return values[0] if values else None


def _recent_processed_stats(recent: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Use a roughly ten-second counter delta, never capture/target FPS.

    Rows are newest-first. A producer clock reset ends the usable history.
    Startup uses the available shorter window; insufficient/invalid data is zero.
    """
    if not recent:
        return None
    latest = recent[0]
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    end = latest.get("elapsed_sec")
    previous_time = end
    candidates = []
    if finite(end):
        for row in recent[1:]:
            start = row.get("elapsed_sec")
            if not finite(start) or start >= previous_time:
                break
            previous_time = start
            if end - start > 15:
                break
            candidates.append(row)
    baseline = min(candidates, key=lambda row: abs(end - row["elapsed_sec"] - 10)) if candidates else None
    older = {s.get("stream_index"): s for s in baseline.get("streams", [])
             if isinstance(s, dict)} if baseline else {}
    streams = []
    for stream in latest.get("streams", []):
        if not isinstance(stream, dict):
            continue
        before = older.get(stream.get("stream_index"), {}).get("processed")
        after = stream.get("processed")
        fps = 0.0
        monotonic = finite(after)
        newer_count = after
        if baseline is not None and monotonic:
            for row in candidates:
                if row["elapsed_sec"] < baseline["elapsed_sec"]:
                    break
                counter = next((s.get("processed") for s in row.get("streams", [])
                                if isinstance(s, dict) and s.get("stream_index") == stream.get("stream_index")), None)
                if not finite(counter) or counter > newer_count:
                    monotonic = False
                    break
                newer_count = counter
        if baseline is not None and monotonic and finite(before) and after >= before:
            fps = (after - before) / (end - baseline["elapsed_sec"])
        streams.append({**stream, "processed_fps": fps})
    return {**latest, "streams": streams}


def _build_dashboard_status(
    run_id: str,
    stats: dict[str, Any] | None,
    *,
    previous_stats: dict[str, Any] | None = None,
    camera_configs: tuple[object, ...] | list[object] | None = None,
    updated_at: float | None = None,
) -> dict[str, Any]:
    streams = stats.get("streams", []) if isinstance(stats, dict) else []
    indexed = {
        int(stream["stream_index"]): stream
        for stream in streams
        if isinstance(stream, dict) and isinstance(stream.get("stream_index"), int)
    }
    previous_streams = (
        previous_stats.get("streams", []) if isinstance(previous_stats, dict) else []
    )
    previous_indexed = {
        int(stream["stream_index"]): stream
        for stream in previous_streams
        if isinstance(stream, dict) and isinstance(stream.get("stream_index"), int)
    }
    cameras = []
    aggregate_fps = 0.0
    if camera_configs is None:
        slots = list(_LEGACY_STATUS_SLOTS)
    else:
        slots = [
            (str(getattr(camera, "relay")), str(getattr(camera, "view")))
            for camera in camera_configs
        ]
    for index, (relay, view) in enumerate(slots):
        stream = indexed.get(index)
        if stream is None:
            cameras.append(
                {
                    "relay": relay,
                    "view": view,
                    "status": "waiting",
                    "fps": 0.0,
                    "source_errors": 0,
                    "p95_latency_ms": 0.0,
                }
            )
            continue
        measured = stream.get("processed_fps")
        fps = float(measured) if type(measured) in (int, float) and math.isfinite(measured) and measured >= 0 else 0.0
        source_errors_total = max(0, int(stream.get("source_errors") or 0))
        previous_stream = previous_indexed.get(index)
        if previous_stream is None:
            source_errors = source_errors_total
        else:
            previous_errors_total = max(
                0, int(previous_stream.get("source_errors") or 0)
            )
            source_errors = (
                source_errors_total - previous_errors_total
                if source_errors_total >= previous_errors_total
                else source_errors_total
            )
        aggregate_fps += fps
        cameras.append(
            {
                "relay": relay,
                "view": str(stream.get("view") or view),
                "status": "online" if source_errors == 0 else "degraded",
                "fps": fps,
                "source_errors": source_errors,
                "p95_latency_ms": float(stream.get("latency_p95_ms") or 0.0),
            }
        )
    return {
        "run_id": run_id,
        "state": "running" if streams else "starting",
        "aggregate_fps": aggregate_fps,
        "updated_at": time.time() if updated_at is None else updated_at,
        "cameras": cameras,
    }


def _write_dashboard_status(
    path: Path,
    run_id: str,
    stats: dict[str, Any] | None,
    *,
    previous_stats: dict[str, Any] | None = None,
    camera_configs: tuple[object, ...] | list[object] | None = None,
    updated_at: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            _build_dashboard_status(
                run_id,
                stats,
                previous_stats=previous_stats,
                camera_configs=camera_configs,
                updated_at=updated_at,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _publish_dashboard_status_loop(
    run_dir: Path,
    dashboard_dir: Path,
    camera_configs: tuple[object, ...] | list[object],
    stop_event: threading.Event,
    *,
    interval_seconds: float = 0.5,
) -> None:
    """Publish dashboard health independently from slower service coordination."""

    stats_path = run_dir / "inference" / "live_stats.jsonl"
    status_path = dashboard_dir / "status.json"
    while not stop_event.is_set():
        try:
            if resolve_run_storage(run_dir).metadata_dir != dashboard_dir.absolute():
                raise ValueError("status metadata authority changed")
            recent_stats = _read_latest_json_lines(stats_path, limit=16, tail_bytes=256 * 1024)
            stats = _recent_processed_stats(recent_stats)
            previous_stats = recent_stats[1] if len(recent_stats) > 1 else None
            try:
                stats_updated_at = (
                    stats_path.stat().st_mtime if stats is not None else None
                )
            except OSError:
                stats_updated_at = None
            _write_dashboard_status(
                status_path,
                run_dir.name,
                stats,
                previous_stats=previous_stats,
                camera_configs=camera_configs,
                updated_at=stats_updated_at,
            )
        except Exception as error:
            # A transient read/write race must not permanently kill the
            # heartbeat.  The next interval retries from the latest stats.
            print(
                f"dashboard status publisher failed: {redact_text(str(error))}",
                file=sys.stderr,
                flush=True,
            )
        stop_event.wait(interval_seconds)


_LIVE_RUN_DIRECTORY = re.compile(r"live_[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _trusted_vlm_run_dirs(
    current_run_dir: Path,
    review_events_after: datetime | None = None,
) -> list[Path]:
    """Return direct, non-symlink run directories that can own review state."""

    try:
        if current_run_dir.is_symlink():
            return []
        current = current_run_dir.resolve(strict=True)
        root = current.parent
        cutoff_epoch = (
            review_events_after.timestamp()
            if review_events_after is not None
            else None
        )
    except OSError:
        return []

    def trusted(candidate: Path) -> Path | None:
        try:
            if (
                candidate.is_symlink()
                or _LIVE_RUN_DIRECTORY.fullmatch(candidate.name) is None
                or not candidate.is_dir()
            ):
                return None
            resolved = candidate.resolve(strict=True)
            if resolved.parent != root:
                return None
            storage = resolve_run_storage(resolved)
            dashboard = storage.metadata_dir
            if dashboard.is_symlink() or not dashboard.is_dir():
                return None
            dashboard_resolved = dashboard.resolve(strict=True)
            if dashboard_resolved != storage.metadata_dir:
                return None
            clips = storage.clips_dir
            if clips.is_symlink() or not clips.is_dir():
                return None
            clips_resolved = clips.resolve(strict=True)
            if clips_resolved.parent != resolved / "dashboard":
                return None
            events_path = dashboard_resolved / "events.json"
            if events_path.is_symlink():
                return None
            details = events_path.lstat()
            if not stat.S_ISREG(details.st_mode):
                return None
            if (
                resolved != current
                and cutoff_epoch is not None
                and details.st_mtime < cutoff_epoch
            ):
                return None
            if events_path.resolve(strict=True).parent != dashboard_resolved:
                return None
            return resolved
        except (OSError, ValueError):
            return None

    runs: list[Path] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for child in children:
        resolved = trusted(child)
        if resolved is not None:
            runs.append(resolved)
    return sorted(set(runs), key=lambda path: path.name)


def _read_vlm_events(
    run_dir: Path,
    state_store: VLMReviewStateStore | None = None,
    *,
    archive_only: bool = False,
) -> list[dict[str, Any]]:
    events_path = resolve_run_storage(run_dir).metadata_dir / "events.json"
    try:
        if events_path.is_symlink() or not stat.S_ISREG(events_path.lstat().st_mode):
            return []
        descriptor = os.open(
            events_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                payload = json.load(handle)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    events = (
        [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, list)
        else []
    )
    if archive_only:
        return events
    if state_store is None:
        state_store = VLMReviewStateStore(
            resolve_run_storage(run_dir).metadata_dir / DEFAULT_VLM_STATE_FILENAME,
            run_dir=run_dir,
        )
    return state_store.merge_events(events)


def _vlm_revision_matches(event: dict[str, Any], config: VLMReviewConfig) -> bool:
    return (
        event.get("vlm_filter_expected_model_version")
        == config.expected_model_version
        and event.get("vlm_filter_expected_prompt_revision")
        == config.expected_prompt_revision
        and event.get("vlm_filter_evidence_revision")
        == config.expected_evidence_revision
    )


def _vlm_event_is_in_scope(event: dict[str, Any], config: VLMReviewConfig) -> bool:
    if event.get("status") != "ready":
        return False
    cutoff = config.review_events_after
    if cutoff is None:
        return True
    occurred_at = event.get("occurred_at")
    if not isinstance(occurred_at, str):
        return False
    try:
        occurred = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (
        occurred.tzinfo is not None
        and occurred.utcoffset() is not None
        and occurred >= cutoff
    )


def _vlm_event_priority(
    event: dict[str, Any], config: VLMReviewConfig, now_epoch: float
) -> int | None:
    """Rank recoverable pending work before retry/new/revision work."""

    if not _vlm_event_is_in_scope(event, config):
        return None
    state = event.get("vlm_filter_result")
    revisions_match = _vlm_revision_matches(event, config)
    attempts = event.get("vlm_filter_attempts", 0)
    attempts = attempts if type(attempts) is int and attempts >= 0 else 0
    if state in {"pass", "filter", "uncertain"}:
        return None if revisions_match else 2
    if state == "pending":
        if not revisions_match:
            return 2
        request_id = event.get("vlm_filter_request_id")
        return (
            0
            if attempts >= 1
            and isinstance(request_id, str)
            and re.fullmatch(r"[A-Fa-f0-9]{64}", request_id)
            else None
        )
    if state == "error":
        if not revisions_match:
            return 2
        if event.get("vlm_filter_retryable") is False:
            return None
        if attempts >= config.max_attempts:
            return None
        retry_at = event.get("vlm_filter_retry_at")
        if isinstance(retry_at, (int, float)) and now_epoch < float(retry_at):
            return None
        return 1
    if state is None:
        # A newly ready event must not wait behind a large backlog created by
        # an evidence/prompt revision rollout.  Persisted pending work remains
        # first within this priority via chronological ordering.
        return 0
    return None


_VLM_SCAN_SNAPSHOT_KEYS = {
    "event_id",
    "status",
    "occurred_at",
    "vlm_filter_result",
    "vlm_filter_attempts",
    "vlm_filter_request_id",
    "vlm_filter_retryable",
    "vlm_filter_retry_at",
    "vlm_filter_error",
    "vlm_filter_expected_model_version",
    "vlm_filter_expected_prompt_revision",
    "vlm_filter_evidence_revision",
}


class VLMArchiveScanCache:
    """Cache only compact archive fields, never live review transitions.

    Archive replacement/removal invalidates the entry. Fresh sidecar state is
    merged on every read; the existing request fencing remains authoritative.
    """

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[tuple, list[dict[str, Any]]]] = {}

    @staticmethod
    def _signature(path: Path) -> tuple:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode):
            raise OSError("not a regular event archive")
        return (details.st_dev, details.st_ino, details.st_size,
                details.st_mtime_ns, details.st_ctime_ns)

    def read(self, run_dir: Path, store: VLMReviewStateStore) -> list[dict[str, Any]]:
        path = resolve_run_storage(run_dir).metadata_dir / "events.json"
        try:
            signature = self._signature(path)
            entry = self._entries.get(path)
            if entry is None or entry[0] != signature:
                self._entries.pop(path, None)
                events = _read_vlm_events(run_dir, archive_only=True)
                if not events or self._signature(path) != signature:
                    return []
                compact = [{key: event[key] for key in _VLM_SCAN_SNAPSHOT_KEYS
                            if key in event} for event in events]
                if len(self._entries) >= 128:
                    self._entries.pop(next(iter(self._entries)))
                entry = (signature, compact)
                self._entries[path] = entry
            return store.merge_events(entry[1])
        except OSError:
            self._entries.pop(path, None)
            return []


def _vlm_scan_snapshot(
    events: list[dict[str, Any]],
    config: VLMReviewConfig,
    now_epoch: float,
) -> list[dict[str, Any]]:
    """Keep only fields needed to rescan historical VLM candidates."""

    snapshot = []
    for event in events:
        priority = _vlm_event_priority(event, config, now_epoch)
        if priority is None and event.get("vlm_filter_result") != "error":
            continue
        snapshot.append(
            {key: event[key] for key in _VLM_SCAN_SNAPSHOT_KEYS if key in event}
        )
    return snapshot


def _vlm_candidate_sort_key(
    current_run: Path,
    candidate_run: Path,
    priority: int,
    occurred_at: str,
    event_id: str,
) -> tuple[int, int, str, str]:
    """Keep live events ahead of historical backfill without starving either."""

    return (
        0 if candidate_run == current_run else 1,
        priority,
        occurred_at,
        event_id,
    )


def _begin_vlm_review(
    state_store: VLMReviewStateStore,
    run_dir: Path,
    event: dict[str, Any],
    config: VLMReviewConfig,
) -> dict[str, Any]:
    event = state_store.merge_events([event])[0]
    event_id = str(event.get("event_id", ""))
    clip_path = run_dir / "dashboard" / "clips" / f"{event_id}.mp4"
    overlay_path = run_dir / "dashboard" / "clips" / f"{event_id}.json"
    request_id = evidence_request_id(
        event_id,
        clip_path,
        overlay_path,
        model_version=config.expected_model_version,
        prompt_revision=config.expected_prompt_revision,
        evidence_revision=config.expected_evidence_revision,
    )
    attempts_value = event.get("vlm_filter_attempts", 0)
    current_attempts = (
        attempts_value if type(attempts_value) is int and attempts_value >= 0 else 0
    )
    state = event.get("vlm_filter_result")
    revisions_match = _vlm_revision_matches(event, config)
    if state == "pending" and revisions_match:
        # A process may have stopped after persisting the claim but before the
        # remote response.  Resubmit the identical request without consuming a
        # retry or changing its fencing token.
        next_attempt = current_attempts
    elif state == "error" and revisions_match:
        next_attempt = current_attempts + 1
    else:
        next_attempt = 1
    return state_store.set_vlm_filter_result(
        event,
        "pending",
        attempts=next_attempt,
        expected_attempts=current_attempts,
        request_id=request_id,
        evidence_revision=config.expected_evidence_revision,
        prompt_revision=config.expected_prompt_revision,
        model_version=config.expected_model_version,
    )


def _open_historical_coordinator(run_dir: Path) -> EventCoordinator:
    storage = resolve_run_storage(run_dir)
    historical_fps = 8.0
    try:
        status = json.loads(
            (storage.metadata_dir / "status.json").read_text(encoding="utf-8")
        )
        candidate_fps = float(status.get("target_fps_per_stream", 8.0))
        if candidate_fps in {8.0, 10.0}:
            historical_fps = candidate_fps
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return EventCoordinator(
        run_dir / "inference" / "frame_events.jsonl",
        storage.metadata_dir,
        run_dir=run_dir,
        generation_file=run_dir / "inference" / "source_generation.json",
        run_started_at=datetime.now(timezone.utc),
        infer_fps=historical_fps,
    )


def _run_services(run_dir: Path, config_path: Path) -> int:
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

    from live_operator.dashboard import DashboardApp

    try:
        live_config = LiveConfig.load(config_path)
        active_cameras = live_config.enabled_cameras
        hidden_event_cameras = {
            camera.relay
            for camera in getattr(live_config, "cameras", active_cameras)
            if str(getattr(camera, "view", "")).startswith("研究院")
        }
        inference_version = resolve_current_inference_version()
        calibration_dir = inference_version.calibration_dir
        infer_fps = getattr(inference_version, "infer_fps", 8)
    except (OSError, ValueError):
        return 2
    generation_file = run_dir / "inference" / "source_generation.json"
    source = run_dir / "inference" / "frame_events.jsonl"
    dashboard_dir = resolve_run_storage(run_dir).metadata_dir
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    try:
        service_instance = _require_owner_token(os.environ.get("JIAN_KONG_OWNER_TOKEN"))
    except LifecycleError:
        return 2
    vlm_config_path = Path(
        os.environ.get("JIAN_KONG_VLM_REVIEW_CONFIG", str(DEFAULT_VLM_REVIEW_CONFIG))
    )
    vlm_config = None
    vlm_worker = None
    try:
        vlm_config = VLMReviewConfig.load_optional(vlm_config_path)
        if vlm_config is not None:
            vlm_worker = VLMReviewWorker(VLMReviewClient(vlm_config))
    except (OSError, ValueError) as error:
        print(
            f"VLM review disabled: {redact_text(str(error))}",
            file=sys.stderr,
            flush=True,
        )

    def coordinate() -> None:
        coordinator = None
        reporter = WorkerStatusReporter(dashboard_dir / "worker_status.json", service_instance, run_dir=run_dir)
        worker = ClipRemuxWorker(
            recordings_dir=run_dir / "recordings", run_dir=run_dir
        )
        futures = {}
        vlm_inflight = None
        vlm_prefetch = None
        vlm_state_stores: dict[Path, VLMReviewStateStore] = {}

        def vlm_state_store(candidate_run: Path) -> VLMReviewStateStore:
            resolved = candidate_run.resolve()
            store = vlm_state_stores.get(resolved)
            if store is None:
                store = VLMReviewStateStore(
                    resolve_run_storage(resolved).metadata_dir / DEFAULT_VLM_STATE_FILENAME,
                    run_dir=resolved,
                )
                vlm_state_stores[resolved] = store
            return store

        vlm_historical_scan_cache: dict[
            Path, tuple[tuple[int, int, int, int], list[dict[str, Any]]]
        ] = {}
        vlm_archive_cache = VLMArchiveScanCache()
        next_vlm_scan_at = 0.0
        vlm_health_ok_until = 0.0
        next_vlm_health_log_at = 0.0
        stop_seen_at = None
        latest_current_events = []
        while True:
            # Validate the committed authority before any writer in this cycle.
            if resolve_run_storage(run_dir).metadata_dir != dashboard_dir:
                raise ValueError("service metadata authority changed")
            vlm_completed_this_cycle = False
            # Startup is not business progress: require an actual successful
            # coordinator cycle, including when producer metadata arrives late.
            cycle_healthy = False
            accepting = True
            try:
                stop_request = json.loads(
                    (dashboard_dir / ".stop_events").read_text(encoding="utf-8")
                )
                accepting = stop_request.get("service_instance") != service_instance
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                accepting = True
            if not accepting and stop_seen_at is None:
                stop_seen_at = time.monotonic()
            if coordinator is None and generation_file.is_file():
                coordinator = EventCoordinator(
                    source,
                    dashboard_dir,
                    run_dir=run_dir,
                    generation_file=generation_file,
                    run_started_at=datetime.now(timezone.utc),
                    infer_fps=infer_fps,
                    stream_cameras={
                        index: camera.relay
                        for index, camera in enumerate(active_cameras)
                    },
                )
            if coordinator is not None:
                latest_current_events = []
                try:
                    events = coordinator.poll()
                    latest_current_events = (
                        vlm_state_store(run_dir).merge_events(events)
                        if vlm_config is not None
                        else events
                    )
                    if accepting:
                        _queue_collecting_clips(events, worker, futures)
                    for event_id, future in list(futures.items()):
                        if not future.done():
                            continue
                        result = future.result()
                        if result.status == "ready" and result.overlay_path is not None:
                            latest_event = next(
                                (item for item in events if str(item.get("event_id", "")) == event_id),
                                None,
                            )
                            if latest_event is not None:
                                refresh_clip_overlay(result.overlay_path, latest_event)
                        ready_event = coordinator.set_status(
                            event_id,
                            "ready" if result.status == "ready" else "failed",
                            error=redact_text(result.error or "") if result.error else None,
                        )
                        del futures[event_id]
                    cycle_healthy = True
                except Exception as error:
                    cycle_healthy = False
                    reporter.warn("Event coordination failed; pending data retained", error)

            if vlm_inflight is not None and vlm_inflight["future"].done():
                # ``latest_current_events`` was captured before this completion
                # is persisted.  Do not let the scan below resubmit that stale
                # pending snapshot in the same loop iteration; the next poll
                # reloads the terminal/retry state from disk.
                vlm_completed_this_cycle = True
                task = vlm_inflight
                target_store = task["state_store"]
                try:
                    try:
                        review = task["future"].result()
                    except Exception as error:
                        retry_at = None
                        retryable = bool(getattr(error, "retryable", True))
                        if (
                            vlm_config is not None
                            and retryable
                            and task["attempts"] < vlm_config.max_attempts
                        ):
                            retry_at = time.time() + vlm_config.retry_delay_seconds(
                                task["attempts"]
                            )
                        message = " ".join(
                            (redact_text(str(error)).strip() or "VLM review failed").split()
                        )[:1000]
                        target_store.set_vlm_filter_result(
                            task["event"],
                            "error",
                            attempts=task["attempts"],
                            expected_attempts=task["attempts"],
                            request_id=task["request_id"],
                            evidence_revision=vlm_config.expected_evidence_revision,
                            prompt_revision=vlm_config.expected_prompt_revision,
                            model_version=vlm_config.expected_model_version,
                            retry_at=retry_at,
                            retryable=retryable,
                            error=message,
                        )
                    else:
                        target_store.set_vlm_filter_result(
                            task["event"],
                            review.result,
                            attempts=task["attempts"],
                            expected_attempts=task["attempts"],
                            request_id=review.request_id,
                            evidence_revision=review.evidence_revision,
                            prompt_revision=review.prompt_revision,
                            model_version=review.model_version,
                            label=review.label,
                            reviewed_at=review.reviewed_at,
                            latency_seconds=review.latency_seconds,
                        )
                except Exception as error:
                    print(
                        f"VLM review state update failed: {redact_text(str(error))}",
                        file=sys.stderr,
                        flush=True,
                    )
                finally:
                    vlm_inflight = None
                    if vlm_prefetch is not None and vlm_worker is not None:
                        promoted = vlm_prefetch
                        promoted["future"] = vlm_worker.submit_prepared(
                            promoted.pop("prepared_future")
                        )
                        vlm_inflight = promoted
                        vlm_prefetch = None

            scan_now = time.monotonic()
            if (
                accepting
                and not vlm_completed_this_cycle
                and coordinator is not None
                and vlm_config is not None
                and vlm_worker is not None
                and (vlm_inflight is None or vlm_prefetch is None)
                and scan_now >= next_vlm_scan_at
            ):
                if vlm_inflight is None and scan_now >= vlm_health_ok_until:
                    try:
                        vlm_worker.client.check_health()
                    except Exception as error:
                        if scan_now >= next_vlm_health_log_at:
                            print(
                                f"VLM review health check deferred: {redact_text(str(error))}",
                                file=sys.stderr,
                                flush=True,
                            )
                            next_vlm_health_log_at = scan_now + 30.0
                        next_vlm_scan_at = scan_now + 5.0
                        continue
                    vlm_health_ok_until = scan_now + 10.0
                now_epoch = time.time()
                current_resolved = run_dir.resolve()
                candidates = []
                busy_event_ids = {
                    str(task.get("event_id", ""))
                    for task in (vlm_inflight, vlm_prefetch)
                    if isinstance(task, dict)
                }
                for event in latest_current_events:
                    if str(event.get("event_id", "")) in busy_event_ids:
                        continue
                    priority = _vlm_event_priority(event, vlm_config, now_epoch)
                    if priority is None:
                        continue
                    occurred_at = str(event.get("occurred_at", ""))
                    event_id = str(event.get("event_id", ""))
                    candidates.append(
                        (
                            _vlm_candidate_sort_key(
                                current_resolved,
                                current_resolved,
                                priority,
                                occurred_at,
                                event_id,
                            ),
                            event_id,
                            current_resolved,
                            event,
                        )
                    )

                # Historical recovery is intentionally lower priority than
                # live review.  Do not parse hundreds of megabytes of old
                # events before every current submission: with a busy live
                # run that unnecessary scan can make an eight-second model
                # call appear stalled for minutes.  Once the current run has
                # no eligible work, resume the durable historical backfill.
                if not candidates:
                    trusted_runs = _trusted_vlm_run_dirs(
                        run_dir, vlm_config.review_events_after
                    )
                    trusted_run_set = set(trusted_runs)
                    for cached_run in tuple(vlm_historical_scan_cache):
                        if cached_run not in trusted_run_set:
                            vlm_historical_scan_cache.pop(cached_run, None)
                    for candidate_run in trusted_runs:
                        if candidate_run == current_resolved:
                            continue
                        metadata_dir = resolve_run_storage(candidate_run).metadata_dir
                        events_path = metadata_dir / "events.json"
                        try:
                            details = events_path.stat()
                            state_path = (
                                metadata_dir
                                / DEFAULT_VLM_STATE_FILENAME
                            )
                            try:
                                state_details = state_path.stat()
                                state_signature = (
                                    state_details.st_mtime_ns,
                                    state_details.st_size,
                                )
                            except OSError:
                                state_signature = (-1, -1)
                            signature = (
                                details.st_mtime_ns,
                                details.st_size,
                                *state_signature,
                            )
                        except OSError:
                            vlm_historical_scan_cache.pop(candidate_run, None)
                            continue
                        cached_run = vlm_historical_scan_cache.get(candidate_run)
                        if cached_run is not None and cached_run[0] == signature:
                            candidate_events = cached_run[1]
                        else:
                            candidate_events = _vlm_scan_snapshot(
                                vlm_archive_cache.read(
                                    candidate_run,
                                    vlm_state_store(candidate_run),
                                ),
                                vlm_config,
                                now_epoch,
                            )
                            vlm_historical_scan_cache[candidate_run] = (
                                signature,
                                candidate_events,
                            )
                        for event in candidate_events:
                            if str(event.get("event_id", "")) in busy_event_ids:
                                continue
                            priority = _vlm_event_priority(event, vlm_config, now_epoch)
                            if priority is None:
                                continue
                            occurred_at = str(event.get("occurred_at", ""))
                            event_id = str(event.get("event_id", ""))
                            candidates.append(
                                (
                                    _vlm_candidate_sort_key(
                                        current_resolved,
                                        candidate_run,
                                        priority,
                                        occurred_at,
                                        event_id,
                                    ),
                                    event_id,
                                    candidate_run,
                                    event,
                                )
                            )
                candidates.sort(key=lambda item: item[:4])
                for (
                    _sort_key,
                    event_id,
                    candidate_run,
                    event,
                ) in candidates:
                    target_store = vlm_state_store(candidate_run)
                    if candidate_run != current_resolved:
                        fresh = next(
                            (
                                item for item in vlm_archive_cache.read(
                                    candidate_run,
                                    target_store,
                                )
                                if str(item.get("event_id", "")) == event_id
                            ),
                            None,
                        )
                        if fresh is None:
                            continue
                        # Re-evaluate from the atomically merged sidecar.
                        if (
                            _vlm_event_priority(fresh, vlm_config, time.time())
                            is None
                        ):
                            continue
                        event = fresh
                    try:
                        state = event.get("vlm_filter_result")
                        if state == "pending" and _vlm_revision_matches(event, vlm_config):
                            claimed = event
                        else:
                            claimed = _begin_vlm_review(
                                target_store,
                                candidate_run,
                                event,
                                vlm_config,
                            )
                        request_id = str(claimed["vlm_filter_request_id"])
                        attempts = int(claimed["vlm_filter_attempts"])
                        clip_path = candidate_run / "dashboard" / "clips" / f"{event_id}.mp4"
                        overlay_path = candidate_run / "dashboard" / "clips" / f"{event_id}.json"
                        task = {
                            "event_id": event_id,
                            "event": claimed,
                            "request_id": request_id,
                            "attempts": attempts,
                            "state_store": target_store,
                        }
                        if vlm_inflight is None:
                            task["future"] = vlm_worker.submit(
                                event_id, request_id, clip_path, overlay_path
                            )
                            vlm_inflight = task
                        else:
                            task["prepared_future"] = vlm_worker.prepare(
                                event_id, request_id, clip_path, overlay_path
                            )
                            vlm_prefetch = task
                    except Exception as error:
                        print(
                            f"VLM review submit failed: {redact_text(str(error))}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    break
                # Start the interval after the scan finishes.  A slow history
                # scan must not immediately trigger another back-to-back scan.
                next_vlm_scan_at = time.monotonic() + 1.0

            # Clip drain controls restart safety.  VLM state is already
            # durable as pending and is deliberately resumed by the next
            # services process (or satisfied from the remote cache).
            active = sum(not future.done() for future in futures.values())
            reporter.publish(
                accepting=accepting,
                active=active,
                healthy=cycle_healthy,
            )
            if not accepting and (
                active == 0
                or time.monotonic() - stop_seen_at >= 8.0
            ):
                break
            time.sleep(0.25)
        if coordinator is not None:
            coordinator.close()
        worker.shutdown(wait=False)
        if vlm_worker is not None:
            vlm_worker.shutdown(wait=False)

    class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True
        block_on_close = False

    class BoundedWSGIRequestHandler(WSGIRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(15.0)

    dashboard_app = DashboardApp(
        run_dir,
        config_path=config_path,
        calibration_dir=calibration_dir,
        hidden_event_cameras=hidden_event_cameras,
    )
    server = make_server(
        "0.0.0.0",
        8767,
        dashboard_app,
        server_class=ThreadingWSGIServer,
        handler_class=BoundedWSGIRequestHandler,
    )
    dashboard_app.start_event_cache_warmup()
    status_stop = threading.Event()
    coordinate_failed = threading.Event()
    coordinate_thread = threading.Thread(
        target=lambda: _supervise_coordinator(coordinate, server.shutdown, coordinate_failed),
        daemon=True,
    )
    status_thread = threading.Thread(
        target=lambda: _publish_dashboard_status_loop(
            run_dir,
            dashboard_dir,
            active_cameras,
            status_stop,
        ),
        daemon=True,
    )
    coordinate_thread.start()
    status_thread.start()
    try:
        server.serve_forever()
    finally:
        status_stop.set()
        status_thread.join(timeout=2.0)
        server.server_close()
    return 1 if coordinate_failed.is_set() else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "config":
        configure(args.config)
        print("configuration saved with private permissions")
        return 0
    if args.command == "_services":
        if not args.run_dir:
            raise SystemExit("--run-dir is required")
        return _run_services(Path(args.run_dir), Path(args.config))
    hooks = RuntimeHooks()
    lifecycle = LifecycleManager(StateStore(args.state), hooks)
    if args.command == "start":
        run_dir = Path(args.run_dir) if args.run_dir else Path(
            "/media/boshi/Data/JianKong/08_inference_results"
        ) / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = lifecycle.start(args.config, run_dir)
    elif args.command == "status":
        result = lifecycle.status()
    else:
        result = lifecycle.stop()
    print(redact_text(json.dumps(result, ensure_ascii=True, sort_keys=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
