from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from live_operator.cli import (
    LifecycleManager,
    LifecycleError,
    RuntimeHooks,
    _build_dashboard_status,
    _read_latest_json_line,
)
from live_operator.processes import ProcessIdentity, StateStore
from live_operator.config import make_legacy_config


def _configured_hooks(**kwargs):
    hooks = RuntimeHooks(**kwargs)
    hooks._active_relays = tuple(f"camera{index:02d}" for index in range(1, 8))
    return hooks


def _camera_config():
    return make_legacy_config("fixture", "fixture", {
        view: f"192.0.2.{index}" for index, view in enumerate(
            ("dianqi1", "dianqi2", "jixie1", "jixie2", "ruanjian1", "ruanjian2", "zoulang"), 1)
    })


class Hooks:
    def __init__(self) -> None:
        self.calls = []
        self.fail = None
        self.identities = {}
        self.stopped = set()

    def validate_config(self, path):
        self.calls.append("config")
        return {"password": "DistinctSecret", "cameras": list(range(7))}

    def probe_sources(self, config):
        self.calls.append("sources")
        if self.fail == "sources":
            raise LifecycleError("source preflight failed")
        return [f"camera{i:02d}" for i in range(1, 8)]

    def check_ports(self):
        self.calls.append("ports")
        if self.fail == "ports":
            raise LifecycleError("port conflict")

    def start_component(self, name, run_dir, config):
        self.calls.append(f"start:{name}")
        if self.fail == name:
            raise LifecycleError(f"{name} failed")
        identity = ProcessIdentity(
            100 + len(self.identities), f"token-{name}", None, f"{len(self.identities) + 1:032x}"
        )
        self.identities[name] = identity
        return identity

    def probe_relays(self):
        self.calls.append("relays")
        if self.fail == "relays":
            raise LifecycleError("relay preflight failed")
        return [f"camera{i:02d}" for i in range(1, 8)]

    def gpu_lock_available(self):
        self.calls.append("gpu-lock")
        if self.fail == "gpu-lock":
            raise LifecycleError("GPU lock busy")

    def write_generation(self, path, generation_id, started_at):
        self.calls.append("generation")
        if self.fail == "generation":
            raise LifecycleError("generation failed")
        assert started_at.tzinfo is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"generation_id": generation_id, "started_at": started_at.isoformat()}))

    def http_self_check(self):
        self.calls.append("http")
        if self.fail == "http":
            raise LifecycleError("HTTP self-check failed")

    def verify_runtime(self, identity, run_dir):
        self.calls.append("core")
        if self.fail == "core":
            raise LifecycleError("DeepStream core is not progressing")

    def stop_component(self, name, identity):
        self.calls.append(f"stop:{name}")
        self.stopped.add((identity.pid, identity.start_token))

    def block_new_events(self, service_instance):
        self.calls.append("block-events")
        return 1000.0

    def drain_clips(self, timeout, service_instance, requested_at):
        self.calls.append(f"drain:{timeout}")

    def is_same_process(self, identity):
        return (
            identity.start_token.startswith("token-")
            and (identity.pid, identity.start_token) not in self.stopped
        )

    def has_owned_processes(self, name, identity):
        return False


def manager(tmp_path: Path, hooks: Hooks) -> LifecycleManager:
    return LifecycleManager(
        StateStore(tmp_path / "state.json"),
        hooks,
        now=lambda: datetime(2026, 7, 17, 8, 0, tzinfo=timezone.utc),
        generation_id=lambda: "generation-unique-001",
    )


def test_start_order_services_wait_for_deepstream_runtime(tmp_path: Path) -> None:
    hooks = Hooks()
    result = manager(tmp_path, hooks).start(tmp_path / "config.json", tmp_path / "run")
    assert hooks.calls == [
        "config", "sources", "ports", "start:mediamtx", "relays",
        "gpu-lock", "generation", "start:deepstream", "core", "start:services", "http",
    ]
    assert result["state"] == "running"
    assert result["sources"] == result["relays"] == [f"camera{i:02d}" for i in range(1, 8)]
    persisted = json.loads((tmp_path / "state.json").read_text())
    assert "DistinctSecret" not in json.dumps(persisted)


@pytest.mark.parametrize("failure", ("sources", "ports", "gpu-lock"))
def test_preflight_failure_never_starts_deepstream(tmp_path: Path, failure: str) -> None:
    hooks = Hooks()
    hooks.fail = failure
    with pytest.raises(LifecycleError):
        manager(tmp_path, hooks).start(tmp_path / "config", tmp_path / "run")
    assert "start:deepstream" not in hooks.calls
    if failure in {"sources", "ports"}:
        assert not any(call.startswith("start:") for call in hooks.calls)


def test_duplicate_start_is_rejected(tmp_path: Path) -> None:
    hooks = Hooks()
    lifecycle = manager(tmp_path, hooks)
    lifecycle.start(tmp_path / "config", tmp_path / "run")
    before = list(hooks.calls)
    with pytest.raises(LifecycleError, match="already running"):
        lifecycle.start(tmp_path / "config", tmp_path / "run2")
    assert hooks.calls == before


def test_failure_rolls_back_reverse_order_and_preserves_diagnostics(tmp_path: Path) -> None:
    hooks = Hooks()
    hooks.fail = "deepstream"
    with pytest.raises(LifecycleError):
        manager(tmp_path, hooks).start(tmp_path / "config", tmp_path / "run")
    assert hooks.calls[-1:] == ["stop:mediamtx"]
    assert "stop:services" not in hooks.calls
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["state"] == "failed"
    assert state["generation_id"] == "generation-unique-001"


def test_pid_reuse_is_stale_and_never_signalled(tmp_path: Path) -> None:
    hooks = Hooks()
    lifecycle = manager(tmp_path, hooks)
    lifecycle.start(tmp_path / "config", tmp_path / "run")
    state = json.loads((tmp_path / "state.json").read_text())
    state["processes"]["deepstream"]["start_token"] = "reused-pid-token"
    (tmp_path / "state.json").write_text(json.dumps(state))
    assert lifecycle.status()["state"] == "stale"
    lifecycle.stop()
    assert "stop:deepstream" not in hooks.calls


def test_stop_order_drains_clips_and_removes_state(tmp_path: Path) -> None:
    hooks = Hooks()
    lifecycle = manager(tmp_path, hooks)
    lifecycle.start(tmp_path / "config", tmp_path / "run")
    hooks.calls.clear()
    result = lifecycle.stop()
    assert hooks.calls == [
        "block-events", "drain:8.0", "stop:deepstream", "stop:services", "stop:mediamtx"
    ]
    assert result["state"] == "stopped"
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("failure", ("relays", "generation", "core", "http"))
def test_runtime_failure_rolls_back_every_started_component(
    tmp_path: Path, failure: str
) -> None:
    hooks = Hooks()
    hooks.fail = failure
    with pytest.raises(LifecycleError):
        manager(tmp_path, hooks).start(tmp_path / "config", tmp_path / "run")
    if failure == "core":
        assert hooks.calls[-2:] == ["stop:deepstream", "stop:mediamtx"]
        assert "stop:services" not in hooks.calls
    elif failure == "http":
        assert hooks.calls[-3:] == ["stop:deepstream", "stop:services", "stop:mediamtx"]
    assert json.loads((tmp_path / "state.json").read_text())["state"] == "failed"


def test_ffprobe_memfd_hides_credentials_and_validates_video(monkeypatch) -> None:
    secret = "DistinctProbePassword"
    observed = {}

    def runner(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        assert secret not in repr(command)
        assert kwargs["timeout"] == 12.0
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"streams": [{"codec_type": "video", "codec_name": "hevc", "width": 2560, "height": 1440, "r_frame_rate": "25/1"}]}
            ),
            stderr="",
        )

    hooks = RuntimeHooks(probe_runner=runner)
    assert hooks._probe(f"rtsp://user:{secret}@192.0.2.1/Streaming/Channels/101")
    assert secret not in repr(observed)
    assert "-rtsp_transport" in observed["command"]
    assert "tcp" in observed["command"]


def test_lifecycle_lock_serializes_concurrent_start_before_components(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingHooks(Hooks):
        def validate_config(self, path):
            entered.set()
            assert release.wait(2)
            return super().validate_config(path)

    first_hooks = BlockingHooks()
    second_hooks = Hooks()
    first = manager(tmp_path, first_hooks)
    second = manager(tmp_path, second_hooks)
    errors = []
    thread = threading.Thread(
        target=lambda: first.start(tmp_path / "config", tmp_path / "run")
    )
    thread.start()
    assert entered.wait(1)

    def second_start():
        try:
            second.start(tmp_path / "config", tmp_path / "run2")
        except Exception as error:
            errors.append(error)

    second_thread = threading.Thread(target=second_start)
    second_thread.start()
    time.sleep(0.1)
    assert second_hooks.calls == []
    release.set()
    thread.join(2)
    second_thread.join(2)
    assert isinstance(errors[0], LifecycleError)
    assert second_hooks.calls == []


def test_production_relay_probe_checks_all_seven_and_rejects_one_failure() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1 if len(calls) == 4 else 0,
            stdout=json.dumps({"streams": [{"codec_type": "video", "codec_name": "hevc", "width": 2560, "height": 1440, "r_frame_rate": "25/1"}]}),
            stderr="",
        )

    with pytest.raises(LifecycleError, match=r"camera0[1-7]"):
        _configured_hooks(probe_runner=runner, relay_ready_timeout=0).probe_relays()
    assert len(calls) == 7


def test_production_relay_probe_retries_a_source_while_mediamtx_warms_up() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1 if len(calls) == 1 else 0,
            stdout=json.dumps({"streams": [{"codec_type": "video", "codec_name": "hevc", "width": 2560, "height": 1440, "r_frame_rate": "25/1"}]}),
            stderr="",
        )

    assert _configured_hooks(probe_runner=runner, relay_ready_timeout=2).probe_relays() == [
        f"camera{index:02d}" for index in range(1, 8)
    ]
    assert len(calls) == 8


def test_production_relay_probe_respects_short_total_warmup_budget() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        time.sleep(0.03)
        return subprocess.CompletedProcess(command, 1, stdout="{}", stderr="")

    started = time.monotonic()
    with pytest.raises(LifecycleError, match="relay preflight failed"):
        _configured_hooks(probe_runner=runner, relay_ready_timeout=0.05).probe_relays()
    assert time.monotonic() - started < 0.2
    assert len(calls) == 7


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires Linux /proc process ownership")
def test_process_group_and_labeled_container_cleanup(monkeypatch) -> None:
    import live_operator.cli as cli_module
    import live_operator.processes as process_module

    if os.name == "posix":
        signals = []
        checks = iter((True, False))
        monkeypatch.setattr(process_module, "is_same_process", lambda identity: next(checks, False))
        monkeypatch.setattr(process_module.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
        process_module.stop_process(ProcessIdentity(10, "token", 410, "1" * 32), timeout=0.01)
        assert signals and signals[0][0] == 410

    commands = []
    monkeypatch.setattr(cli_module, "stop_process", lambda identity: None)

    def run(command, **kwargs):
        commands.append(command)
        stdout = "container-id\n" if command[1] == "ps" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cli_module.subprocess, "run", run)
    RuntimeHooks.stop_component(
        "deepstream", ProcessIdentity(10, "token", 410, "1" * 32)
    )
    assert ["docker", "ps", "-aq", "--filter", f"label=jiankong.owner={'1' * 32}"] in commands
    assert ["docker", "rm", "-f", "container-id"] in commands


def test_start_component_passes_unique_strict_owner_token_to_launcher(
    tmp_path: Path, monkeypatch
) -> None:
    import live_operator.cli as cli_module

    observed = []

    class Process:
        pid = 123

    monkeypatch.setattr(cli_module.subprocess, "Popen", lambda command, **kwargs: observed.append((command, kwargs)) or Process())
    monkeypatch.setattr(
        cli_module,
        "capture_identity",
        lambda process, owner_token=None: ProcessIdentity(process.pid, "start", 123, owner_token),
    )
    hooks = RuntimeHooks()
    hooks.config_path = tmp_path / "live.json"
    first = hooks.start_component("services", tmp_path, object())
    second = hooks.start_component("services", tmp_path, object())
    assert first.owner_token != second.owner_token
    for (_, kwargs), identity in zip(observed, (first, second)):
        token = kwargs["env"]["JIAN_KONG_OWNER_TOKEN"]
        assert token == identity.owner_token
        assert len(token) == 32 and all(character in "0123456789abcdef" for character in token)
        assert kwargs["env"]["PYTHONPATH"] == str(Path(cli_module.__file__).resolve().parents[1])


def test_deepstream_component_uses_long_lived_production_duration(
    tmp_path: Path, monkeypatch
) -> None:
    import live_operator.cli as cli_module

    observed = []

    class Process:
        pid = 123

    version = SimpleNamespace(
        deepstream_binary=tmp_path / "pipeline",
        deepstream_binary_sha256="a" * 64,
        pose_plan=tmp_path / "pose.plan",
        phone_engine=tmp_path / "phone.engine",
        calibration_dir=tmp_path / "calibrations",
    )
    monkeypatch.setattr(cli_module, "resolve_current_inference_version", lambda: version)
    monkeypatch.setattr(
        cli_module.subprocess,
        "Popen",
        lambda command, **kwargs: observed.append((command, kwargs)) or Process(),
    )
    monkeypatch.setattr(
        cli_module,
        "capture_identity",
        lambda process, owner_token=None: ProcessIdentity(process.pid, "start", 123, owner_token),
    )

    RuntimeHooks().start_component("deepstream", tmp_path, _camera_config())

    assert observed[0][1]["env"]["DURATION_SEC"] == "31536000"


def test_deepstream_component_launches_from_active_release(
    tmp_path: Path, monkeypatch
) -> None:
    import live_operator.cli as cli_module

    release_root = tmp_path / "release"
    launcher = (
        release_root
        / "deepstream"
        / "custom_pipeline"
        / "scripts"
        / "run_container_50p2.sh"
    )
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(
        cli_module, "__file__", str(release_root / "live_operator" / "cli.py")
    )
    monkeypatch.setattr(
        cli_module.os,
        "access",
        lambda path, mode: Path(path) == launcher and mode == os.X_OK,
    )
    version = SimpleNamespace(
        deepstream_binary=tmp_path / "pipeline",
        deepstream_binary_sha256="a" * 64,
        pose_plan=tmp_path / "pose.plan",
        phone_engine=tmp_path / "phone.engine",
        calibration_dir=tmp_path / "calibrations",
    )
    observed = []

    class Process:
        pid = 123

    monkeypatch.setattr(cli_module, "resolve_current_inference_version", lambda: version)
    monkeypatch.setattr(
        cli_module.subprocess,
        "Popen",
        lambda command, **kwargs: observed.append((command, kwargs)) or Process(),
    )
    monkeypatch.setattr(
        cli_module,
        "capture_identity",
        lambda process, owner_token=None: ProcessIdentity(
            process.pid, "start", 123, owner_token
        ),
    )

    run_dir = tmp_path / "run"
    RuntimeHooks().start_component("deepstream", run_dir, _camera_config())

    command, kwargs = observed[0]
    environment = kwargs["env"]
    assert command == [str(launcher)]
    assert environment["SOURCE_DIR"] == str(
        release_root / "deepstream" / "custom_pipeline"
    )
    assert environment["LIVE_OPERATOR_PYTHONPATH"] == str(release_root)
    assert environment["LOCAL_RELAY_BASE"] == "rtsp://127.0.0.1:8554"
    assert environment["OUTPUT_DIR"] == str(run_dir / "inference")
    assert environment["DEEPSTREAM_BINARY"] == str(version.deepstream_binary)
    assert environment["DEEPSTREAM_BINARY_SHA256"] == version.deepstream_binary_sha256
    assert environment["POSE_ENGINE"] == str(version.pose_plan)
    assert environment["PHONE_ENGINE"] == str(version.phone_engine)
    assert environment["CALIB_DIR"] == str(version.calibration_dir)


@pytest.mark.parametrize(
    ("create_launcher", "executable", "message"),
    (
        (False, False, "DeepStream launcher not found"),
        (True, False, "DeepStream launcher is not executable"),
    ),
)
def test_deepstream_component_rejects_invalid_release_launcher_before_spawn(
    tmp_path: Path,
    monkeypatch,
    create_launcher: bool,
    executable: bool,
    message: str,
) -> None:
    import live_operator.cli as cli_module

    release_root = tmp_path / "release"
    launcher = (
        release_root
        / "deepstream"
        / "custom_pipeline"
        / "scripts"
        / "run_container_50p2.sh"
    )
    if create_launcher:
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(
        cli_module, "__file__", str(release_root / "live_operator" / "cli.py")
    )
    monkeypatch.setattr(cli_module.os, "access", lambda path, mode: executable)
    monkeypatch.setattr(
        cli_module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Popen must not be called"),
    )

    with pytest.raises(LifecycleError, match=message):
        RuntimeHooks().start_component("deepstream", tmp_path / "run", object())


def test_latest_stats_are_converted_to_seven_camera_dashboard_status(tmp_path: Path) -> None:
    stats_path = tmp_path / "live_stats.jsonl"
    stats_path.write_text(
        "not-json\n"
        + json.dumps({
            "elapsed_sec": 12.5,
            "streams": [
                {
                    "stream_index": index,
                    "view": f"view-{index}",
                    "processed_fps": 7.9 + index / 100,
                    "capture_fps": 8.0,
                    "source_errors": 1 if index == 3 else 0,
                    "latency_p95_ms": 31.0 + index,
                }
                for index in range(7)
            ],
        })
        + "\n",
        encoding="utf-8",
    )

    latest = _read_latest_json_line(stats_path)
    status = _build_dashboard_status(tmp_path.name, latest)

    assert status["state"] == "running"
    assert status["aggregate_fps"] == pytest.approx(sum(7.9 + i / 100 for i in range(7)))
    assert [camera["relay"] for camera in status["cameras"]] == [
        f"camera{index:02d}" for index in range(1, 8)
    ]
    assert status["cameras"][3]["status"] == "degraded"
    assert status["cameras"][0]["p95_latency_ms"] == 31.0


def test_dashboard_status_preserves_stats_file_timestamp() -> None:
    status = _build_dashboard_status(
        "live-run",
        {"streams": [{"stream_index": 0, "processed_fps": 8.0}]},
        updated_at=123.5,
    )

    assert status["updated_at"] == 123.5


def test_runtime_verification_uses_seven_stream_stats_not_alarm_events(
    tmp_path: Path, monkeypatch
) -> None:
    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()
    (inference_dir / "frame_events.jsonl").write_text("", encoding="utf-8")
    stats_path = inference_dir / "live_stats.jsonl"
    stats_path.write_text("", encoding="utf-8")
    hooks = _configured_hooks(runtime_ready_timeout=1.0)
    monkeypatch.setattr(hooks, "is_same_process", lambda _identity: True)

    def publish_stats() -> None:
        time.sleep(0.05)
        stats_path.write_text(
            json.dumps({
                "streams": [
                    {"stream_index": index, "processed": 1, "processed_fps": 8.0}
                    for index in range(7)
                ]
            }) + "\n",
            encoding="utf-8",
        )

    writer = threading.Thread(target=publish_stats)
    writer.start()
    hooks.verify_runtime(ProcessIdentity(10, "token", 10), tmp_path)
    writer.join(1)


def test_dashboard_status_starts_with_seven_waiting_cameras() -> None:
    status = _build_dashboard_status("live-run", None)

    assert status["state"] == "starting"
    assert status["aggregate_fps"] == 0.0
    assert len(status["cameras"]) == 7
    assert all(camera["status"] == "waiting" for camera in status["cameras"])


def test_runtime_progress_accepts_truncated_stats_after_restart(
    tmp_path: Path, monkeypatch
) -> None:
    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()
    stats_path = inference_dir / "live_stats.jsonl"
    stats_path.write_text("x" * 10000, encoding="utf-8")
    hooks = _configured_hooks(runtime_ready_timeout=1.0)
    monkeypatch.setattr(hooks, "is_same_process", lambda _identity: True)

    def replace_stats() -> None:
        time.sleep(0.05)
        stats_path.write_text(
            json.dumps({
                "streams": [
                    {"stream_index": index, "processed": 1, "processed_fps": 8.0}
                    for index in range(7)
                ]
            }) + "\n",
            encoding="utf-8",
        )

    writer = threading.Thread(target=replace_stats)
    writer.start()
    hooks.verify_runtime(ProcessIdentity(10, "token", 10), tmp_path)
    writer.join(1)


def test_http_self_check_retries_until_dashboard_is_listening(
    tmp_path: Path, monkeypatch
) -> None:
    attempts = 0

    class Response:
        def __init__(self, status=200):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Opener:
        def open(self, request, timeout):
            nonlocal attempts
            attempts += 1
            assert timeout <= 5
            if attempts == 1:
                raise urllib.error.URLError(ConnectionRefusedError())
            return Response(206 if request.get_header("Range") else 200)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    hooks = RuntimeHooks(http_ready_timeout=1.0)
    hooks.run_dir = tmp_path

    hooks.http_self_check()

    assert attempts == 4


def test_docker_remove_failure_is_not_ignored(monkeypatch) -> None:
    import live_operator.cli as cli_module

    monkeypatch.setattr(cli_module, "stop_process", lambda identity: None)

    def run(command, **kwargs):
        if command[1] == "ps":
            return subprocess.CompletedProcess(command, 0, stdout="container-id\n", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="remove failed")

    monkeypatch.setattr(cli_module.subprocess, "run", run)
    with pytest.raises(LifecycleError, match="docker rm"):
        RuntimeHooks.stop_component(
            "deepstream", ProcessIdentity(10, "token", 410, "2" * 32)
        )


def test_empty_docker_query_is_not_owned(monkeypatch) -> None:
    import live_operator.cli as cli_module

    monkeypatch.setattr(cli_module, "owned_process_group_exists", lambda identity: False)
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    assert not RuntimeHooks.has_owned_processes(
        "deepstream", ProcessIdentity(10, "token", 410, "3" * 32)
    )


def test_drain_requires_same_service_fresh_stop_ack_then_zero(tmp_path: Path) -> None:
    from live_operator.cli import _write_worker_status

    hooks = RuntimeHooks()
    hooks.run_dir = tmp_path
    status = tmp_path / "dashboard" / "worker_status.json"
    status.parent.mkdir(parents=True)
    service_instance = "4" * 32
    requested_at = hooks.block_new_events(service_instance)
    status.write_text(
        json.dumps({
            "service_instance": service_instance,
            "updated_at": requested_at,
            "accepting": True,
            "active": 0,
        }),
        encoding="utf-8",
    )

    def acknowledge() -> None:
        time.sleep(0.05)
        _write_worker_status(status, service_instance, accepting=False, active=1)
        time.sleep(0.05)
        _write_worker_status(status, service_instance, accepting=False, active=0)

    writer = threading.Thread(target=acknowledge)
    writer.start()
    hooks.drain_clips(1.0, service_instance, requested_at)
    writer.join(1)


@pytest.mark.parametrize(
    "payload",
    (
        None,
        "not-json",
        json.dumps({"service_instance": "5" * 32, "updated_at": 0, "accepting": False, "active": 0}),
        json.dumps({"service_instance": "6" * 32, "updated_at": time.time(), "accepting": False, "active": 0}),
    ),
)
def test_drain_missing_malformed_stale_or_wrong_instance_times_out(
    tmp_path: Path, payload: str | None
) -> None:
    hooks = RuntimeHooks()
    hooks.run_dir = tmp_path
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir(parents=True)
    if payload is not None:
        (dashboard / "worker_status.json").write_text(payload, encoding="utf-8")
    with pytest.raises(LifecycleError, match="forcing stop"):
        hooks.drain_clips(0.12, "5" * 32, time.time())


def test_drain_timeout_forces_stop_and_records_result(tmp_path: Path) -> None:
    class TimeoutHooks(Hooks):
        def drain_clips(self, timeout, service_instance, requested_at):
            self.calls.append(f"drain:{timeout}")
            raise LifecycleError("clip drain handshake timed out; forcing stop")

    hooks = TimeoutHooks()
    lifecycle = manager(tmp_path, hooks)
    lifecycle.start(tmp_path / "config", tmp_path / "run")
    hooks.calls.clear()
    result = lifecycle.stop()
    assert result["state"] == "stopped"
    assert result["forced"] is True
    assert hooks.calls[-3:] == ["stop:deepstream", "stop:services", "stop:mediamtx"]
    recorded = json.loads((tmp_path / "run" / "dashboard" / "operator_stop.json").read_text())
    assert recorded["forced"] is True


def test_cleanup_failure_persists_stop_failed_state(tmp_path: Path) -> None:
    class RemovalFailureHooks(Hooks):
        def stop_component(self, name, identity):
            self.calls.append(f"stop:{name}")
            if name == "deepstream":
                raise LifecycleError("docker rm failed for owned container")
            self.stopped.add((identity.pid, identity.start_token))

    hooks = RemovalFailureHooks()
    lifecycle = manager(tmp_path, hooks)
    lifecycle.start(tmp_path / "config", tmp_path / "run")
    with pytest.raises(LifecycleError, match="docker rm"):
        lifecycle.stop()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["state"] == "stop_failed"
    assert "deepstream" in state["error"]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX flock")
def test_state_lock_serializes_separate_processes(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    acquired = tmp_path / "acquired"
    source = str(Path(__file__).resolve().parents[2])
    holder_code = (
        "from pathlib import Path; from live_operator.processes import StateStore; import time; "
        f"s=StateStore({str(state)!r}); r=Path({str(ready)!r}); x=Path({str(release)!r}); "
        "\nwith s.lock():\n r.write_text('ready')\n while not x.exists(): time.sleep(.02)"
    )
    waiter_code = (
        "from pathlib import Path; from live_operator.processes import StateStore; "
        f"s=StateStore({str(state)!r}); a=Path({str(acquired)!r}); "
        "\nwith s.lock(): a.write_text('acquired')"
    )
    env = dict(os.environ, PYTHONPATH=source)
    holder = subprocess.Popen([sys.executable, "-c", holder_code], env=env)
    try:
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        waiter = subprocess.Popen([sys.executable, "-c", waiter_code], env=env)
        time.sleep(0.15)
        assert not acquired.exists()
        release.write_text("release")
        assert waiter.wait(2) == 0
        assert acquired.exists()
    finally:
        release.write_text("release")
        holder.wait(2)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires Linux process groups and /proc")
def test_stop_process_cleans_real_owned_process_group() -> None:
    from live_operator.processes import capture_identity, owned_process_group_exists, stop_process

    owner = "7" * 32
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        env=dict(os.environ, JIAN_KONG_OWNER_TOKEN=owner),
        start_new_session=True,
    )
    identity = capture_identity(process, owner_token=owner)
    assert owned_process_group_exists(identity)
    stop_process(identity, timeout=2.0)
    process.wait(2)
    assert not owned_process_group_exists(identity)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires Linux /proc and executable fake ffprobe")
def test_ffprobe_subprocess_reads_memfd_without_secret_argv_and_times_out(
    tmp_path: Path, monkeypatch
) -> None:
    fake = tmp_path / "ffprobe"
    secret = "SubprocessProbeSecret"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        "source=sys.argv[sys.argv.index('-i')+1]\n"
        "raw=open(source, encoding='utf-8').read()\n"
        f"assert {secret!r} in raw\n"
        f"assert {secret!r}.encode() not in open('/proc/self/cmdline','rb').read()\n"
        "print(json.dumps({'streams':[{'codec_type':'video','codec_name':'hevc','width':2560,'height':1440,'r_frame_rate':'25/1'}]}))\n",
        encoding="utf-8",
    )
    fake.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert RuntimeHooks()._probe(f"rtsp://user:{secret}@192.0.2.1/live")
    fake.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n", encoding="utf-8")
    fake.chmod(0o700)
    started = time.monotonic()
    assert not RuntimeHooks(probe_timeout=0.15)._probe("rtsp://127.0.0.1/live")
    assert time.monotonic() - started < 1.0
