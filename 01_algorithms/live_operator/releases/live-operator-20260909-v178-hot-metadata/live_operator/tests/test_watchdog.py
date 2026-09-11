import json
import subprocess
from pathlib import Path

import pytest

from live_operator.processes import ProcessIdentity, StateStore
from live_operator.watchdog import (
    ComponentSnapshot,
    ProcessedProgressMonitor,
    RuntimeSnapshot,
    WatchdogController,
    WatchdogProbe,
    _parser,
    _process_category,
    _read_latest_progress_json,
    decide_recovery,
    owned_gpu_compute_pids,
)


@pytest.mark.parametrize(
    ("command", "category"),
    [
        (b"/opt/jiankong/jiankong_custom_pipeline\0--config\0x\0", "pipeline"),
        (
            b"docker\0run\0image\0bash\0/opt/jiankong/run_7x8.sh\0",
            "pipeline",
        ),
        (b"bash\0/opt/jiankong/run_container_50p2.sh\0", "launcher"),
        (b"python\0-m\0live_operator.cli\0_services\0--run-dir\0x\0", "services"),
        (b"/opt/mediamtx/mediamtx\0/opt/mediamtx/config.yml\0", "mediamtx"),
    ],
)
def test_process_category_accepts_only_real_runtime_entry_points(
    command: bytes, category: str
) -> None:
    assert _process_category(command) == category


@pytest.mark.parametrize(
    "command",
    [
        b"pgrep\0-af\0live_operator.cli _services\0",
        b"grep\0jiankong_custom_pipeline\0/proc/123/cmdline\0",
        b"ps\0-C\0mediamtx\0",
        b"bash\0-c\0pgrep -f run_7x8.sh\0",
        b"python\0-c\0print('live_operator.cli _services')\0",
    ],
)
def test_process_category_ignores_diagnostic_search_commands(command: bytes) -> None:
    assert _process_category(command) is None


def snapshot_with_missing_deepstream(**overrides) -> RuntimeSnapshot:
    values = {
        "mediamtx": ComponentSnapshot(identity_alive=True),
        "deepstream": ComponentSnapshot(identity_alive=False),
        "services": ComponentSnapshot(identity_alive=True),
        "gpu_lock_free": True,
        "gpu_owned_by_current": False,
        "gpu_compute_pids": (),
        "docker_ready": True,
        "network_ready": True,
        "port_8767_free": False,
    }
    values.update(overrides)
    return RuntimeSnapshot(**values)


def snapshot_with_running_runtime(**overrides) -> RuntimeSnapshot:
    values = {
        "mediamtx": ComponentSnapshot(identity_alive=True),
        "deepstream": ComponentSnapshot(identity_alive=True),
        "services": ComponentSnapshot(identity_alive=True),
        "gpu_lock_free": False,
        "gpu_owned_by_current": True,
        "gpu_compute_pids": (4321,),
        "gpu_owned_compute_pids": (4321,),
        "docker_ready": True,
        "network_ready": True,
        "port_8767_free": False,
        "runtime_identity_key": "runtime-1",
    }
    values.update(overrides)
    return RuntimeSnapshot(**values)


def progress_payload(*processed: int) -> dict:
    return {
        "streams": [
            {"stream_index": index, "processed": value}
            for index, value in enumerate(processed)
        ]
    }


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_owned_support_services_with_missing_deepstream_trigger_full_restart() -> None:
    decision = decide_recovery(snapshot_with_missing_deepstream())

    assert decision.action == "restart_full"
    assert decision.state == "recovering"
    assert decision.reason_code == "deepstream_missing"


def test_missing_deepstream_never_restarts_around_unknown_gpu_compute() -> None:
    decision = decide_recovery(
        snapshot_with_missing_deepstream(gpu_compute_pids=(4321,))
    )

    assert decision.action == "blocked"
    assert decision.reason_code == "unknown_gpu_compute"


def test_other_partial_runtime_remains_alert_only() -> None:
    decision = decide_recovery(
        snapshot_with_missing_deepstream(
            mediamtx=ComponentSnapshot(identity_alive=False)
        )
    )

    assert decision.action == "blocked"
    assert decision.reason_code == "partial_runtime"


def test_gpu_pid_owned_by_clip_service_is_trusted_but_foreign_pid_is_not() -> None:
    result = owned_gpu_compute_pids(
        (8123, 9001, 9901),
        {
            "deepstream": (8123,),
            "services": (9001,),
            "mediamtx": (9901,),
        },
    )

    assert result == (8123, 9001)


def test_processed_progress_requires_45_real_seconds_and_growth_clears() -> None:
    clock = FakeClock()
    monitor = ProcessedProgressMonitor(monotonic=clock)
    run = ("/runs/live-1", "generation-1", "camera01", "camera02")

    assert monitor.observe(run, progress_payload(10, 10), 2) == ()
    for _ in range(100):
        assert monitor.observe(run, progress_payload(10, 10), 2) == ()
    clock.value = 10.0
    assert monitor.observe(run, progress_payload(11, 10), 2) == ()
    clock.value = 44.999
    assert monitor.observe(run, progress_payload(12, 10), 2) == ()
    clock.value = 45.0
    assert monitor.observe(run, progress_payload(13, 10), 2) == (1,)
    clock.value = 46.0
    assert monitor.observe(run, progress_payload(14, 11), 2) == ()


def test_processed_progress_new_run_and_counter_rollback_reset_baseline() -> None:
    clock = FakeClock()
    monitor = ProcessedProgressMonitor(monotonic=clock)
    first = ("/runs/live-1", "generation-1", "camera01")
    second = ("/runs/live-2", "generation-2", "camera01")

    assert monitor.observe(first, progress_payload(20), 1) == ()
    clock.value = 44.0
    assert monitor.observe(first, progress_payload(20), 1) == ()
    clock.value = 45.0
    assert monitor.observe(second, progress_payload(20), 1) == ()
    clock.value = 89.9
    assert monitor.observe(second, progress_payload(20), 1) == ()
    clock.value = 90.0
    assert monitor.observe(second, progress_payload(5), 1) == ()
    clock.value = 134.9
    assert monitor.observe(second, progress_payload(5), 1) == ()
    clock.value = 135.0
    assert monitor.observe(second, progress_payload(5), 1) == (0,)


def test_processed_progress_bad_or_missing_data_resets_baseline() -> None:
    clock = FakeClock()
    monitor = ProcessedProgressMonitor(monotonic=clock)
    run = ("/runs/live-1", "generation-1", "camera01")

    assert monitor.observe(run, progress_payload(10), 1) == ()
    clock.value = 44.0
    assert monitor.observe(run, progress_payload(10), 1) == ()
    clock.value = 45.0
    assert monitor.observe(run, {"streams": [{"stream_index": 0}]}, 1) == ()
    clock.value = 100.0
    assert monitor.observe(run, progress_payload(10), 1) == ()
    clock.value = 145.0
    assert monitor.observe(run, None, 1) == ()
    clock.value = 200.0
    assert monitor.observe(run, progress_payload(10), 1) == ()


def test_latest_progress_reader_rejects_malformed_newest_record(tmp_path: Path) -> None:
    path = tmp_path / "live_stats.jsonl"
    path.write_text(json.dumps(progress_payload(10)) + "\n{bad json}\n", encoding="utf-8")

    assert _read_latest_progress_json(path) is None
    assert _read_latest_progress_json(tmp_path / "missing.jsonl") is None


def test_watchdog_probe_reads_real_state_and_live_stats_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "live_1"
    stats_path = run_dir / "inference" / "live_stats.jsonl"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text(json.dumps(progress_payload(10, 10)) + "\n", encoding="utf-8")
    store = StateStore(tmp_path / "state.json")
    store.save(
        {
            "state": "running",
            "run_dir": str(run_dir),
            "generation_id": "generation-1",
            "relays": ["camera01", "camera02"],
            "processes": {
                name: ProcessIdentity(
                    100 + index,
                    f"start-{index}",
                    100 + index,
                    f"{index + 1:032x}",
                ).to_dict()
                for index, name in enumerate(("mediamtx", "deepstream", "services"))
            },
        }
    )
    clock = FakeClock()

    def completed(command: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout, "")

    probe = WatchdogProbe(
        state_store=store,
        same_process=lambda _identity: True,
        owned_group=lambda _identity: False,
        owned_group_pids=lambda _identity: (),
        process_scanner=lambda: {},
        gpu_lock_probe=lambda: False,
        gpu_runner=lambda command: completed(command),
        docker_runner=lambda command: completed(command),
        network_probe=lambda: True,
        port_probe=lambda _port: False,
        progress_monitor=ProcessedProgressMonitor(monotonic=clock),
    )

    assert probe.snapshot().inference_stalled_streams == ()
    clock.value = 44.0
    stats_path.write_text(json.dumps(progress_payload(11, 10)) + "\n", encoding="utf-8")
    assert probe.snapshot().inference_stalled_streams == ()
    clock.value = 45.0
    stats_path.write_text(json.dumps(progress_payload(12, 10)) + "\n", encoding="utf-8")
    assert probe.snapshot().inference_stalled_streams == (1,)
    clock.value = 46.0
    stats_path.write_text(json.dumps(progress_payload(13, 11)) + "\n", encoding="utf-8")
    assert probe.snapshot().inference_stalled_streams == ()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_parser_rejects_nonpositive_or_nonfinite_interval(value: str) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--release-root", "/tmp/release", "--interval", value])


def test_parser_accepts_small_positive_interval_without_shortening_stall_time() -> None:
    args = _parser().parse_args(
        ["--release-root", "/tmp/release", "--interval", "0.001"]
    )

    assert args.interval == pytest.approx(0.001)


def test_controller_reprobe_cancels_stall_restart_after_progress_recovers(
    tmp_path: Path,
) -> None:
    class SequenceProbe:
        def __init__(self) -> None:
            self.snapshots = [
                snapshot_with_running_runtime(inference_stalled_streams=(1,)),
                snapshot_with_running_runtime(inference_stalled_streams=()),
            ]
            self.calls = 0

        def snapshot(self) -> RuntimeSnapshot:
            self.calls += 1
            return self.snapshots.pop(0)

    probe = SequenceProbe()
    commands: list[list[str]] = []
    controller = WatchdogController(
        state_store=StateStore(tmp_path / "state.json"),
        probe=probe,  # type: ignore[arg-type]
        release_root=tmp_path,
        health_path=tmp_path / "health.json",
        watchdog_lock_path=tmp_path / "watchdog.lock",
        command_runner=lambda command, **_kwargs: commands.append(command),
        full_start=lambda: commands.append(["full-start"]),
        check_interval=0.001,
    )

    decision = controller.check_once()

    assert probe.calls == 2
    assert decision.action == "none"
    assert decision.reason_code == "healthy"
    assert commands == []


def test_sustained_owned_inference_stall_triggers_full_restart() -> None:
    decision = decide_recovery(
        snapshot_with_running_runtime(inference_stalled_streams=(1,))
    )

    assert decision.action == "restart_full"
    assert decision.state == "recovering"
    assert decision.reason_code == "inference_progress_stalled"


def test_inference_stall_without_current_gpu_lock_is_blocked() -> None:
    decision = decide_recovery(
        snapshot_with_running_runtime(
            gpu_lock_free=True,
            gpu_owned_by_current=False,
            inference_stalled_streams=(1,),
        )
    )

    assert decision.action == "blocked"
    assert decision.reason_code == "inference_stall_restart_gates"


def test_inference_stall_without_owned_gpu_compute_is_blocked() -> None:
    decision = decide_recovery(
        snapshot_with_running_runtime(
            gpu_compute_pids=(),
            gpu_owned_compute_pids=(),
            inference_stalled_streams=(1,),
        )
    )

    assert decision.action == "blocked"
    assert decision.reason_code == "inference_stall_restart_gates"


def test_inference_stall_never_restarts_around_unknown_gpu_compute() -> None:
    decision = decide_recovery(
        snapshot_with_running_runtime(
            gpu_compute_pids=(4321, 9999),
            gpu_owned_compute_pids=(4321,),
            inference_stalled_streams=(1,),
        )
    )

    assert decision.action == "blocked"
    assert decision.reason_code == "unknown_gpu_compute"


def test_inference_stall_respects_recovery_backoff() -> None:
    decision = decide_recovery(
        snapshot_with_running_runtime(inference_stalled_streams=(1,)),
        retry_allowed=False,
    )

    assert decision.action == "blocked"
    assert decision.reason_code == "retry_not_allowed"
