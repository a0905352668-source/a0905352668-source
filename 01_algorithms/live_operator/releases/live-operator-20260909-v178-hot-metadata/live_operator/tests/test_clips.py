import json
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from live_operator.clips import (
    CLIP_DURATION_TOLERANCE_SECONDS,
    ClipNotReady,
    ClipRemuxWorker,
    ClipRequest,
    MissingRecordingSegments,
    RecordingIndex,
    refresh_clip_overlay,
)
from live_operator.config import CameraConfig


BASE_TIME = datetime(2026, 7, 17, 12, 0, 0)


def create_segment(recordings_dir: Path, relay: str, start: datetime) -> Path:
    relay_dir = recordings_dir / relay
    relay_dir.mkdir(parents=True, exist_ok=True)
    path = relay_dir / f"{start:%Y-%m-%d_%H-%M-%S-%f}.mp4"
    path.write_bytes(b"fake-fmp4")
    return path


def create_window_segments(recordings_dir: Path, relay: str = "camera01") -> list[Path]:
    return [
        create_segment(recordings_dir, relay, BASE_TIME + timedelta(seconds=offset))
        for offset in range(4, 16, 2)
    ]


def request(event_id: str = "event-001", **overrides: object) -> ClipRequest:
    values: dict[str, object] = {
        "event_id": event_id,
        "relay": "camera01",
        "occurred_at": BASE_TIME + timedelta(seconds=10),
        "event_stream_time_sec": 100.0,
        "overlay": {
            "camera": "camera01",
            "bbox_timeline": [
                {"time_sec": 98.0, "frame_index": 1, "bbox": [1, 2, 3, 4]},
                {"time_sec": 100.0, "frame_index": 2, "bbox": [2, 3, 4, 5]},
                {"time_sec": 102.0, "frame_index": 3, "bbox": [3, 4, 5, 6]},
            ],
        },
    }
    values.update(overrides)
    return ClipRequest(**values)


def test_plan_uses_five_seconds_before_and_after_across_two_second_segments(
    tmp_path: Path,
) -> None:
    recordings_dir = tmp_path / "recordings"
    expected_segments = create_window_segments(recordings_dir)

    plan = RecordingIndex(recordings_dir).plan(
        request(), now=BASE_TIME + timedelta(seconds=17)
    )

    assert plan.window_start == BASE_TIME + timedelta(seconds=5)
    assert plan.window_end == BASE_TIME + timedelta(seconds=15)
    assert plan.ready_at == BASE_TIME + timedelta(seconds=17)
    assert list(plan.segments) == expected_segments
    assert plan.trim_start_seconds == pytest.approx(1.0)
    assert plan.duration_seconds == pytest.approx(10.0)


@pytest.mark.parametrize("missing_index", [0, 2, 5], ids=["leading", "internal", "trailing"])
def test_plan_rejects_leading_internal_and_trailing_recording_gaps(
    tmp_path: Path, missing_index: int
) -> None:
    recordings_dir = tmp_path / "recordings"
    segments = create_window_segments(recordings_dir)
    segments[missing_index].unlink()

    with pytest.raises(MissingRecordingSegments, match="camera01"):
        RecordingIndex(recordings_dir).plan(
            request(), now=BASE_TIME + timedelta(seconds=19)
        )


def test_plan_waits_until_the_post_event_window_is_recorded(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    create_window_segments(recordings_dir)

    with pytest.raises(ClipNotReady) as error:
        RecordingIndex(recordings_dir).plan(
            request(), now=BASE_TIME + timedelta(seconds=12)
        )

    assert error.value.ready_at == BASE_TIME + timedelta(seconds=17)
    assert error.value.wait_seconds == pytest.approx(5.0)


def test_plan_retries_transient_missing_segment_then_marks_it_permanently_missing(
    tmp_path: Path,
) -> None:
    recordings_dir = tmp_path / "recordings"
    segments = create_window_segments(recordings_dir)
    segments[-1].unlink()
    index = RecordingIndex(recordings_dir, finalization_grace_seconds=2.0)

    with pytest.raises(ClipNotReady) as transient:
        index.plan(request(), now=BASE_TIME + timedelta(seconds=17))
    assert transient.value.ready_at == BASE_TIME + timedelta(seconds=17, milliseconds=250)

    with pytest.raises(MissingRecordingSegments, match="camera01"):
        index.plan(request(), now=BASE_TIME + timedelta(seconds=19))


def test_aware_event_time_uses_the_same_timezone_for_naive_mediamtx_names(
    tmp_path: Path,
) -> None:
    recordings_dir = tmp_path / "recordings"
    create_window_segments(recordings_dir)
    local_tz = timezone(timedelta(hours=8))
    aware_request = request(
        occurred_at=(BASE_TIME + timedelta(seconds=10)).replace(tzinfo=local_tz)
    )

    plan = RecordingIndex(recordings_dir).plan(
        aware_request,
        now=(BASE_TIME + timedelta(seconds=17)).replace(tzinfo=local_tz),
    )

    assert plan.window_start.isoformat() == "2026-07-17T12:00:05+08:00"
    assert plan.window_end.isoformat() == "2026-07-17T12:00:15+08:00"


def test_utc_event_matches_mediamtx_local_wall_clock_segments(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    expected_segments = create_window_segments(recordings_dir)
    local_tz = timezone(timedelta(hours=8))
    occurred_local = (BASE_TIME + timedelta(seconds=10)).replace(tzinfo=local_tz)
    occurred_utc = occurred_local.astimezone(timezone.utc)

    plan = RecordingIndex(
        recordings_dir,
        recording_timezone=local_tz,
    ).plan(
        request(occurred_at=occurred_utc),
        now=(BASE_TIME + timedelta(seconds=17)).replace(tzinfo=local_tz),
    )

    assert list(plan.segments) == expected_segments
    assert plan.window_start == occurred_utc - timedelta(seconds=5)
    assert plan.window_end == occurred_utc + timedelta(seconds=5)


def test_plan_accepts_keyframe_aligned_four_second_recording_cadence(
    tmp_path: Path,
) -> None:
    recordings_dir = tmp_path / "recordings"
    expected = [
        create_segment(recordings_dir, "camera01", BASE_TIME + timedelta(seconds=offset))
        for offset in range(0, 21, 4)
    ]

    plan = RecordingIndex(recordings_dir).plan(
        request(), now=BASE_TIME + timedelta(seconds=20)
    )

    assert list(plan.segments) == expected[1:4]


class FakeRunner:
    def __init__(
        self,
        *,
        ffmpeg_returncode: int = 0,
        ffprobe_returncode: int = 0,
        stderr: str = "",
        delay: float = 0.0,
        probe_duration: float = 10.0,
        probe_codec: str = "h264",
        probe_pix_fmt: str = "yuv420p",
    ) -> None:
        self.ffmpeg_returncode = ffmpeg_returncode
        self.ffprobe_returncode = ffprobe_returncode
        self.stderr = stderr
        self.delay = delay
        self.probe_duration = probe_duration
        self.probe_codec = probe_codec
        self.probe_pix_fmt = probe_pix_fmt
        self.commands: list[list[str]] = []
        self.final_path: Path | None = None
        self.final_existed_during_probe: bool | None = None
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        self.commands.append(list(command))
        executable = Path(command[0]).name.lower()
        if executable.startswith("ffmpeg"):
            with self._lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            try:
                if self.delay:
                    time.sleep(self.delay)
                if self.ffmpeg_returncode == 0:
                    Path(command[-1]).write_bytes(b"remuxed-mp4")
                return subprocess.CompletedProcess(
                    command, self.ffmpeg_returncode, "", self.stderr
                )
            finally:
                with self._lock:
                    self._active -= 1
        if executable.startswith("ffprobe"):
            if self.final_path is not None:
                self.final_existed_during_probe = self.final_path.exists()
            return subprocess.CompletedProcess(
                command,
                self.ffprobe_returncode,
                json.dumps({
                    "streams": [{
                        "codec_name": self.probe_codec,
                        "pix_fmt": self.probe_pix_fmt,
                    }],
                    "format": {"duration": str(self.probe_duration)},
                }),
                self.stderr,
            )
        raise AssertionError(f"unexpected executable: {command[0]}")


def make_worker(tmp_path: Path, runner: FakeRunner) -> ClipRemuxWorker:
    recordings_dir = tmp_path / "recordings"
    create_window_segments(recordings_dir)
    run_dir = tmp_path / "run"
    runner.final_path = run_dir / "dashboard" / "clips" / "event-001.mp4"
    return ClipRemuxWorker(
        recordings_dir=recordings_dir,
        run_dir=run_dir,
        runner=runner,
        clock=lambda: BASE_TIME + timedelta(seconds=17),
        sleeper=lambda seconds: None,
    )


def test_worker_transcodes_concat_to_browser_h264_and_publishes_only_after_ffprobe(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner)
    try:
        result = worker.submit(request()).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.status == "ready"
    assert result.output_path == runner.final_path
    assert result.output_path is not None and result.output_path.exists()
    assert result.output_path.name == "event-001.mp4"
    assert runner.final_existed_during_probe is False
    assert not result.output_path.with_name("event-001.partial.mp4").exists()
    ffmpeg_command = runner.commands[0]
    assert ffmpeg_command[0] == "ffmpeg"
    assert ffmpeg_command[1:3] == ["-hide_banner", "-loglevel"]
    assert ffmpeg_command[ffmpeg_command.index("-f") + 1] == "concat"
    assert ffmpeg_command[ffmpeg_command.index("-c:v") + 1] == "h264_nvenc"
    assert ffmpeg_command[ffmpeg_command.index("-preset") + 1] == "p4"
    assert ffmpeg_command[ffmpeg_command.index("-tune") + 1] == "hq"
    assert ffmpeg_command[ffmpeg_command.index("-cq") + 1] == "20"
    assert ffmpeg_command[ffmpeg_command.index("-b:v") + 1] == "6000k"
    assert ffmpeg_command[ffmpeg_command.index("-maxrate") + 1] == "8M"
    assert ffmpeg_command[ffmpeg_command.index("-bufsize") + 1] == "16M"
    assert ffmpeg_command[ffmpeg_command.index("-pix_fmt") + 1] == "yuv420p"
    assert ffmpeg_command[ffmpeg_command.index("-movflags") + 1] == "+faststart"
    assert "copy" not in ffmpeg_command
    assert float(ffmpeg_command[ffmpeg_command.index("-ss") + 1]) == pytest.approx(1.0)
    assert float(ffmpeg_command[ffmpeg_command.index("-t") + 1]) == pytest.approx(10.0)
    assert ffmpeg_command[-1].endswith("event-001.partial.mp4")
    assert Path(runner.commands[1][0]).name == "ffprobe"


def test_worker_writes_aligned_overlay_json_atomically(tmp_path: Path) -> None:
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner)
    try:
        result = worker.submit(request()).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.overlay_path is not None
    assert json.loads(result.overlay_path.read_text(encoding="utf-8")) == {
        "event_id": "event-001",
        "relay": "camera01",
        "window_start": "2026-07-17T12:00:05",
        "window_end": "2026-07-17T12:00:15",
        "overlay": {
            "camera": "camera01",
            "bbox_timeline": [
                {"time_sec": 3.0, "frame_index": 1, "bbox": [1, 2, 3, 4]},
                {"time_sec": 5.0, "frame_index": 2, "bbox": [2, 3, 4, 5]},
                {"time_sec": 7.0, "frame_index": 3, "bbox": [3, 4, 5, 6]},
            ],
        },
    }
    assert not result.overlay_path.with_name("event-001.partial.json").exists()


def test_worker_aligns_each_overlay_sample_from_captured_wall_time(tmp_path: Path) -> None:
    runner = FakeRunner()
    wall_time_request = request(
        occurred_at=(BASE_TIME + timedelta(seconds=10)).replace(tzinfo=timezone.utc),
        overlay={
            "camera": "camera01",
            "bbox_timeline": [
                {
                    "time_sec": 999.0,
                    "captured_at": "2026-07-17T12:00:06.250Z",
                    "frame_index": 1,
                    "bbox": [1, 2, 3, 4],
                },
                {
                    "time_sec": -999.0,
                    "captured_at": "2026-07-17T12:00:11.750+00:00",
                    "frame_index": 2,
                    "bbox": [2, 3, 4, 5],
                },
            ],
        },
    )
    worker = make_worker(tmp_path, runner)
    worker.index.recording_timezone = timezone.utc
    worker.clock = lambda: (BASE_TIME + timedelta(seconds=17)).replace(
        tzinfo=timezone.utc
    )
    try:
        result = worker.submit(wall_time_request).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.overlay_path is not None
    payload = json.loads(result.overlay_path.read_text(encoding="utf-8"))
    assert [sample["time_sec"] for sample in payload["overlay"]["bbox_timeline"]] == [
        1.25,
        6.75,
    ]


def test_ready_clip_overlay_is_refreshed_with_latest_alarm_timeline(tmp_path: Path) -> None:
    overlay_path = tmp_path / "event-001.json"
    overlay_path.write_text(json.dumps({
        "event_id": "event-001",
        "relay": "camera01",
        "window_start": "2026-07-17T12:00:05",
        "window_end": "2026-07-17T12:00:15",
        "overlay": {"bbox_timeline": []},
    }), encoding="utf-8")
    event = {
        "event_id": "event-001",
        "camera": "camera01",
        "event_stream_time_sec": 100.0,
        "risk_peak": 0.87,
        "level": "alarm",
        "bbox_timeline": [
            {"time_sec": 99.875, "frame_index": 799, "bbox": [1, 2, 3, 4]},
            {"time_sec": 104.0, "frame_index": 832, "bbox": [5, 6, 7, 8]},
        ],
    }

    refresh_clip_overlay(overlay_path, event)

    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert payload["overlay"]["risk_peak"] == pytest.approx(0.87)
    assert payload["overlay"]["level"] == "alarm"
    assert [sample["time_sec"] for sample in payload["overlay"]["bbox_timeline"]] == [
        4.875,
        9.0,
    ]
    assert not overlay_path.with_name(".event-001.json.tmp").exists()


def test_ready_clip_overlay_uses_payload_window_start_for_wall_time(tmp_path: Path) -> None:
    overlay_path = tmp_path / "event-001.json"
    overlay_path.write_text(json.dumps({
        "event_id": "event-001",
        "relay": "camera01",
        "window_start": "2026-07-17T12:00:05+08:00",
        "window_end": "2026-07-17T12:00:15+08:00",
        "overlay": {"bbox_timeline": []},
    }), encoding="utf-8")
    event = {
        "event_id": "event-001",
        "event_stream_time_sec": 100.0,
        "bbox_timeline": [{
            "time_sec": 999.0,
            "captured_at": "2026-07-17T04:00:08.250Z",
            "frame_index": 799,
            "bbox": [1, 2, 3, 4],
        }],
    }

    refresh_clip_overlay(overlay_path, event)

    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert payload["overlay"]["bbox_timeline"][0]["time_sec"] == 3.25


def test_worker_waits_for_post_window_before_planning(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    create_window_segments(recordings_dir)
    runner = FakeRunner()
    sleeps: list[float] = []
    worker = ClipRemuxWorker(
        recordings_dir=recordings_dir,
        run_dir=tmp_path / "run",
        runner=runner,
        clock=lambda: BASE_TIME + timedelta(seconds=12),
        sleeper=sleeps.append,
    )
    try:
        result = worker.submit(request()).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.status == "ready"
    assert sleeps == [pytest.approx(5.0)]


def test_worker_retries_until_the_final_segment_is_closed(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    segments = create_window_segments(recordings_dir)
    final_segment = segments[-1]
    final_segment.unlink()
    runner = FakeRunner()
    current = [BASE_TIME + timedelta(seconds=15)]
    sleeps: list[float] = []

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        current[0] += timedelta(seconds=seconds)
        if current[0] >= BASE_TIME + timedelta(seconds=17):
            create_segment(recordings_dir, "camera01", BASE_TIME + timedelta(seconds=14))

    worker = ClipRemuxWorker(
        recordings_dir=recordings_dir,
        run_dir=tmp_path / "run",
        runner=runner,
        clock=lambda: current[0],
        sleeper=advance,
    )
    try:
        result = worker.submit(request()).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.status == "ready"
    assert sum(sleeps) == pytest.approx(2.0)


def test_worker_serializes_simultaneous_alarm_remuxes(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    create_window_segments(recordings_dir)
    runner = FakeRunner(delay=0.05)
    worker = ClipRemuxWorker(
        recordings_dir=recordings_dir,
        run_dir=tmp_path / "run",
        runner=runner,
        clock=lambda: BASE_TIME + timedelta(seconds=17),
        sleeper=lambda seconds: None,
    )
    try:
        first = worker.submit(request("event-001"))
        second = worker.submit(request("event-002"))
        assert first.result(timeout=2).status == "ready"
        assert second.result(timeout=2).status == "ready"
    finally:
        worker.shutdown()

    assert runner.max_active == 1


def test_worker_marks_failure_and_redacts_ffmpeg_stderr(tmp_path: Path) -> None:
    CameraConfig(
        relay="camera01",
        view="dianqi1",
        ip="192.0.2.11",
        calibration="camera_01_screen_calibration_v21.json",
        has_screen=False,
        username="clip-user",
        password="fake-clip-secret",
    )
    runner = FakeRunner(
        ffmpeg_returncode=1,
        stderr="rtsp://clip-user:fake-clip-secret@192.0.2.11 failed",
    )
    worker = make_worker(tmp_path, runner)
    try:
        result = worker.submit(request()).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.status == "failed"
    assert result.error is not None
    assert "fake-clip-secret" not in result.error
    assert "clip-user:fake-clip-secret" not in result.error
    assert "[REDACTED]" in result.error
    assert result.output_path is None
    assert runner.final_path is not None and not runner.final_path.exists()


def test_worker_does_not_publish_when_ffprobe_rejects_partial(tmp_path: Path) -> None:
    runner = FakeRunner(ffprobe_returncode=1, stderr="invalid media")
    worker = make_worker(tmp_path, runner)
    try:
        result = worker.submit(request()).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.status == "failed"
    assert result.error is not None and "ffprobe" in result.error
    assert runner.final_path is not None and not runner.final_path.exists()
    assert not runner.final_path.with_name("event-001.partial.mp4").exists()


def test_worker_rejects_probe_duration_outside_explicit_ten_second_tolerance(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(probe_duration=10.0 + CLIP_DURATION_TOLERANCE_SECONDS + 0.01)
    worker = make_worker(tmp_path, runner)
    try:
        result = worker.submit(request()).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.status == "failed"
    assert result.error is not None and "duration" in result.error


def test_worker_accepts_event_specific_extended_clip_duration(tmp_path: Path) -> None:
    runner = FakeRunner(probe_duration=20.0)
    worker = make_worker(tmp_path, runner)
    for offset in range(16, 28, 2):
        create_segment(worker.index.recordings_dir, "camera01", BASE_TIME + timedelta(seconds=offset))
    try:
        result = worker.submit(request(post_event_seconds=15.0)).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.status == "ready"


def test_worker_rejects_non_browser_video_codec(tmp_path: Path) -> None:
    runner = FakeRunner(probe_codec="hevc", probe_pix_fmt="yuvj420p")
    worker = make_worker(tmp_path, runner)
    try:
        result = worker.submit(request()).result(timeout=2)
    finally:
        worker.shutdown()

    assert result.status == "failed"
    assert result.error is not None and "browser-compatible H.264" in result.error
    assert runner.final_path is not None and not runner.final_path.exists()
