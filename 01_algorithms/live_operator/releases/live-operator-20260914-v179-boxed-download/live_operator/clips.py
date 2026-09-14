"""Plan browser-compatible alarm clips from MediaMTX fMP4 segments."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from copy import deepcopy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from statistics import median
from threading import BoundedSemaphore, Lock
from typing import Callable, Mapping, Optional

from live_operator.config import redact_text


SEGMENT_DURATION_SECONDS = 2.0
PRE_EVENT_SECONDS = 3.0
POST_EVENT_SECONDS = 17.0
# MediaMTX segment filenames are timestamped at segment open, but a keyframe-
# aligned segment is not visible to the clip worker until it is closed.  The
# configured two-second target commonly closes near four seconds and has
# observed long-GOP tails up to twelve seconds, so keep retrying through that
# real close-time envelope before declaring the recording permanently missing.
FINALIZATION_GRACE_SECONDS = 12.0
RECORDING_RETRY_INTERVAL_SECONDS = 0.25
CLIP_DURATION_TOLERANCE_SECONDS = 0.75
BROWSER_PROXY_SUFFIX = ".browser.mp4"
_SEGMENT_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


SegmentDurationProbe = Callable[[Path], Optional[float]]


def _aligned_overlay(
    overlay: Mapping[str, object], event_stream_time_sec: float, window_start: datetime
) -> dict[str, object]:
    aligned_overlay = deepcopy(dict(overlay))
    timeline = aligned_overlay.get("bbox_timeline")
    if not isinstance(timeline, list):
        return aligned_overlay

    aligned_timeline = []
    for sample in timeline:
        if not isinstance(sample, Mapping):
            continue
        aligned = dict(sample)
        captured_at = aligned.get("captured_at")
        captured_time = None
        if isinstance(captured_at, str):
            normalized = (
                f"{captured_at[:-1]}+00:00"
                if captured_at.endswith("Z")
                else captured_at
            )
            try:
                candidate = datetime.fromisoformat(normalized)
            except ValueError:
                pass
            else:
                if candidate.tzinfo is not None and candidate.utcoffset() is not None:
                    captured_time = candidate
        source_time = aligned.get("time_sec")
        if captured_time is not None:
            aligned["time_sec"] = round(
                (captured_time - window_start).total_seconds(),
                6,
            )
        elif isinstance(source_time, (int, float)) and math.isfinite(float(source_time)):
            aligned["time_sec"] = round(
                PRE_EVENT_SECONDS + float(source_time) - float(event_stream_time_sec),
                6,
            )
        aligned_timeline.append(aligned)
    aligned_overlay["bbox_timeline"] = aligned_timeline
    return aligned_overlay


def refresh_clip_overlay(overlay_path: Path, event: Mapping[str, object]) -> None:
    """Replace a clip's early event snapshot with the latest event timeline."""
    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    event_stream_time_sec = event.get("event_stream_time_sec")
    if not isinstance(event_stream_time_sec, (int, float)) or not math.isfinite(
        float(event_stream_time_sec)
    ):
        raise ValueError("event_stream_time_sec must be finite")
    window_start_value = payload.get("window_start")
    if not isinstance(window_start_value, str):
        raise ValueError("window_start must be an ISO-8601 timestamp")
    try:
        window_start = datetime.fromisoformat(window_start_value)
    except ValueError as error:
        raise ValueError("window_start must be an ISO-8601 timestamp") from error
    payload["overlay"] = _aligned_overlay(
        event, float(event_stream_time_sec), window_start
    )

    temporary = overlay_path.with_name(f".{overlay_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, overlay_path)
    finally:
        temporary.unlink(missing_ok=True)


class ClipNotReady(RuntimeError):
    """Raised when the post-event recording window is not complete yet."""

    def __init__(self, ready_at: datetime, wait_seconds: float) -> None:
        self.ready_at = ready_at
        self.wait_seconds = wait_seconds
        super().__init__(f"clip recording is ready in {wait_seconds:.3f} seconds")


class MissingRecordingSegments(RuntimeError):
    """Raised when available segments do not continuously cover a clip window."""


class ClipQueueFull(RuntimeError):
    """No clip was accepted; leave the event on disk for a later submission."""


@dataclass(frozen=True)
class ClipRequest:
    """Request using an absolute event time and Task 4 stream time.

    MediaMTX's naive recording filenames are interpreted in the recorder
    host's local timezone. ``event_stream_time_sec`` is Task 4's stable-alarm
    stream time.
    """

    event_id: str
    relay: str
    occurred_at: datetime
    event_stream_time_sec: float
    overlay: Mapping[str, object]
    post_event_seconds: float = POST_EVENT_SECONDS

    def __post_init__(self) -> None:
        for name, value in (("event_id", self.event_id), ("relay", self.relay)):
            if not isinstance(value, str) or _SAFE_IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"invalid {name}")
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("occurred_at must be a datetime")
        if not isinstance(self.event_stream_time_sec, (int, float)) or not math.isfinite(
            float(self.event_stream_time_sec)
        ):
            raise ValueError("event_stream_time_sec must be finite")
        if not isinstance(self.overlay, Mapping):
            raise TypeError("overlay must be a mapping")
        if (
            not isinstance(self.post_event_seconds, (int, float))
            or not math.isfinite(float(self.post_event_seconds))
            or float(self.post_event_seconds) < 0
        ):
            raise ValueError("post_event_seconds must be a finite non-negative value")


@dataclass(frozen=True)
class ClipPlan:
    request: ClipRequest
    window_start: datetime
    window_end: datetime
    ready_at: datetime
    segments: tuple[Path, ...]
    trim_start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class ClipResult:
    event_id: str
    status: str
    output_path: Path | None = None
    overlay_path: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class _RecordingSegment:
    path: Path
    start: datetime
    duration_seconds: float = SEGMENT_DURATION_SECONDS

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_seconds)


class RecordingIndex:
    """Index timestamp-named MediaMTX recordings for a single clip request."""

    def __init__(
        self,
        recordings_dir: str | os.PathLike[str],
        *,
        finalization_grace_seconds: float = FINALIZATION_GRACE_SECONDS,
        retry_interval_seconds: float = RECORDING_RETRY_INTERVAL_SECONDS,
        recording_timezone: tzinfo | None = None,
        segment_duration_probe: SegmentDurationProbe | None = None,
    ) -> None:
        if finalization_grace_seconds < 0:
            raise ValueError("finalization_grace_seconds cannot be negative")
        if retry_interval_seconds <= 0:
            raise ValueError("retry_interval_seconds must be positive")
        self.recordings_dir = Path(recordings_dir)
        self.finalization_grace_seconds = finalization_grace_seconds
        self.retry_interval_seconds = retry_interval_seconds
        self.recording_timezone = recording_timezone or datetime.now().astimezone().tzinfo
        self.segment_duration_probe = (
            segment_duration_probe or self._probe_segment_duration
        )
        self._duration_cache: dict[tuple[Path, int, int], float | None] = {}

    def plan(self, request: ClipRequest, *, now: datetime | None = None) -> ClipPlan:
        window_start = request.occurred_at - timedelta(seconds=PRE_EVENT_SECONDS)
        window_end = request.occurred_at + timedelta(seconds=request.post_event_seconds)
        availability_at = window_end + timedelta(seconds=SEGMENT_DURATION_SECONDS)
        missing_deadline = availability_at + timedelta(
            seconds=self.finalization_grace_seconds
        )
        current_time = now or datetime.now(tz=request.occurred_at.tzinfo)
        if current_time < availability_at:
            raise ClipNotReady(
                ready_at=availability_at,
                wait_seconds=(availability_at - current_time).total_seconds(),
            )

        segments = self._segments(
            request.relay,
            self.recording_timezone if request.occurred_at.tzinfo is not None else None,
        )
        selected = tuple(
            segment
            for segment in segments
            if segment.start < window_end and segment.end > window_start
        )
        try:
            self._require_continuous_coverage(
                request.relay, selected, window_start, window_end
            )
        except MissingRecordingSegments:
            if current_time < missing_deadline:
                ready_at = min(
                    current_time + timedelta(seconds=self.retry_interval_seconds),
                    missing_deadline,
                )
                raise ClipNotReady(
                    ready_at=ready_at,
                    wait_seconds=(ready_at - current_time).total_seconds(),
                )
            raise
        first_start = selected[0].start
        return ClipPlan(
            request=request,
            window_start=window_start,
            window_end=window_end,
            ready_at=availability_at,
            segments=tuple(segment.path for segment in selected),
            trim_start_seconds=(window_start - first_start).total_seconds(),
            duration_seconds=(window_end - window_start).total_seconds(),
        )

    def _segments(self, relay: str, timezone_info: tzinfo | None) -> list[_RecordingSegment]:
        relay_dir = self.recordings_dir / relay
        parsed: list[tuple[Path, datetime]] = []
        for path in relay_dir.glob("*.mp4"):
            try:
                start = datetime.strptime(path.stem, _SEGMENT_TIMESTAMP_FORMAT)
            except ValueError:
                continue
            if timezone_info is not None:
                # MediaMTX encodes the recorder host's local wall time without
                # an offset; event timestamps may use a different timezone.
                start = start.replace(tzinfo=timezone_info)
            parsed.append((path, start))
        parsed.sort(key=lambda item: (item[1], str(item[0])))
        deltas = [
            (following[1] - current[1]).total_seconds()
            for current, following in zip(parsed, parsed[1:])
            if following[1] > current[1]
        ]
        nominal_duration = median(deltas) if deltas else SEGMENT_DURATION_SECONDS
        maximum_normal_span = nominal_duration * 1.5
        segments: list[_RecordingSegment] = []
        for index, (path, start) in enumerate(parsed):
            duration = nominal_duration
            if index + 1 < len(parsed):
                delta = (parsed[index + 1][1] - start).total_seconds()
                if delta > 0:
                    if delta <= maximum_normal_span:
                        duration = delta
                    else:
                        # MediaMTX's target duration is keyframe-aligned. A
                        # healthy H.264 segment can therefore span 8 or 12
                        # seconds even when the median cadence is about 4s.
                        # Probe only these outliers: a real long segment covers
                        # the interval, while a genuinely missing file retains
                        # its shorter on-disk duration and still exposes a gap.
                        probed = self._cached_segment_duration(path)
                        duration = (
                            probed
                            if probed is not None
                            else min(delta, maximum_normal_span)
                        )
            segments.append(
                _RecordingSegment(path=path, start=start, duration_seconds=duration)
            )
        return segments

    def _cached_segment_duration(self, path: Path) -> float | None:
        try:
            details = path.stat()
        except OSError:
            return None
        key = (path, details.st_size, details.st_mtime_ns)
        if key not in self._duration_cache:
            self._duration_cache[key] = self.segment_duration_probe(path)
        return self._duration_cache[key]

    @staticmethod
    def _probe_segment_duration(path: Path) -> float | None:
        try:
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        try:
            duration = float(completed.stdout.strip())
        except ValueError:
            return None
        if not math.isfinite(duration) or duration <= 0 or duration > 30.0:
            return None
        return duration

    @staticmethod
    def _require_continuous_coverage(
        relay: str,
        segments: tuple[_RecordingSegment, ...],
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        cursor = window_start
        for segment in segments:
            if segment.start > cursor:
                break
            if segment.end > cursor:
                cursor = segment.end
            if cursor >= window_end:
                return
        raise MissingRecordingSegments(
            f"recording segments for {relay} do not cover "
            f"{window_start.isoformat()} through {window_end.isoformat()}"
        )


Runner = Callable[..., subprocess.CompletedProcess[str]]
Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]


class ClipRemuxWorker:
    """Serialize browser H.264 jobs so DeepStream is never blocked by FFmpeg."""

    def __init__(
        self,
        *,
        recordings_dir: str | os.PathLike[str],
        run_dir: str | os.PathLike[str],
        runner: Runner = subprocess.run,
        clock: Clock | None = None,
        sleeper: Sleeper = time.sleep,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        max_pending: int = 8,
        ffmpeg_timeout_seconds: float = 120.0,
        ffprobe_timeout_seconds: float = 15.0,
    ) -> None:
        if (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or max_pending <= 0
        ):
            raise ValueError("max_pending must be a positive integer")
        for name, value in (
            ("ffmpeg_timeout_seconds", ffmpeg_timeout_seconds),
            ("ffprobe_timeout_seconds", ffprobe_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        self.index = RecordingIndex(recordings_dir)
        self.run_dir = Path(run_dir)
        self.runner = runner
        self.clock = clock
        self.sleeper = sleeper
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.ffmpeg_timeout_seconds = float(ffmpeg_timeout_seconds)
        self.ffprobe_timeout_seconds = float(ffprobe_timeout_seconds)
        self._pending_slots = BoundedSemaphore(max_pending)
        self._submission_lock = Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jiankong-clip-remux"
        )

    def submit(self, request: ClipRequest) -> Future[ClipResult]:
        """Accept at most max_pending running/queued jobs, without waiting."""
        with self._submission_lock:
            if self._closed:
                raise RuntimeError("cannot schedule new clips after shutdown")
            if not self._pending_slots.acquire(blocking=False):
                raise ClipQueueFull("clip worker is full; retry the event later")
            try:
                future = self._executor.submit(self._process, request)
            except BaseException:
                self._pending_slots.release()
                raise
            # Includes successful/failed results, raised errors and cancellation.
            future.add_done_callback(lambda _: self._pending_slots.release())
            return future

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._submission_lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _process(self, request: ClipRequest) -> ClipResult:
        clips_dir = self.run_dir / "dashboard" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        partial_video = clips_dir / f"{request.event_id}.partial.mp4"
        final_video = clips_dir / f"{request.event_id}.mp4"
        partial_browser_video = clips_dir / f"{request.event_id}.partial{BROWSER_PROXY_SUFFIX}"
        final_browser_video = clips_dir / f"{request.event_id}{BROWSER_PROXY_SUFFIX}"
        partial_overlay = clips_dir / f"{request.event_id}.partial.json"
        final_overlay = clips_dir / f"{request.event_id}.json"
        manifest = clips_dir / f".{request.event_id}.concat.txt"
        for temporary in (partial_video, partial_browser_video, partial_overlay, manifest):
            temporary.unlink(missing_ok=True)

        try:
            now = self.clock() if self.clock is not None else datetime.now(
                tz=request.occurred_at.tzinfo
            )
            while True:
                try:
                    plan = self.index.plan(request, now=now)
                    break
                except ClipNotReady as not_ready:
                    self.sleeper(not_ready.wait_seconds)
                    observed = (
                        self.clock()
                        if self.clock is not None
                        else datetime.now(tz=request.occurred_at.tzinfo)
                    )
                    now = max(observed, not_ready.ready_at)
            self._write_manifest(manifest, plan.segments)
            ffmpeg_result = self.runner(
                self._ffmpeg_command(plan, manifest, partial_video),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.ffmpeg_timeout_seconds,
            )
            if ffmpeg_result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed: {redact_text(ffmpeg_result.stderr or '')}"
                )
            self._validate_partial(
                partial_video,
                expected_duration_seconds=plan.duration_seconds,
            )
            browser_result = self.runner(
                self._browser_proxy_command(partial_video, partial_browser_video),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.ffmpeg_timeout_seconds,
            )
            if browser_result.returncode == 0:
                try:
                    self._validate_partial(
                        partial_browser_video,
                        expected_duration_seconds=plan.duration_seconds,
                    )
                    os.replace(partial_browser_video, final_browser_video)
                except Exception:
                    partial_browser_video.unlink(missing_ok=True)
            self._write_overlay(partial_overlay, plan)
            os.replace(partial_overlay, final_overlay)
            os.replace(partial_video, final_video)
            return ClipResult(
                event_id=request.event_id,
                status="ready",
                output_path=final_video,
                overlay_path=final_overlay,
            )
        except Exception as error:
            partial_video.unlink(missing_ok=True)
            partial_browser_video.unlink(missing_ok=True)
            partial_overlay.unlink(missing_ok=True)
            if not final_video.exists():
                final_overlay.unlink(missing_ok=True)
            return ClipResult(
                event_id=request.event_id,
                status="failed",
                error=redact_text(str(error)),
            )
        finally:
            manifest.unlink(missing_ok=True)

    def _validate_partial(
        self,
        partial_video: Path,
        *,
        expected_duration_seconds: float,
    ) -> None:
        probe_result = self.runner(
            [
                self.ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,pix_fmt:format=duration",
                "-of",
                "json",
                str(partial_video),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.ffprobe_timeout_seconds,
        )
        if probe_result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed: {redact_text(probe_result.stderr or '')}"
            )
        if not partial_video.is_file() or partial_video.stat().st_size == 0:
            raise RuntimeError("ffprobe validation failed: clip output is empty")
        try:
            payload = json.loads(probe_result.stdout or "")
            duration = float(payload["format"]["duration"])
            video = payload["streams"][0]
            codec_name = str(video["codec_name"]).lower()
            pixel_format = str(video["pix_fmt"]).lower()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("ffprobe validation failed: missing video metadata") from error
        if codec_name != "h264" or pixel_format != "yuv420p":
            raise RuntimeError(
                "ffprobe validation failed: clip is not browser-compatible H.264/yuv420p"
            )
        if abs(duration - expected_duration_seconds) > CLIP_DURATION_TOLERANCE_SECONDS:
            raise RuntimeError(
                "ffprobe validation failed: duration "
                f"{duration:.3f}s is outside {expected_duration_seconds:.3f}s "
                f"+/- {CLIP_DURATION_TOLERANCE_SECONDS:.3f}s"
            )

    def _ffmpeg_command(
        self, plan: ClipPlan, manifest: Path, partial_video: Path
    ) -> list[str]:
        return [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-ss",
            f"{plan.trim_start_seconds:.6f}",
            "-t",
            f"{plan.duration_seconds:.6f}",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "20",
            "-b:v",
            "6000k",
            "-maxrate",
            "8M",
            "-bufsize",
            "16M",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(partial_video),
        ]

    def _browser_proxy_command(
        self, source_video: Path, partial_browser_video: Path
    ) -> list[str]:
        return [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_video),
            "-an",
            "-vf",
            "scale=1920:1080:flags=lanczos",
            "-c:v",
            "h264_nvenc",
            "-profile:v",
            "baseline",
            "-level:v",
            "4.1",
            "-g",
            "25",
            "-bf",
            "0",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "23",
            "-b:v",
            "3500k",
            "-maxrate",
            "5M",
            "-bufsize",
            "10M",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(partial_browser_video),
        ]

    @staticmethod
    def _write_manifest(destination: Path, segments: tuple[Path, ...]) -> None:
        lines = []
        for segment in segments:
            escaped = segment.resolve().as_posix().replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_overlay(destination: Path, plan: ClipPlan) -> None:
        overlay = _aligned_overlay(
            plan.request.overlay,
            float(plan.request.event_stream_time_sec),
            plan.window_start,
        )
        payload = {
            "event_id": plan.request.event_id,
            "relay": plan.request.relay,
            "window_start": plan.window_start.isoformat(),
            "window_end": plan.window_end.isoformat(),
            "overlay": overlay,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
