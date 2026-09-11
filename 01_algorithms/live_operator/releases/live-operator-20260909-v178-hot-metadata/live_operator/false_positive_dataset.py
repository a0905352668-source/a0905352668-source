"""Build an editable Person-ROI LabelMe sample for a reviewed false positive."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


LABELME_VERSION = "2.4.4"
ROI_SAMPLES_PER_EVENT = 3
MODEL_MISDETECT_SAMPLES_PER_EVENT = 5
_EOF_CAPTURE_MARGIN_SECONDS = 0.25


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _camera(record: Mapping[str, Any]) -> str:
    explicit = record.get("camera") or record.get("camera_id")
    if explicit is not None:
        return str(explicit)
    stream_index = record.get("stream_index")
    if isinstance(stream_index, int) and 0 <= stream_index <= 6:
        return f"camera{stream_index + 1:02d}"
    return str(record.get("input_video", "unknown"))


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
        return None
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return x1, y1, x2, y2


def _expanded_box(
    value: Any, *, width: int | None, height: int | None, ratio: float = 0.10
) -> tuple[int, int, int, int] | None:
    parsed = _box(value)
    if parsed is None:
        return None
    x1, y1, x2, y2 = parsed
    pad_x = (x2 - x1) * ratio
    pad_y = (y2 - y1) * ratio
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    if width is not None:
        x2 = min(float(width), x2 + pad_x)
    else:
        x2 += pad_x
    if height is not None:
        y2 = min(float(height), y2 + pad_y)
    else:
        y2 += pad_y
    result = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    return result if result[2] - result[0] >= 2 and result[3] - result[1] >= 2 else None


def _timeline(overlay_payload: Any, event: Mapping[str, Any]) -> list[dict[str, Any]]:
    timeline: Any = None
    if isinstance(overlay_payload, Mapping):
        overlay = overlay_payload.get("overlay")
        if isinstance(overlay, Mapping):
            timeline = overlay.get("bbox_timeline")
    if not isinstance(timeline, list):
        timeline = event.get("bbox_timeline")
    return [item for item in timeline or [] if isinstance(item, dict) and _box(item.get("bbox"))]


def _find_raw_samples(
    source: Path,
    *,
    camera: str,
    person_id: str,
    frame_indexes: set[int],
    window_start: datetime | None,
    window_end: datetime | None,
) -> dict[int, tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    if not source.is_file() or not frame_indexes:
        return {}
    found: dict[int, tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = {}
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("frame_index") not in frame_indexes:
                continue
            if _camera(record) != camera:
                continue
            captured_at = _parse_time(record.get("captured_at"))
            if window_start is not None and captured_at is not None:
                if captured_at < window_start or (window_end is not None and captured_at > window_end):
                    continue
            person = next(
                (
                    item
                    for item in record.get("persons", [])
                    if isinstance(item, dict)
                    and str(item.get("track_id", item.get("person_id"))) == person_id
                ),
                None,
            )
            if person is None:
                continue
            phones = [
                item
                for item in record.get("phones", [])
                if isinstance(item, dict)
                and str(item.get("track_id", item.get("person_id"))) == person_id
                and _box(item.get("box")) is not None
            ]
            found[int(record["frame_index"])] = (record, person, phones)
            if len(found) == len(frame_indexes):
                break
    return found


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def _deduplicate_phones(phones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        phones,
        key=lambda item: float(item.get("phone_score", item.get("confidence", 0.0)) or 0.0),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for candidate in ranked:
        candidate_box = _box(candidate.get("box"))
        if candidate_box is None:
            continue
        if any(_iou(candidate_box, _box(existing["box"])) > 0.50 for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _central_labelme_dir(run_dir: Path) -> Path:
    """Return the persistent, flat LabelMe dataset root for all live runs."""
    return Path(run_dir).parent / "reviewed_false_positive_labelme"


def _select_time_distributed_samples(
    timeline: Sequence[dict[str, Any]], *, sample_limit: int = ROI_SAMPLES_PER_EVENT
) -> list[dict[str, Any]]:
    """Keep chronologically separated person observations up to ``sample_limit``."""
    if sample_limit < 1:
        return []
    unique: dict[tuple[str, int | float], dict[str, Any]] = {}
    for item in timeline:
        frame_index = item.get("frame_index")
        if isinstance(frame_index, int):
            key: tuple[str, int | float] = ("frame", frame_index)
        else:
            key = ("time", float(item.get("time_sec", 0.0) or 0.0))
        unique.setdefault(key, item)
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            float(item.get("time_sec", 0.0) or 0.0),
            int(item.get("frame_index", -1) or -1),
        ),
    )
    if len(ordered) <= sample_limit:
        return ordered
    if sample_limit == 1:
        return [ordered[0]]
    last_index = len(ordered) - 1
    return [
        ordered[(sample_index * last_index) // (sample_limit - 1)]
        for sample_index in range(sample_limit)
    ]


def _clip_duration_seconds(ffmpeg_bin: str, clip_path: Path) -> float | None:
    """Read a clip duration when ffprobe is available.

    Event metadata can contain observations emitted just after the ten-second
    clip boundary.  FFmpeg reports success for an EOF seek but writes no
    image, so use the actual file duration to keep the capture inside it.
    """
    ffmpeg_path = Path(ffmpeg_bin)
    candidates = [ffmpeg_path.with_name("ffprobe")]
    discovered = shutil.which("ffprobe")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            completed = subprocess.run(
                [
                    str(candidate),
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(clip_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            duration = float(completed.stdout.strip())
        except (OSError, subprocess.TimeoutExpired, ValueError):
            continue
        if completed.returncode == 0 and math.isfinite(duration) and duration > 0.0:
            return duration
    return None


def _extract_crop(
    ffmpeg_bin: str,
    clip_path: Path,
    destination: Path,
    *,
    clip_time_sec: float,
    roi: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = roi
    requested_time = max(0.0, clip_time_sec)
    duration = _clip_duration_seconds(ffmpeg_bin, clip_path)
    if duration is not None:
        # Seek a little before EOF: seeking exactly to the container duration
        # is valid to FFmpeg but can produce no decodable video frame.
        requested_time = min(
            requested_time,
            max(0.0, duration - min(_EOF_CAPTURE_MARGIN_SECONDS, duration / 2.0)),
        )
    # Keep a short fallback for clips whose duration cannot be probed.
    seek_times = [requested_time]
    if requested_time >= _EOF_CAPTURE_MARGIN_SECONDS:
        seek_times.append(requested_time - _EOF_CAPTURE_MARGIN_SECONDS)
    errors: list[str] = []
    for seek_time in seek_times:
        destination.unlink(missing_ok=True)
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-threads",
            "1",
            "-ss",
            f"{seek_time:.6f}",
            "-i",
            str(clip_path),
            "-vf",
            f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-threads",
            "1",
            str(destination),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
        except subprocess.TimeoutExpired:
            errors.append("ffmpeg ROI extraction timed out")
            continue
        if (
            completed.returncode == 0
            and destination.is_file()
            and destination.stat().st_size > 0
        ):
            return
        errors.append((completed.stderr or "ffmpeg did not create the ROI image").strip())
    destination.unlink(missing_ok=True)
    error = next(
        (error for error in reversed(errors) if error),
        "ffmpeg did not create the ROI image",
    )
    raise OSError(error[-1000:])


def archive_person_roi(
    *,
    run_dir: Path,
    clip_path: Path,
    overlay_path: Path,
    archive_clip_path: Path,
    event: Mapping[str, Any],
    ffmpeg_bin: str | None = None,
    review_reason: str | None = None,
    display_event_id: str | None = None,
    reviewed_date: str | None = None,
) -> bool:
    """Save review-driven Person ROIs and LabelMe prelabels.

    A model-detection error is high-value hard-negative training data, so it
    keeps five representative frames.  Other review reasons keep three.
    """
    if review_reason == "other":
        return False
    overlay_payload = _read_json(overlay_path, {})
    timeline = _timeline(overlay_payload, event)
    if not timeline:
        return False
    event_id = str(event.get("event_id") or archive_clip_path.stem)
    camera = str(event.get("camera") or event.get("camera_id") or "unknown")
    person_id = str(event.get("person_id") or timeline[0].get("track_id") or "unknown")
    frame_indexes = {
        int(item["frame_index"])
        for item in timeline
        if isinstance(item.get("frame_index"), int)
    }
    window_start = _parse_time(overlay_payload.get("window_start")) if isinstance(overlay_payload, Mapping) else None
    window_end = _parse_time(overlay_payload.get("window_end")) if isinstance(overlay_payload, Mapping) else None
    has_embedded_prelabel = any(item.get("roi") or item.get("phone_boxes") for item in timeline)
    raw_samples = {}
    if not has_embedded_prelabel:
        raw_samples = _find_raw_samples(
            Path(run_dir) / "inference" / "frame_events.jsonl",
            camera=camera,
            person_id=person_id,
            frame_indexes=frame_indexes,
            window_start=window_start,
            window_end=window_end,
        )
    resolved_ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if not resolved_ffmpeg:
        raise OSError("ffmpeg is required to create the Person ROI sample")

    roi_dir = _central_labelme_dir(run_dir)
    sample_limit = (
        MODEL_MISDETECT_SAMPLES_PER_EVENT
        if review_reason == "model_misdetect"
        else ROI_SAMPLES_PER_EVENT
    )
    samples = _select_time_distributed_samples(timeline, sample_limit=sample_limit)
    readable_event_id = str(display_event_id or event_id)
    reviewed_suffix = str(reviewed_date or "unknown")
    archived_any = False
    for sample_number, selected in enumerate(samples, start=1):
        frame_index = int(selected.get("frame_index", -1))
        raw = raw_samples.get(frame_index)
        record, person, phones = raw if raw is not None else (
            {},
            {"roi": selected.get("roi")},
            [item for item in selected.get("phone_boxes", []) if isinstance(item, dict)],
        )
        width = int(record["width"]) if isinstance(record.get("width"), int) else (
            int(selected["frame_width"]) if isinstance(selected.get("frame_width"), int) else None
        )
        height = int(record["height"]) if isinstance(record.get("height"), int) else (
            int(selected["frame_height"]) if isinstance(selected.get("frame_height"), int) else None
        )
        raw_roi = _box(person.get("roi"))
        roi = (
            _expanded_box(raw_roi, width=width, height=height, ratio=0.0)
            if raw_roi is not None
            else _expanded_box(selected.get("bbox"), width=width, height=height)
        )
        if roi is None:
            continue
        clip_time_sec = float(selected.get("time_sec", 5.0) or 0.0)
        captured_at = _parse_time(record.get("captured_at"))
        if captured_at is not None and window_start is not None:
            clip_time_sec = (captured_at - window_start).total_seconds()
        sample_id = (
            f"{readable_event_id}__marked-fp-{reviewed_suffix}"
            f"__event-{event_id}__roi-{sample_number:02d}"
            if review_reason == "model_misdetect"
            else f"{event_id}__roi-{sample_number:02d}"
        )
        image_path = roi_dir / f"{sample_id}.jpg"
        label_path = roi_dir / f"{sample_id}.json"
        if image_path.is_file() and label_path.is_file():
            archived_any = True
            continue
        if image_path.exists() != label_path.exists():
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
        roi_dir.mkdir(parents=True, exist_ok=True)
        temporary_image = roi_dir / f".{sample_id}.tmp.jpg"
        temporary_label = roi_dir / f".{sample_id}.tmp.json"
        x1, y1, x2, y2 = roi
        crop_width, crop_height = x2 - x1, y2 - y1
        shapes: list[dict[str, Any]] = []
        # Keep the model's original box for every review reason.  In a
        # model-misdetect review it is intentionally wrong, but displaying it
        # lets the annotator remove or correct it rather than recreating the
        # context from scratch.
        prelabel_phones = _deduplicate_phones(phones)
        for phone in prelabel_phones:
            phone_box = _box(phone.get("box"))
            if phone_box is None:
                continue
            px1 = max(0.0, min(float(crop_width), phone_box[0] - x1))
            py1 = max(0.0, min(float(crop_height), phone_box[1] - y1))
            px2 = max(0.0, min(float(crop_width), phone_box[2] - x1))
            py2 = max(0.0, min(float(crop_height), phone_box[3] - y1))
            if px2 - px1 < 2.0 or py2 - py1 < 2.0:
                continue
            confidence = float(phone.get("phone_score", phone.get("confidence", 0.0)) or 0.0)
            shapes.append(
                {
                    "label": "phone",
                    "points": [[px1, py1], [px2, py1], [px2, py2], [px1, py2]],
                    "group_id": None,
                    "description": (
                        f"prelabel confidence={confidence:.4f} person_id={person_id} "
                        f"frame_index={frame_index}"
                    ),
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )
        flags: dict[str, Any] = {
            "reviewed_false_positive": True,
            "preannotated": True,
        }
        if review_reason is not None:
            flags[f"reason_{review_reason}"] = True
        label = {
            "version": LABELME_VERSION,
            "flags": flags,
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": crop_height,
            "imageWidth": crop_width,
            "description": (
                f"Person ROI from reviewed false positive; event_id={event_id}; camera={camera}; "
                f"person_id={person_id}; sample={sample_number}/{len(samples)}; "
                f"frame_index={frame_index}; clip_time_sec={clip_time_sec:.3f}; "
                f"roi_xyxy_full_frame={[x1, y1, x2, y2]}; source_clip={archive_clip_path.name}; "
                f"review_reason={review_reason or 'unspecified'}"
            ),
        }
        try:
            _extract_crop(
                resolved_ffmpeg,
                clip_path,
                temporary_image,
                clip_time_sec=clip_time_sec,
                roi=roi,
            )
            temporary_label.write_text(
                json.dumps(label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary_image, image_path)
            os.replace(temporary_label, label_path)
            archived_any = True
        finally:
            temporary_image.unlink(missing_ok=True)
            temporary_label.unlink(missing_ok=True)
    return archived_any


def remove_person_roi(archive_clip_path: Path, *, run_dir: Path | None = None) -> None:
    roi_dir = archive_clip_path.parent / "person_roi"
    event_id = archive_clip_path.stem
    (roi_dir / f"{event_id}.jpg").unlink(missing_ok=True)
    (roi_dir / f"{event_id}.json").unlink(missing_ok=True)
    try:
        roi_dir.rmdir()
    except OSError:
        pass
    if run_dir is not None:
        central_dir = _central_labelme_dir(run_dir)
        for sample_number in range(1, ROI_SAMPLES_PER_EVENT + 1):
            sample_id = f"{event_id}__roi-{sample_number:02d}"
            (central_dir / f"{sample_id}.jpg").unlink(missing_ok=True)
            (central_dir / f"{sample_id}.json").unlink(missing_ok=True)
        for path in central_dir.glob(f"*__event-{event_id}__roi-*.jpg"):
            path.unlink(missing_ok=True)
        for path in central_dir.glob(f"*__event-{event_id}__roi-*.json"):
            path.unlink(missing_ok=True)
        try:
            central_dir.rmdir()
        except OSError:
            pass
