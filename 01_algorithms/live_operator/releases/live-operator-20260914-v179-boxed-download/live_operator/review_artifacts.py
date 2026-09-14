"""Persist reviewed event evidence and user-confirmed fixed-phone templates."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from live_operator import false_positive_dataset


FALSE_POSITIVE_REASONS = {
    "fixed_phone",
    "model_misdetect",
    "non_screen_use",
    "phone_call",
    "other",
    "unspecified",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_REVIEWED_RESULTS = ("confirmed", "false_positive")
_TEMPLATE_SAMPLE_COUNT = 3


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _safe_camera(event: Mapping[str, Any]) -> str:
    camera = event.get("camera") or event.get("camera_id") or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(camera)) or "unknown"


def _event_date(event: Mapping[str, Any]) -> str:
    value = event.get("occurred_at")
    if not isinstance(value, str):
        return "unknown"
    try:
        parsed = datetime.fromisoformat(
            f"{value[:-1]}+00:00" if value.endswith("Z") else value
        )
    except ValueError:
        return "unknown"
    return parsed.date().isoformat()


def _copy_once(source: Path, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def reviewed_event_root(run_dir: Path) -> Path:
    return Path(run_dir).parent / "reviewed_event_pool"


def reviewed_event_dir(
    run_dir: Path, result: str, event_id: str, event: Mapping[str, Any]
) -> Path:
    if result not in _REVIEWED_RESULTS or not _SAFE_ID.fullmatch(event_id):
        raise ValueError("invalid reviewed event path")
    return reviewed_event_root(run_dir) / result / _event_date(event) / event_id


def remove_reviewed_event(
    run_dir: Path, event_id: str, event: Mapping[str, Any]
) -> None:
    """Remove only the exact reviewed-event directories for this event."""
    if not _SAFE_ID.fullmatch(event_id):
        return
    for result in _REVIEWED_RESULTS:
        destination = reviewed_event_dir(run_dir, result, event_id, event)
        if destination.is_dir():
            shutil.rmtree(destination)


def archive_reviewed_event(
    *,
    run_dir: Path,
    event_id: str,
    result: str,
    reason: str | None,
    event: Mapping[str, Any],
    clip_path: Path,
    overlay_path: Path,
    reviewed_at: str,
    category: str | None = None,
) -> Path | None:
    """Archive event-level evidence for regression and later model evaluation."""
    remove_reviewed_event(run_dir, event_id, event)
    if result not in _REVIEWED_RESULTS:
        return None
    destination = reviewed_event_dir(run_dir, result, event_id, event)
    destination.mkdir(parents=True, exist_ok=True)
    _copy_once(clip_path, destination / "event.mp4")
    _copy_once(overlay_path, destination / "overlay.json")
    metadata = {
        "event_id": event_id,
        "result": result,
        "reason": reason if result == "false_positive" else None,
        "category": category if result == "confirmed" else None,
        "reviewed_at": reviewed_at,
        "camera": _safe_camera(event),
        "occurred_at": event.get("occurred_at"),
        "source_run_id": Path(run_dir).name,
    }
    _atomic_write_json(destination / "review.json", metadata)
    return destination


def fixed_template_root(run_dir: Path) -> Path:
    return Path(run_dir).parent / "fixed_object_templates"


def _phone_box(item: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    boxes = item.get("phone_boxes")
    if not isinstance(boxes, list):
        return None
    candidates: list[tuple[float, tuple[float, float, float, float]]] = []
    for phone in boxes:
        if not isinstance(phone, Mapping):
            continue
        parsed = false_positive_dataset._box(phone.get("box"))
        if parsed is None:
            continue
        score = phone.get("phone_score", phone.get("confidence", 0.0))
        candidates.append((float(score or 0.0), parsed))
    return max(candidates, default=(0.0, None), key=lambda candidate: candidate[0])[1]


def _context_roi(
    box: tuple[float, float, float, float],
    *,
    frame_width: int | None,
    frame_height: int | None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    context_width = max(96.0, width * 5.0)
    context_height = max(96.0, height * 5.0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = center_x - context_width / 2.0
    top = center_y - context_height / 2.0
    right = center_x + context_width / 2.0
    bottom = center_y + context_height / 2.0
    if frame_width is not None:
        left = max(0.0, left)
        right = min(float(frame_width), right)
    if frame_height is not None:
        top = max(0.0, top)
        bottom = min(float(frame_height), bottom)
    return math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)


def remove_fixed_template(run_dir: Path, template_id: str) -> bool:
    if not _SAFE_ID.fullmatch(template_id):
        return False
    destination = fixed_template_root(run_dir) / template_id
    if not destination.is_dir():
        return False
    shutil.rmtree(destination)
    return True


def create_fixed_template(
    *,
    run_dir: Path,
    event_id: str,
    event: Mapping[str, Any],
    clip_path: Path,
    overlay_path: Path,
    reviewed_at: str,
    ffmpeg_bin: str | None = None,
) -> bool:
    """Create three appearance-and-context samples for a confirmed fixed phone."""
    if not _SAFE_ID.fullmatch(event_id):
        return False
    payload = _read_json(overlay_path, {})
    timeline = false_positive_dataset._timeline(payload, event)
    candidates = [item for item in timeline if _phone_box(item) is not None]
    if len(candidates) < _TEMPLATE_SAMPLE_COUNT:
        return False
    samples = false_positive_dataset._select_time_distributed_samples(candidates)
    if len(samples) < _TEMPLATE_SAMPLE_COUNT:
        return False
    samples = samples[:_TEMPLATE_SAMPLE_COUNT]
    resolved_ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if not resolved_ffmpeg:
        raise OSError("ffmpeg is required to create fixed-phone templates")

    destination = fixed_template_root(run_dir) / event_id
    destination.mkdir(parents=True, exist_ok=True)
    sample_metadata: list[dict[str, Any]] = []
    try:
        for index, sample in enumerate(samples, start=1):
            box = _phone_box(sample)
            assert box is not None
            frame_width = (
                int(sample["frame_width"])
                if isinstance(sample.get("frame_width"), int)
                else None
            )
            frame_height = (
                int(sample["frame_height"])
                if isinstance(sample.get("frame_height"), int)
                else None
            )
            roi = _context_roi(
                box, frame_width=frame_width, frame_height=frame_height
            )
            image_name = f"sample-{index:02d}.jpg"
            image_path = destination / image_name
            temporary = destination / f".{image_name}.tmp.jpg"
            false_positive_dataset._extract_crop(
                resolved_ffmpeg,
                clip_path,
                temporary,
                clip_time_sec=float(sample.get("time_sec", 0.0) or 0.0),
                roi=roi,
            )
            os.replace(temporary, image_path)
            sample_metadata.append(
                {
                    "image": image_name,
                    "time_sec": float(sample.get("time_sec", 0.0) or 0.0),
                    "frame_index": sample.get("frame_index"),
                    "phone_box": list(box),
                    "context_roi": list(roi),
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                }
            )
        _atomic_write_json(
            destination / "template.json",
            {
                "template_id": event_id,
                "event_id": event_id,
                "camera": _safe_camera(event),
                "occurred_at": event.get("occurred_at"),
                "created_at": reviewed_at,
                "source_run_id": Path(run_dir).name,
                "active": True,
                "samples": sample_metadata,
            },
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return True


def list_fixed_templates(run_dir: Path) -> list[dict[str, Any]]:
    root = fixed_template_root(run_dir)
    templates: list[dict[str, Any]] = []
    if not root.is_dir():
        return templates
    for metadata_path in root.glob("*/template.json"):
        payload = _read_json(metadata_path, {})
        template_id = payload.get("template_id") if isinstance(payload, dict) else None
        if not isinstance(template_id, str) or not _SAFE_ID.fullmatch(template_id):
            continue
        samples = payload.get("samples")
        thumbnail = None
        if isinstance(samples, list) and samples and isinstance(samples[0], dict):
            image = samples[0].get("image")
            if isinstance(image, str) and _SAFE_ID.fullmatch(Path(image).stem):
                thumbnail = f"/fixed-objects/{template_id}/{image}"
        templates.append(
            {
                "template_id": template_id,
                "event_id": payload.get("event_id"),
                "camera": payload.get("camera"),
                "occurred_at": payload.get("occurred_at"),
                "created_at": payload.get("created_at"),
                "active": bool(payload.get("active", True)),
                "sample_count": len(samples) if isinstance(samples, list) else 0,
                "thumbnail_url": thumbnail,
            }
        )
    templates.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return templates


def fixed_template_image(
    run_dir: Path, template_id: str, image_name: str
) -> Path | None:
    if (
        not _SAFE_ID.fullmatch(template_id)
        or not re.fullmatch(r"sample-\d{2}\.jpg", image_name)
    ):
        return None
    source = fixed_template_root(run_dir) / template_id / image_name
    return source if source.is_file() else None
