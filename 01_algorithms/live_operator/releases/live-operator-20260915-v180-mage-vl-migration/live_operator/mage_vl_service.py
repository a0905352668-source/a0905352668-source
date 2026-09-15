"""Authenticated Mage-VL service for second-stage alarm review.

This module is intentionally isolated from the production DeepStream process.
It is run on a separate GPU host and receives one completed alarm clip plus its
overlay.  Frames are decoded once in chronological order to avoid unreliable
random seeks on long-GOP H.264 clips.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import hmac
import io
import json
import math
import os
import re
import select
import shutil
import socket
import ssl
import stat
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

from live_operator.vlm_review import (
    VLM_EVIDENCE_REVISION,
    VLMReviewResult,
    evidence_request_id,
    load_shared_secret,
    request_signature,
    response_signature,
)


PROMPT = """You are a conservative final review gate for a workplace phone-use alarm.
The chronological video frames are tight crops of the same target person around the
detector's alarm moment. The detector may be wrong; verify the visible object and
interaction across multiple frames.

Choose exactly one decision:
- KEEP_NON_CALL_PHONE_USE: clear evidence that the target actively uses a genuine
  mobile phone away from the ear, including viewing, tapping, gaming, messaging,
  holding/aiming, or filming. This remains an alarm.
- FILTER_FALSE_POSITIVE: the candidate is not a genuine phone (pen, paper/notebook
  edge, badge, cup, remote, bare hand, or another object); or no active phone use is
  visible; or a genuine phone is held/pressed at the ear in a calling posture.
- UNCERTAIN: the pixels or temporal evidence are insufficient. Do not guess from the
  detector event alone.

Return exactly one ASCII line and nothing else: LABEL=KEEP_NON_CALL_PHONE_USE,
LABEL=FILTER_FALSE_POSITIVE, or LABEL=UNCERTAIN."""
PROMPT_REVISION = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()

_LABEL_PATTERN = re.compile(
    r"(?:LABEL=)?(KEEP_NON_CALL_PHONE_USE|FILTER_FALSE_POSITIVE|UNCERTAIN)\Z"
)
_RESULT_BY_LABEL = {
    "KEEP_NON_CALL_PHONE_USE": "pass",
    "FILTER_FALSE_POSITIVE": "filter",
    "UNCERTAIN": "uncertain",
}
_SAFE_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_REQUEST_ID = re.compile(r"[A-Fa-f0-9]{64}\Z")
_EXPECTED_ARCHIVE_MEMBERS = frozenset({"request.json", "clip.mp4", "overlay.json"})
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_AUTH_WINDOW_SECONDS = 120
_UPLOAD_DEADLINE_SECONDS = 30.0
_TLS_HANDSHAKE_DEADLINE_SECONDS = 10.0
_HEADER_DEADLINE_SECONDS = 10.0
_MAX_VIDEO_WIDTH = 4096
_MAX_VIDEO_HEIGHT = 2160
_MAX_VIDEO_FPS = 120.0
_MAX_VIDEO_DURATION_SECONDS = 30.0


@dataclass(frozen=True)
class CandidateSequence:
    track_id: str
    entries: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int


def _model_directory_fingerprint(model_path: Path) -> str:
    """Hash the immutable local checkpoint and its executable remote code."""

    digest = hashlib.sha256()
    files: list[Path] = []
    for path in model_path.rglob("*"):
        if path.is_symlink():
            raise ValueError("Mage-VL model directory must not contain symlinks")
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError("Mage-VL model directory is empty")
    for path in sorted(files, key=lambda item: item.relative_to(model_path).as_posix()):
        relative = path.relative_to(model_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def select_candidate_sequences(
    overlay_payload: Mapping[str, Any],
    *,
    frame_count: int = 16,
    max_candidates: int = 8,
    target_span_seconds: float = 4.0,
    video_fps: float | None = None,
    video_frame_count: int | None = None,
) -> list[CandidateSequence]:
    """Select chronological per-person evidence that exists in the actual clip."""

    if (video_fps is None) != (video_frame_count is None):
        raise ValueError("video fps and frame count must be supplied together")
    if video_fps is not None and (
        not math.isfinite(video_fps)
        or video_fps <= 0
        or video_frame_count is None
        or video_frame_count <= 0
    ):
        raise ValueError("invalid video bounds")

    overlay = overlay_payload.get("overlay", overlay_payload)
    if not isinstance(overlay, Mapping):
        return []
    timeline = overlay.get("bbox_timeline")
    if not isinstance(timeline, list):
        return []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for raw_entry in timeline:
        if not isinstance(raw_entry, Mapping):
            continue
        time_sec = _finite_number(raw_entry.get("time_sec"))
        if time_sec is None or time_sec < 0:
            continue
        if video_fps is not None:
            source_index = int(round(time_sec * video_fps))
            if source_index < 0 or source_index >= int(video_frame_count):
                continue
        track_id = str(raw_entry.get("track_id", overlay.get("person_id", "target")))
        grouped.setdefault(track_id, []).append(raw_entry)

    # When the timeline contains more than one sample mapped to the same source
    # frame, keep a single representative.  The reviewer requires 16 genuinely
    # distinct decoded frames and must never mistake repeated metadata samples
    # for temporal evidence.
    if video_fps is not None:
        for track_id, entries in list(grouped.items()):
            by_source_frame: dict[int, Mapping[str, Any]] = {}
            for entry in entries:
                time_sec = float(_finite_number(entry.get("time_sec")) or 0.0)
                source_index = int(round(time_sec * video_fps))
                existing = by_source_frame.get(source_index)
                if existing is None or (
                    entry.get("alarm") is True and existing.get("alarm") is not True
                ):
                    by_source_frame[source_index] = entry
            grouped[track_id] = list(by_source_frame.values())

    ranked: list[tuple[tuple[float, float, float, str], str, list[Mapping[str, Any]]]] = []
    for track_id, entries in grouped.items():
        entries.sort(
            key=lambda item: (
                _finite_number(item.get("time_sec")) or 0.0,
                _finite_number(item.get("frame_index")) or 0.0,
            )
        )
        alarm_indices = [index for index, item in enumerate(entries) if item.get("alarm") is True]
        if not alarm_indices:
            continue
        alarm_count = len(alarm_indices)
        risk_peak = max(
            (_finite_number(item.get("risk_score")) or 0.0 for item in entries),
            default=0.0,
        )
        first_alarm_time = (
            _finite_number(entries[alarm_indices[0]].get("time_sec"))
            if alarm_indices
            else float("inf")
        )
        ranked.append(
            ((-float(alarm_count), -risk_peak, float(first_alarm_time), track_id), track_id, entries)
        )
    ranked.sort(key=lambda item: item[0])

    sequences: list[CandidateSequence] = []
    for _rank, track_id, entries in ranked[:max_candidates]:
        alarm_indices = [index for index, item in enumerate(entries) if item.get("alarm") is True]
        center = alarm_indices[0]
        if len(entries) > frame_count:
            # This is not the first 16 clip frames. Keep eight real timeline
            # observations before the first formal alarm and the following
            # alarm evidence, matching the corrected 40-case baseline.
            start = max(
                0,
                min(center - frame_count // 2, len(entries) - frame_count),
            )
            selected = entries[start : start + frame_count]
        else:
            # Never fabricate temporal evidence by repeating a short sequence.
            selected = entries
        if selected:
            sequences.append(CandidateSequence(track_id, tuple(selected)))
    return sequences


def _crop_box(
    entry: Mapping[str, Any],
    *,
    video_width: int,
    video_height: int,
) -> tuple[int, int, int, int] | None:
    source_width = _finite_number(entry.get("frame_width")) or float(video_width)
    source_height = _finite_number(entry.get("frame_height")) or float(video_height)
    roi = entry.get("roi")
    if isinstance(roi, Sequence) and not isinstance(roi, (str, bytes)) and len(roi) == 4:
        values = [_finite_number(item) for item in roi]
        if any(item is None for item in values):
            return None
        left, top, right, bottom = (float(item) for item in values)
    else:
        bbox = entry.get("bbox")
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            return None
        values = [_finite_number(item) for item in bbox]
        if any(item is None for item in values):
            return None
        # DeepStream persists boxes as [x1, y1, x2, y2].  Do not guess an
        # xywh variant here: the formats are ambiguous and a wrong guess can
        # crop a different person, which must fail open instead of filtering.
        left, top, right, bottom = (float(item) for item in values)
    left = int(round(left * video_width / source_width))
    right = int(round(right * video_width / source_width))
    top = int(round(top * video_height / source_height))
    bottom = int(round(bottom * video_height / source_height))
    left, right = max(0, left), min(video_width, right)
    top, bottom = max(0, top), min(video_height, bottom)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _phone_boxes_in_crop(
    entry: Mapping[str, Any],
    *,
    crop_box: tuple[int, int, int, int],
    video_width: int,
    video_height: int,
) -> list[tuple[int, int, int, int]]:
    source_width = _finite_number(entry.get("frame_width")) or float(video_width)
    source_height = _finite_number(entry.get("frame_height")) or float(video_height)
    raw_boxes = entry.get("phone_boxes")
    if not isinstance(raw_boxes, list):
        return []
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    output = []
    for raw in raw_boxes:
        if not isinstance(raw, Mapping) or raw.get("accepted") is not True:
            continue
        box = raw.get("box")
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            continue
        values = [_finite_number(value) for value in box]
        if any(value is None for value in values):
            continue
        left, top, right, bottom = (float(value) for value in values)
        left = int(round(left * video_width / source_width))
        right = int(round(right * video_width / source_width))
        top = int(round(top * video_height / source_height))
        bottom = int(round(bottom * video_height / source_height))
        left, right = max(crop_left, left), min(crop_right, right)
        top, bottom = max(crop_top, top), min(crop_bottom, bottom)
        if right <= left or bottom <= top:
            continue
        output.append(
            (left - crop_left, top - crop_top, right - crop_left, bottom - crop_top)
        )
    return output


def _phone_focus_box(
    phone_boxes: Sequence[tuple[int, int, int, int]],
    *,
    crop_width: int,
    crop_height: int,
) -> tuple[int, int, int, int] | None:
    """Return a square close-up around the strongest accepted phone candidate."""

    if not phone_boxes or crop_width <= 0 or crop_height <= 0:
        return None
    left, top, right, bottom = max(
        phone_boxes,
        key=lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1]),
    )
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    # Keep enough hand/context to distinguish active use from a paper edge or
    # pen, while making a 20-30 pixel distant candidate materially visible.
    side = min(
        max(crop_width, crop_height),
        max(48, int(round(max(width, height) * 2.75))),
    )
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    focus_left = int(round(center_x - side / 2.0))
    focus_top = int(round(center_y - side / 2.0))
    focus_left = min(max(0, focus_left), max(0, crop_width - side))
    focus_top = min(max(0, focus_top), max(0, crop_height - side))
    return (
        focus_left,
        focus_top,
        min(crop_width, focus_left + side),
        min(crop_height, focus_top + side),
    )


def _evidence_panel(
    image: Any,
    phone_boxes: Sequence[tuple[int, int, int, int]],
    *,
    image_size: int,
) -> Any:
    """Letterbox the unobstructed full-person crop used by the validated baseline."""

    from PIL import Image

    canvas = Image.new("RGB", (image_size, image_size), (128, 128, 128))

    # Preserve the detector crop's native pixels. Upscaling a distant
    # 100-150px person to 448px magnifies interpolation blur and made pens,
    # paper edges, and hands look more phone-like in regression. The validated
    # baseline only downsizes oversized crops and centers smaller ones.
    person_scale = min(1.0, image_size / image.width, image_size / image.height)
    person_size = (
        max(1, min(image_size, int(round(image.width * person_scale)))),
        max(1, min(image_size, int(round(image.height * person_scale)))),
    )
    person = (
        image.resize(person_size, Image.Resampling.LANCZOS)
        if person_size != (image.width, image.height)
        else image.copy()
    )
    person_origin = (
        (image_size - person.width) // 2,
        (image_size - person.height) // 2,
    )
    canvas.paste(person, person_origin)

    return canvas


def _validated_video_metadata(capture: Any) -> VideoMetadata:
    import cv2

    metadata = VideoMetadata(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if (
        metadata.width <= 0
        or metadata.width > _MAX_VIDEO_WIDTH
        or metadata.height <= 0
        or metadata.height > _MAX_VIDEO_HEIGHT
        or not math.isfinite(metadata.fps)
        or metadata.fps <= 0
        or metadata.fps > _MAX_VIDEO_FPS
        or metadata.frame_count <= 0
        or metadata.frame_count / metadata.fps > _MAX_VIDEO_DURATION_SECONDS
    ):
        raise ValueError("invalid review clip metadata")
    return metadata


def probe_video_metadata(video_path: Path) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("cannot open review clip")
    try:
        return _validated_video_metadata(capture)
    finally:
        capture.release()


def decode_candidate_frames(
    video_path: Path, sequences: Sequence[CandidateSequence], *, image_size: int = 448
) -> list[tuple[str, list[Any], tuple[int, ...]]]:
    """Decode once from frame zero and return letterboxed PIL crops per target."""

    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("cannot open review clip")
    try:
        metadata = _validated_video_metadata(capture)
        video_width = metadata.width
        video_height = metadata.height
        video_fps = metadata.fps
        video_frame_count = metadata.frame_count
        indexed: list[
            tuple[str, list[tuple[int, int, Mapping[str, Any]]]]
        ] = []
        targets_by_frame: dict[
            int, list[tuple[str, int, Mapping[str, Any]]]
        ] = {}
        for sequence in sequences:
            targets: list[tuple[int, int, Mapping[str, Any]]] = []
            for order, entry in enumerate(sequence.entries):
                time_sec = _finite_number(entry.get("time_sec"))
                if time_sec is None or time_sec < 0:
                    continue
                index = int(round(time_sec * video_fps))
                if index < 0 or index >= video_frame_count:
                    continue
                targets.append((index, order, entry))
                targets_by_frame.setdefault(index, []).append(
                    (sequence.track_id, order, entry)
                )
            if targets:
                indexed.append((sequence.track_id, targets))
        if not targets_by_frame:
            return []

        cropped: dict[str, dict[int, tuple[Any, int]]] = {
            track_id: {} for track_id, _targets in indexed
        }
        maximum_index = max(targets_by_frame)
        for frame_index in range(maximum_index + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"review clip decode failed at frame {frame_index}")
            for track_id, order, entry in targets_by_frame.get(frame_index, []):
                box = _crop_box(
                    entry,
                    video_width=video_width,
                    video_height=video_height,
                )
                if box is None:
                    continue
                left, top, right, bottom = box
                rgb = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb).convert("RGB")
                phone_boxes = _phone_boxes_in_crop(
                    entry,
                    crop_box=box,
                    video_width=video_width,
                    video_height=video_height,
                )
                canvas = _evidence_panel(
                    image,
                    phone_boxes,
                    image_size=image_size,
                )
                # The validated 40-event regression used quality-92 JPEG
                # person crops.  Preserve that exact transport transform in
                # memory so production inference sees the same pixels without
                # writing per-event frame files to disk.
                encoded = io.BytesIO()
                canvas.save(encoded, format="JPEG", quality=92)
                encoded.seek(0)
                with Image.open(encoded) as reopened:
                    transported = reopened.convert("RGB").copy()
                cropped[track_id][order] = (transported, frame_index)

        output: list[tuple[str, list[Any], tuple[int, ...]]] = []
        for track_id, _targets in indexed:
            ordered = [cropped[track_id][order] for order in sorted(cropped[track_id])]
            if ordered:
                output.append(
                    (
                        track_id,
                        [frame for frame, _frame_index in ordered],
                        tuple(frame_index for _frame, frame_index in ordered),
                    )
                )
        return output
    finally:
        capture.release()


class MageVLReviewer:
    """Long-lived model wrapper; every generation is serialized."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_version: str,
        gpu_weight_memory: str,
        cpu_memory: str,
    ) -> None:
        if not model_path.is_dir():
            raise ValueError("Mage-VL model directory does not exist")
        if _SAFE_EVENT_ID.fullmatch(model_version) is None:
            raise ValueError("invalid model version")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.model_path = model_path
        self.model_version = model_version
        self.model_fingerprint = _model_directory_fingerprint(model_path)
        if self.model_fingerprint[:12] not in self.model_version:
            raise ValueError("model version must contain the checkpoint fingerprint prefix")
        self._lock = threading.Lock()
        self.processor, self.model = self._load_model(
            model_path, gpu_weight_memory, cpu_memory
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        self.chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _load_model(model_path: Path, gpu_weight_memory: str, cpu_memory: str):
        import torch
        from accelerate import dispatch_model, infer_auto_device_map
        from transformers import AutoModelForCausalLM, AutoProcessor, CompressedTensorsConfig

        raw_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        raw_quantization = dict(raw_config["quantization_config"])
        # compressed-tensors eager execution has no Mage-VL INT4 kernel.  Force
        # the known-good BF16 decompression path, then split weights explicitly.
        raw_quantization["run_compressed"] = False
        quantization = CompressedTensorsConfig.from_dict(raw_quantization)
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        ).eval()
        device_map = infer_auto_device_map(
            model,
            max_memory={0: gpu_weight_memory, "cpu": cpu_memory},
            no_split_module_classes=getattr(model, "_no_split_modules", None),
            dtype=torch.bfloat16,
            offload_buffers=False,
            fallback_allocation=True,
        )
        model = dispatch_model(model, device_map=device_map, offload_buffers=False)
        model.eval()
        return processor, model

    def review(self, video_path: Path, overlay_path: Path) -> dict[str, Any]:
        import torch

        try:
            payload = json.loads(overlay_path.read_text(encoding="utf-8"))
            metadata = probe_video_metadata(video_path)
            sequences = select_candidate_sequences(
                payload,
                video_fps=metadata.fps,
                video_frame_count=metadata.frame_count,
            )
            candidate_frames = decode_candidate_frames(video_path, sequences)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return {
                "result": "uncertain",
                "label": "UNCERTAIN",
                "model_version": self.model_version,
                "prompt_revision": PROMPT_REVISION,
                "evidence_revision": VLM_EVIDENCE_REVISION,
                "candidate_count": 0,
                "candidate_labels": [],
                "evidence_complete": False,
            }
        frame_map = {
            track_id: (frames, source_indices)
            for track_id, frames, source_indices in candidate_frames
        }
        overlay = payload.get("overlay", payload)
        timeline = overlay.get("bbox_timeline", []) if isinstance(overlay, Mapping) else []
        alarm_track_ids = {
            str(entry.get("track_id", overlay.get("person_id", "target")))
            for entry in timeline
            if isinstance(entry, Mapping) and entry.get("alarm") is True
        }
        declared_raw = overlay.get("person_ids") if isinstance(overlay, Mapping) else None
        declared_count = (
            overlay.get("alarm_track_count") if isinstance(overlay, Mapping) else None
        )
        declared_ids = (
            [str(value) for value in declared_raw]
            if isinstance(declared_raw, list)
            else []
        )
        declarations_complete = (
            bool(declared_ids)
            and len(declared_ids) == len(set(declared_ids))
            and type(declared_count) is int
            and declared_count == len(declared_ids)
            and set(declared_ids) == alarm_track_ids
        )
        # The top-level event declaration is the authority for how many alarm
        # people must be reviewed.  If a timeline track disappeared or was
        # truncated, never let the remaining candidates unanimously filter the
        # whole event.
        incomplete = not declarations_complete or len(sequences) < len(alarm_track_ids)
        labels = []
        if sequences:
            with self._lock:
                for sequence in sequences:
                    frames, source_indices = frame_map.get(sequence.track_id, ([], ()))
                    if (
                        len(sequence.entries) != 16
                        or len(frames) != 16
                        or len(set(source_indices)) != 16
                    ):
                        incomplete = True
                        continue
                    inputs = self.processor(
                        text=[self.chat_text],
                        videos=[frames],
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {
                        key: (value.to(self.model.device) if hasattr(value, "to") else value)
                        for key, value in inputs.items()
                    }
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
                    with torch.inference_mode():
                        output = self.model.generate(
                            **inputs, max_new_tokens=24, do_sample=False
                        )
                    raw = self.processor.tokenizer.decode(
                        output[0, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    ).strip()
                    match = _LABEL_PATTERN.fullmatch(raw)
                    label = match.group(1) if match else "UNCERTAIN"
                    labels.append(label)
                    del output, inputs
                    gc.collect()
                    torch.cuda.empty_cache()
                    if label == "KEEP_NON_CALL_PHONE_USE":
                        break
        if "KEEP_NON_CALL_PHONE_USE" in labels:
            aggregate = "KEEP_NON_CALL_PHONE_USE"
        elif (
            labels
            and not incomplete
            and len(labels) == len(sequences)
            and all(label == "FILTER_FALSE_POSITIVE" for label in labels)
        ):
            aggregate = "FILTER_FALSE_POSITIVE"
        else:
            aggregate = "UNCERTAIN"
        return {
            "result": _RESULT_BY_LABEL[aggregate],
            "label": aggregate,
            "model_version": self.model_version,
            "prompt_revision": PROMPT_REVISION,
            "evidence_revision": VLM_EVIDENCE_REVISION,
            "candidate_count": len(candidate_frames),
            "candidate_labels": labels,
            "evidence_complete": not incomplete,
        }


class ReviewApplication:
    def __init__(
        self,
        *,
        reviewer: MageVLReviewer,
        shared_secret: bytes,
        cache_dir: Path,
        max_request_bytes: int,
        clock: Any = time.time,
    ) -> None:
        self.reviewer = reviewer
        self.shared_secret = shared_secret
        self.cache_dir = cache_dir
        self.max_request_bytes = max_request_bytes
        self.clock = clock
        self.model_fingerprint = getattr(
            reviewer,
            "model_fingerprint",
            hashlib.sha256(reviewer.model_version.encode("utf-8")).hexdigest(),
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(self.cache_dir, 0o700)
        self._cache_lock = threading.Lock()
        self._inference_slot = threading.BoundedSemaphore(1)

    def health(self) -> dict[str, Any]:
        return {
            "ready": True,
            "model_version": self.reviewer.model_version,
            "prompt_revision": PROMPT_REVISION,
            "evidence_revision": VLM_EVIDENCE_REVISION,
        }

    def authenticate(self, timestamp: str | None, signature: str | None, body: bytes) -> bool:
        if timestamp is None or signature is None or not timestamp.isdigit():
            return False
        if abs(int(timestamp) - int(self.clock())) > _AUTH_WINDOW_SECONDS:
            return False
        expected = request_signature(self.shared_secret, timestamp, body)
        return hmac.compare_digest(expected, signature)

    def review_archive(self, body: bytes) -> dict[str, Any]:
        if len(body) > self.max_request_bytes:
            raise ValueError("request is too large")
        with tempfile.TemporaryDirectory(prefix="jiankong-mage-vl-") as temporary_name:
            temporary = Path(temporary_name)
            (
                event_id,
                request_id,
                clip_path,
                overlay_path,
            ) = self._unpack(body, temporary)
            expected_request_id = evidence_request_id(
                event_id,
                clip_path,
                overlay_path,
                model_version=self.reviewer.model_version,
                prompt_revision=PROMPT_REVISION,
                evidence_revision=VLM_EVIDENCE_REVISION,
            )
            if not hmac.compare_digest(expected_request_id, request_id):
                raise ValueError("request evidence fence mismatch")
            input_sha256 = self._input_sha256(clip_path, overlay_path)
            cached = self._read_cache(event_id, request_id, input_sha256)
            if cached is not None:
                return cached
            started = time.perf_counter()
            decision = self.reviewer.review(clip_path, overlay_path)
            response = {
                "schema_version": 1,
                "event_id": event_id,
                "request_id": request_id,
                **decision,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "latency_seconds": round(time.perf_counter() - started, 6),
            }
            self._write_cache(event_id, input_sha256, response)
            return response

    @staticmethod
    def _input_sha256(clip_path: Path, overlay_path: Path) -> str:
        digest = hashlib.sha256()
        for path in (clip_path, overlay_path):
            digest.update(path.name.encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _unpack(body: bytes, destination: Path) -> tuple[str, str, Path, Path]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(body), "r")
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError("invalid request archive") from error
        with archive:
            member_names = archive.namelist()
            names = set(member_names)
            if (
                len(member_names) != len(_EXPECTED_ARCHIVE_MEMBERS)
                or names != _EXPECTED_ARCHIVE_MEMBERS
            ):
                raise ValueError("unexpected request archive members")
            info_by_name = {info.filename: info for info in archive.infolist()}
            if any(
                info.compress_type != zipfile.ZIP_STORED
                or info.file_size < 0
                or info.compress_size != info.file_size
                for info in info_by_name.values()
            ):
                raise ValueError("compressed request members are not allowed")
            if sum(info.file_size for info in info_by_name.values()) > len(body):
                raise ValueError("request archive expands beyond its size limit")
            for metadata_name in ("request.json", "overlay.json"):
                if info_by_name[metadata_name].file_size > _MAX_METADATA_BYTES:
                    raise ValueError("request metadata is too large")
            try:
                request_payload = json.loads(archive.read("request.json").decode("utf-8"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("invalid request metadata") from error
            event_id = request_payload.get("event_id") if isinstance(request_payload, dict) else None
            request_id = request_payload.get("request_id") if isinstance(request_payload, dict) else None
            prompt_revision = (
                request_payload.get("prompt_revision")
                if isinstance(request_payload, dict)
                else None
            )
            evidence_revision = (
                request_payload.get("evidence_revision")
                if isinstance(request_payload, dict)
                else None
            )
            if (
                not isinstance(event_id, str)
                or _SAFE_EVENT_ID.fullmatch(event_id) is None
                or not isinstance(request_id, str)
                or _SAFE_REQUEST_ID.fullmatch(request_id) is None
                or prompt_revision != PROMPT_REVISION
                or evidence_revision != VLM_EVIDENCE_REVISION
                or request_payload.get("schema_version") != 1
            ):
                raise ValueError("invalid request event_id")
            clip_path = destination / "clip.mp4"
            overlay_path = destination / "overlay.json"
            for source_name, target in (("clip.mp4", clip_path), ("overlay.json", overlay_path)):
                with archive.open(source_name, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            if not clip_path.stat().st_size or not overlay_path.stat().st_size:
                raise ValueError("empty review input")
            try:
                overlay_payload = json.loads(overlay_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("invalid overlay metadata") from error
            if not isinstance(overlay_payload, dict) or overlay_payload.get("event_id") != event_id:
                raise ValueError("overlay event_id mismatch")
            return event_id, request_id, clip_path, overlay_path

    def _cache_path(self, event_id: str) -> Path:
        return self.cache_dir / f"{event_id}.json"

    def _read_cache(
        self, event_id: str, request_id: str, input_sha256: str
    ) -> dict[str, Any] | None:
        with self._cache_lock:
            try:
                payload = json.loads(self._cache_path(event_id).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return None
        if (
            isinstance(payload, dict)
            and payload.get("input_sha256") == input_sha256
            and payload.get("model_fingerprint") == self.model_fingerprint
            and isinstance(payload.get("response"), dict)
        ):
            response = dict(payload["response"])
            response["request_id"] = request_id
            try:
                if response.get("schema_version") != 1 or response.get("event_id") != event_id:
                    raise ValueError("cache identity mismatch")
                VLMReviewResult(
                    event_id=event_id,
                    request_id=request_id,
                    result=response["result"],
                    label=response["label"],
                    model_version=response["model_version"],
                    prompt_revision=response["prompt_revision"],
                    evidence_revision=response["evidence_revision"],
                    reviewed_at=response["reviewed_at"],
                    latency_seconds=response.get("latency_seconds"),
                )
                if (
                    response["model_version"] != self.reviewer.model_version
                    or response["prompt_revision"] != PROMPT_REVISION
                    or response["evidence_revision"] != VLM_EVIDENCE_REVISION
                ):
                    raise ValueError("cache revision mismatch")
            except (AttributeError, KeyError, TypeError, ValueError):
                return None
            return response
        return None

    def _write_cache(self, event_id: str, input_sha256: str, response: dict[str, Any]) -> None:
        target = self._cache_path(event_id)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with self._cache_lock:
            try:
                temporary.write_text(
                    json.dumps(
                        {
                            "input_sha256": input_sha256,
                            "model_fingerprint": self.model_fingerprint,
                            "response": response,
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if os.name == "posix":
                    os.chmod(temporary, 0o600)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def make_handler(application: ReviewApplication):
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "JianKongMageVL/1"
        protocol_version = "HTTP/1.0"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(15.0)

        def handle(self) -> None:
            self._headers_complete = threading.Event()

            def expire_headers() -> None:
                if self._headers_complete.is_set():
                    return
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            timer = threading.Timer(_HEADER_DEADLINE_SECONDS, expire_headers)
            timer.daemon = True
            timer.start()
            try:
                super().handle()
            finally:
                self._headers_complete.set()
                timer.cancel()

        def _mark_headers_complete(self) -> None:
            self._headers_complete.set()

        def do_GET(self) -> None:  # noqa: N802
            self._mark_headers_complete()
            if self.path != "/healthz":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            timestamp = self.headers.get("X-Jiankong-Timestamp")
            request_digest = self.headers.get("X-Jiankong-Signature")
            if not application.authenticate(timestamp, request_digest, b""):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "authentication failed"})
                return
            self._json(
                HTTPStatus.OK,
                application.health(),
                request_digest=str(request_digest),
            )

        def do_POST(self) -> None:  # noqa: N802
            self._mark_headers_complete()
            if self.path != "/v1/review":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if self.headers.get_content_type() != "application/zip":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "invalid content type"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length <= 0 or length > application.max_request_bytes:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid content length"})
                return
            try:
                body = self._read_body(length)
            except (OSError, TimeoutError):
                self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "request body timed out"})
                return
            if len(body) != length:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "incomplete request body"})
                return
            if not application.authenticate(
                self.headers.get("X-Jiankong-Timestamp"),
                self.headers.get("X-Jiankong-Signature"),
                body,
            ):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "authentication failed"})
                return
            # Upload and authentication never occupy the only model slot.  A
            # slow or unauthenticated client therefore cannot stall inference.
            if not application._inference_slot.acquire(blocking=False):
                self.send_response(int(HTTPStatus.SERVICE_UNAVAILABLE))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Retry-After", "2")
                busy_body = b'{"error":"review service is busy"}'
                self.send_header("Content-Length", str(len(busy_body)))
                self.end_headers()
                self.wfile.write(busy_body)
                return
            try:
                try:
                    response = application.review_archive(body)
                except ValueError:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid review input"})
                    return
                except Exception:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "review failed"})
                    return
                self._json(
                    HTTPStatus.OK,
                    response,
                    request_digest=str(self.headers.get("X-Jiankong-Signature")),
                )
            finally:
                application._inference_slot.release()

        def _read_body(self, length: int) -> bytes:
            deadline = time.monotonic() + _UPLOAD_DEADLINE_SECONDS
            body = bytearray()
            try:
                while len(body) < length:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise TimeoutError("request upload deadline exceeded")
                    self.connection.settimeout(min(15.0, max(0.1, remaining_seconds)))
                    read_once = getattr(self.rfile, "read1", self.rfile.read)
                    chunk = read_once(min(64 * 1024, length - len(body)))
                    if not chunk:
                        break
                    body.extend(chunk)
            finally:
                self.connection.settimeout(15.0)
            return bytes(body)

        def _json(
            self,
            status: HTTPStatus,
            payload: Mapping[str, Any],
            *,
            request_digest: str | None = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if request_digest is not None:
                self.send_header(
                    "X-Jiankong-Response-Signature",
                    response_signature(application.shared_secret, request_digest, body),
                )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            # Avoid logging event IDs, signatures, or request paths by default.
            return

    return ReviewHandler


class BoundedReviewHTTPServer(ThreadingHTTPServer):
    """Cap accepted request threads in addition to serializing inference."""

    request_queue_size = 8

    def __init__(self, *args: Any, max_request_threads: int = 2, **kwargs: Any) -> None:
        self._request_slots = threading.BoundedSemaphore(max_request_threads)
        super().__init__(*args, **kwargs)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            if isinstance(request, ssl.SSLSocket):
                deadline = time.monotonic() + _TLS_HANDSHAKE_DEADLINE_SECONDS
                request.setblocking(False)
                while True:
                    try:
                        request.do_handshake()
                        break
                    except ssl.SSLWantReadError:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not select.select(
                            [request], [], [], remaining
                        )[0]:
                            raise TimeoutError("TLS handshake timed out")
                    except ssl.SSLWantWriteError:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not select.select(
                            [], [request], [], remaining
                        )[1]:
                            raise TimeoutError("TLS handshake timed out")
                request.settimeout(15.0)
            super().process_request_thread(request, client_address)
        except (OSError, TimeoutError, ssl.SSLError):
            self.shutdown_request(request)
        finally:
            self._request_slots.release()


def _acquire_gpu_lock(path: Path) -> int:
    if not path.is_absolute():
        raise SystemExit("GPU lock path must be absolute")
    try:
        if path.is_symlink():
            raise SystemExit("GPU lock path must not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise SystemExit("cannot open Mage-VL GPU lock") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid():
            raise SystemExit("Mage-VL GPU lock must be an owned regular file")
        os.fchmod(descriptor, 0o600)
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("Mage-VL GPU lock is already held") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jiankong-mage-vl-service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--shared-secret-file", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--gpu-weight-memory", default="3800MiB")
    parser.add_argument("--cpu-memory", default="24GiB")
    parser.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--tls-cert-file", type=Path)
    parser.add_argument("--tls-key-file", type=Path)
    parser.add_argument("--gpu-lock-file", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("invalid port")
    if not 1024 <= args.max_request_bytes <= 256 * 1024 * 1024:
        raise SystemExit("invalid max request size")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and (
        args.tls_cert_file is None or args.tls_key_file is None
    ):
        raise SystemExit("non-loopback service binding requires TLS")
    if (args.tls_cert_file is None) != (args.tls_key_file is None):
        raise SystemExit("both TLS certificate and key are required")
    secret = load_shared_secret(args.shared_secret_file)
    lock_descriptor = _acquire_gpu_lock(args.gpu_lock_file)
    server = None
    try:
        # Reserve the endpoint before allocating model memory.  Activation is
        # delayed until the reviewer is fully ready.
        server = BoundedReviewHTTPServer(
            (args.host, args.port),
            BaseHTTPRequestHandler,
            bind_and_activate=False,
        )
        server.server_bind()
        if args.tls_cert_file is not None and args.tls_key_file is not None:
            if (
                args.tls_cert_file.is_symlink()
                or args.tls_key_file.is_symlink()
                or not args.tls_cert_file.is_file()
                or not args.tls_key_file.is_file()
            ):
                raise SystemExit("TLS certificate and key must be regular non-symlink files")
            key_details = args.tls_key_file.lstat()
            if (
                key_details.st_uid != os.geteuid()
                or stat.S_IMODE(key_details.st_mode) != 0o600
            ):
                raise SystemExit("TLS private key must be owned by the service user with mode 0600")
            tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
            tls_context.load_cert_chain(
                certfile=str(args.tls_cert_file), keyfile=str(args.tls_key_file)
            )
            server.socket = tls_context.wrap_socket(
                server.socket,
                server_side=True,
                do_handshake_on_connect=False,
            )
        reviewer = MageVLReviewer(
            model_path=args.model,
            model_version=args.model_version,
            gpu_weight_memory=args.gpu_weight_memory,
            cpu_memory=args.cpu_memory,
        )
        application = ReviewApplication(
            reviewer=reviewer,
            shared_secret=secret,
            cache_dir=args.cache_dir,
            max_request_bytes=args.max_request_bytes,
        )
        server.RequestHandlerClass = make_handler(application)
        server.server_activate()
        server.daemon_threads = True
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        os.close(lock_descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
