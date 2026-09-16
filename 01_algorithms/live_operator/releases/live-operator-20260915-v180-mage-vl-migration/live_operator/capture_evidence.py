"""Chronological person-and-screen crop evidence for capture review."""

from __future__ import annotations

from dataclasses import dataclass
import io
import hashlib
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from live_operator.capture_markings import capture_markings_profile, draw_capture_markings


CAPTURE_FRAME_COUNT = 30
CAPTURE_CROP_MARGIN_RATIO = 0.25
CAPTURE_INPUT_PROFILES = {'legacy': (30, 5.0), 'video5s30': (30, 5.0), 'video10s60': (60, 10.0)}
CAPTURE_EVIDENCE_REVISION = "person-nearby-screens-clean-video-timed-5s30-10s60-jpeg92-v5"


@dataclass(frozen=True)
class CaptureSequence:
    track_id: str
    frames: tuple[Any, ...]
    source_frame_indices: tuple[int, ...]
    times: tuple[float, ...]
    screen_ids: tuple[str, ...] = ()
    crop_box: tuple[int, int, int, int] | None = None


def build_capture_frame(
    full_frame: Any,
    *,
    crop_box: tuple[int, int, int, int],
    image_size: int = 448,
) -> Any:
    from PIL import Image

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    frame = full_frame.convert("RGB")
    view = _crop(frame, crop_box, "person-screen")
    scale = min(image_size / view.width, image_size / view.height)
    resized_size = (
        max(1, int(round(view.width * scale))),
        max(1, int(round(view.height * scale))),
    )
    view = view.resize(resized_size, Image.Resampling.LANCZOS)
    evidence = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    origin = (
        (image_size - view.width) // 2,
        (image_size - view.height) // 2,
    )
    evidence.paste(view, origin)
    return evidence


def decode_capture_frames(
    video_path: Path,
    overlay: Any,
    visibility: Any,
    *,
    target_frames: int = CAPTURE_FRAME_COUNT,
    window_seconds: float = 5.0,
) -> tuple[CaptureSequence, ...]:
    import cv2
    from PIL import Image

    if type(target_frames) is not int or target_frames not in {30,60} or window_seconds not in {5.0,10.0}:
        raise ValueError('unsupported offline evidence window')
    markings = capture_markings_profile(visibility)

    overlay_root = overlay.get("overlay", overlay) if isinstance(overlay, Mapping) else {}
    timeline = overlay_root.get("bbox_timeline") if isinstance(overlay_root, Mapping) else None
    if not isinstance(timeline, list):
        return ()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("cannot open capture-review clip")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if (
            width <= 0
            or width > 4096
            or height <= 0
            or height > 2160
            or not math.isfinite(fps)
            or fps <= 0
            or fps > 120
            or frame_count <= 0
            or frame_count / fps > 30
        ):
            raise ValueError("invalid capture-review clip metadata")

        selected_by_track = _select_entries(timeline, fps=fps, frame_count=frame_count, target_frames=target_frames, window_seconds=window_seconds)
        if not selected_by_track:
            return ()
        screen_polygons = _visibility_screens(
            visibility, video_width=width, video_height=height
        )
        crop_specs: dict[str, tuple[int, int, int, int]] = {}
        screen_specs: dict[str, tuple[str, ...]] = {}
        for track_id, entries in selected_by_track:
            person_boxes = [
                box
                for _frame_index, _time_sec, entry in entries
                if (
                    box := _scaled_box(
                        entry.get("roi", entry.get("bbox")),
                        entry,
                        video_width=width,
                        video_height=height,
                    )
                )
                is not None
            ]
            candidates = _associated_screens(
                entries,
                screen_polygons,
                person_boxes,
                video_width=width,
                video_height=height,
            )
            if not person_boxes or not candidates:
                continue
            screen_specs[track_id] = tuple(candidates)
            crop_specs[track_id] = _joint_crop_box(
                person_boxes,
                tuple(point for polygon in candidates.values() for point in polygon),
                frame_width=width,
                frame_height=height,
            )
        targets: dict[int, list[tuple[str, int, float, Mapping[str, Any]]]] = {}
        for track_id, entries in selected_by_track:
            if track_id not in crop_specs:
                continue
            for order, (frame_index, time_sec, entry) in enumerate(entries):
                targets.setdefault(frame_index, []).append(
                    (track_id, order, time_sec, entry)
                )
        if not targets:
            return ()

        panels: dict[str, dict[int, tuple[Any, int, float]]] = {
            track_id: {} for track_id, _ in selected_by_track
        }
        maximum_index = max(targets)
        for frame_index in range(maximum_index + 1):
            ok, raw_frame = capture.read()
            if not ok:
                raise ValueError(f"capture-review decode failed at frame {frame_index}")
            frame_targets = targets.get(frame_index)
            if not frame_targets:
                continue
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(rgb).convert("RGB")
            for track_id, order, time_sec, entry in frame_targets:
                person_box = _scaled_box(
                    entry.get("roi", entry.get("bbox")),
                    entry,
                    video_width=width,
                    video_height=height,
                )
                if person_box is None:
                    continue
                evidence = build_capture_frame(
                    frame,
                    crop_box=crop_specs[track_id],
                )
                transported = io.BytesIO()
                evidence.save(transported, format="JPEG", quality=92)
                transported.seek(0)
                with Image.open(transported) as reopened:
                    final_panel = reopened.convert("RGB").copy()
                if markings != 'clean':
                    accepted_phones = []
                    raw_phones = entry.get('phone_boxes')
                    for phone in raw_phones if isinstance(raw_phones,list) else []:
                        if isinstance(phone, Mapping) and phone.get('accepted') is True:
                            box = _scaled_box(phone.get('box'),entry,video_width=width,video_height=height)
                            if box is not None:
                                accepted_phones.append(box)
                    target_box = _scaled_box(entry.get('bbox'),entry,video_width=width,video_height=height) or person_box
                    final_panel = draw_capture_markings(final_panel,crop_box=crop_specs[track_id],person_box=target_box,phone_boxes=accepted_phones,screen_polygons={sid:screen_polygons[sid] for sid in screen_specs[track_id]})
                panels[track_id][order] = (final_panel, frame_index, time_sec)

        sequences = []
        for track_id, _entries in selected_by_track:
            ordered = [panels[track_id][index] for index in sorted(panels[track_id])]
            if ordered:
                sequences.append(
                    CaptureSequence(
                        track_id=track_id,
                        frames=tuple(item[0] for item in ordered),
                        source_frame_indices=tuple(item[1] for item in ordered),
                        times=tuple(item[2] for item in ordered),
                        screen_ids=screen_specs[track_id],
                        crop_box=crop_specs[track_id],
                    )
                )
        return tuple(sequences)
    finally:
        capture.release()


def _crop(image: Any, box: tuple[int, int, int, int], label: str) -> Any:
    left, top, right, bottom = box
    left, right = max(0, int(left)), min(image.width, int(right))
    top, bottom = max(0, int(top)), min(image.height, int(bottom))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid {label} crop")
    return image.crop((left, top, right, bottom))


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _select_entries(
    timeline: list[Any], *, fps: float, frame_count: int,
    target_frames: int = CAPTURE_FRAME_COUNT, window_seconds: float = 5.0,
) -> tuple[tuple[str, tuple[tuple[int, float, Mapping[str, Any]], ...]], ...]:
    grouped: dict[str, dict[int, tuple[float, Mapping[str, Any]]]] = {}
    for raw_entry in timeline:
        if not isinstance(raw_entry, Mapping):
            continue
        time_sec = _number(raw_entry.get("time_sec"))
        if time_sec is None or time_sec < 0:
            continue
        frame_index = int(round(time_sec * fps))
        if frame_index < 0 or frame_index >= frame_count:
            continue
        track_id = str(raw_entry.get("track_id", "target"))
        existing = grouped.setdefault(track_id, {}).get(frame_index)
        if existing is None or (
            raw_entry.get("alarm") is True and existing[1].get("alarm") is not True
        ):
            grouped[track_id][frame_index] = (time_sec, raw_entry)

    selected: list[tuple[str, tuple[tuple[int, float, Mapping[str, Any]], ...]]] = []
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            -sum(entry.get("alarm") is True for _, entry in item[1].values()),
            item[0],
        ),
    )
    for track_id, by_frame in ranked[:8]:
        ordered = sorted(
            ((index, time_sec, entry) for index, (time_sec, entry) in by_frame.items()),
            key=lambda item: item[0],
        )
        alarms = [item for item in ordered if item[2].get("alarm") is True]
        if not alarms:
            continue
        center_time = alarms[0][1]
        window = [item for item in ordered if abs(item[1] - center_time) <= window_seconds/2]
        if len(window) > target_frames:
            positions = [
                round(index * (len(window) - 1) / (target_frames - 1))
                for index in range(target_frames)
            ]
            window = [window[position] for position in positions]
        selected.append((track_id, tuple(window)))
    return tuple(selected)


def process_capture_video(processor: Any, chat_text: str, sequence: CaptureSequence, *, audit: dict[str, Any] | None = None) -> Any:
    """Process a lossless video file with actual sampled-frame timestamps.

    Caller holds the reviewer's model lock. The processor is always restored
    before returning, so stage-one preprocessing is never changed.
    """
    import cv2
    import numpy as np

    count = len(sequence.frames)
    if count < 2 or count != len(sequence.times):
        raise ValueError('invalid timed sequence')
    times = [float(t)-sequence.times[0] for t in sequence.times]
    if any(not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('frame times must be strictly increasing')
    original = processor.video_processor
    class TimedVideoProcessor:
        def __getattr__(self, name: str) -> Any:
            return getattr(original,name)
        def __setattr__(self, name: str, value: Any) -> None:
            setattr(original,name,value)
        def __call__(self, **kwargs: Any) -> Any:
            output = original(**kwargs)
            if int(output['video_grid_thw'][0,0]) != count:
                raise ValueError('video processor changed frame count')
            output['frame_timestamps'] = [list(times)]
            return output

    with tempfile.TemporaryDirectory(prefix='capture-video-') as temporary:
        path = Path(temporary)/'person-screens.avi'
        writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),(count-1)/times[-1],(448,448))
        if not writer.isOpened():
            writer.release()
            raise ValueError('lossless video encoder unavailable')
        try:
            for frame in sequence.frames:
                if frame.size != (448,448):
                    raise ValueError('unexpected video frame dimensions')
                writer.write(cv2.cvtColor(np.asarray(frame.convert('RGB')),cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        if audit is not None:
            audit['video_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        processor.video_processor = TimedVideoProcessor()
        try:
            return processor(text=[chat_text],videos=[str(path)],num_frames=count,return_tensors='pt',padding=True)
        finally:
            processor.video_processor = original


def _scaled_box(
    raw_box: object,
    entry: Mapping[str, Any],
    *,
    video_width: int,
    video_height: int,
) -> tuple[int, int, int, int] | None:
    if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)) or len(raw_box) != 4:
        return None
    values = [_number(value) for value in raw_box]
    if any(value is None for value in values):
        return None
    source_width = _number(entry.get("frame_width")) or float(video_width)
    source_height = _number(entry.get("frame_height")) or float(video_height)
    if source_width <= 0 or source_height <= 0:
        return None
    left, top, right, bottom = (float(value) for value in values)
    box = (
        max(0, int(round(left * video_width / source_width))),
        max(0, int(round(top * video_height / source_height))),
        min(video_width, int(round(right * video_width / source_width))),
        min(video_height, int(round(bottom * video_height / source_height))),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def _visibility_screens(
    visibility: Any, *, video_width: int, video_height: int
) -> dict[str, tuple[tuple[float, float], ...]]:
    if not isinstance(visibility, Mapping):
        return {}
    source_width = _number(visibility.get("frame_width")) or float(video_width)
    source_height = _number(visibility.get("frame_height")) or float(video_height)

    def scaled(raw_polygon: object) -> tuple[tuple[float, float], ...] | None:
        if not isinstance(raw_polygon, Sequence) or isinstance(raw_polygon, (str, bytes)):
            return None
        points = []
        for raw_point in raw_polygon:
            if not isinstance(raw_point, Sequence) or len(raw_point) != 2:
                return None
            x, y = _number(raw_point[0]), _number(raw_point[1])
            if x is None or y is None:
                return None
            points.append((x * video_width / source_width, y * video_height / source_height))
        return tuple(points) if len(points) >= 3 else None

    screens: dict[str, tuple[tuple[float, float], ...]] = {}
    raw_screens = visibility.get("screens")
    if isinstance(raw_screens, list):
        for screen in raw_screens:
            if isinstance(screen, Mapping):
                polygon = scaled(screen.get("screen_polygon", screen.get("screen_poly")))
                screen_id = screen.get("screen_id")
                if polygon is not None and isinstance(screen_id, str) and screen_id:
                    screens[screen_id] = polygon
    return screens


def _associated_screens(
    entries: Sequence[tuple[int, float, Mapping[str, Any]]],
    screens: Mapping[str, tuple[tuple[float, float], ...]],
    person_boxes: Sequence[tuple[int, int, int, int]],
    *,
    video_width: int,
    video_height: int,
) -> dict[str, tuple[tuple[float, float], ...]]:
    # Historical IDs are not spatial evidence. Keep nearby alternatives, not one
    # assumed target. This is a 2-D crop heuristic, never a visibility verdict.
    if not screens or not person_boxes:
        return {}
    phone_centers = []
    for _, _, entry in entries:
        raw_phones = entry.get("phone_boxes")
        if not isinstance(raw_phones, list):
            continue
        for phone in raw_phones:
            if not isinstance(phone, Mapping) or phone.get("accepted") is not True:
                continue
            box = _scaled_box(
                phone.get("box"), entry,
                video_width=video_width, video_height=video_height,
            )
            if box:
                phone_centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    centers = phone_centers or [
        ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in person_boxes
    ]
    x, y = (statistics.median(p[i] for p in centers) for i in (0, 1))

    def distance(polygon: Sequence[tuple[float, float]]) -> float:
        # Point-to-polygon distance; large adjacent screens must not lose merely
        # because their centroid is farther away than a small background screen.
        inside = False
        minimum = math.inf
        for a, b in zip(polygon, (*polygon[1:], polygon[0])):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = dx * dx + dy * dy
            t = max(0.0, min(1.0, ((x-a[0])*dx + (y-a[1])*dy) / length)) if length else 0.0
            minimum = min(minimum, math.hypot(x-a[0]-t*dx, y-a[1]-t*dy))
            if (a[1] > y) != (b[1] > y) and x < dx * (y-a[1]) / dy + a[0]:
                inside = not inside
        return 0.0 if inside else minimum

    ranked = sorted(screens, key=lambda sid: (distance(screens[sid]), sid))
    allowance = max(24.0, statistics.median(b[3] - b[1] for b in person_boxes) * 0.5)
    limit = distance(screens[ranked[0]]) + allowance
    return {sid: screens[sid] for sid in ranked[:3] if distance(screens[sid]) <= limit}


def _joint_crop_box(
    person_boxes: Sequence[tuple[int, int, int, int]],
    screen_polygon: Sequence[tuple[float, float]],
    *,
    frame_width: int,
    frame_height: int,
    margin_ratio: float = CAPTURE_CROP_MARGIN_RATIO,
) -> tuple[int, int, int, int]:
    xs = [float(box[0]) for box in person_boxes] + [
        float(box[2]) for box in person_boxes
    ] + [point[0] for point in screen_polygon]
    ys = [float(box[1]) for box in person_boxes] + [
        float(box[3]) for box in person_boxes
    ] + [point[1] for point in screen_polygon]
    if not xs or not ys:
        raise ValueError("missing person-screen geometry")
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    margin_x = max(24.0, width * margin_ratio)
    margin_y = max(24.0, height * margin_ratio)
    left = max(0, int(math.floor(min(xs) - margin_x)))
    top = max(0, int(math.floor(min(ys) - margin_y)))
    right = min(frame_width, int(math.ceil(max(xs) + margin_x)))
    bottom = min(frame_height, int(math.ceil(max(ys) + margin_y)))
    if right <= left or bottom <= top:
        raise ValueError("invalid person-screen crop")
    return left, top, right, bottom
