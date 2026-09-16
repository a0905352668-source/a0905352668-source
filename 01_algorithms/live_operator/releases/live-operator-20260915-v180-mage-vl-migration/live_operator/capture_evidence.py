"""Chronological person-and-screen crop evidence for capture review."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CAPTURE_FRAME_COUNT = 30
CAPTURE_EVIDENCE_REVISION = "person-nearest-screen-span5s-30f-jpeg92-v2"


@dataclass(frozen=True)
class CaptureSequence:
    track_id: str
    frames: tuple[Any, ...]
    source_frame_indices: tuple[int, ...]
    times: tuple[float, ...]


def build_capture_frame(
    full_frame: Any,
    *,
    crop_box: tuple[int, int, int, int],
    person_box: tuple[int, int, int, int],
    screen_polygon: Sequence[tuple[float, float]],
    image_size: int = 448,
) -> Any:
    from PIL import Image, ImageDraw

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
    crop_left, crop_top, _crop_right, _crop_bottom = crop_box

    def transported(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round((point[0] - crop_left) * scale)) + origin[0],
            int(round((point[1] - crop_top) * scale)) + origin[1],
        )

    draw = ImageDraw.Draw(evidence)
    screen_points = [transported(point) for point in screen_polygon]
    if len(screen_points) >= 3:
        draw.line(
            screen_points + [screen_points[0]], fill=(255, 0, 0), width=3
        )
    person_left, person_top, person_right, person_bottom = person_box
    draw.rectangle(
        (*transported((person_left, person_top)), *transported((person_right, person_bottom))),
        outline=(255, 165, 0),
        width=3,
    )
    return evidence


def decode_capture_frames(
    video_path: Path,
    overlay: Any,
    visibility: Any,
) -> tuple[CaptureSequence, ...]:
    import cv2
    from PIL import Image

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

        selected_by_track = _select_entries(timeline, fps=fps, frame_count=frame_count)
        if not selected_by_track:
            return ()
        screen_polygons = _visibility_screens(
            visibility, video_width=width, video_height=height
        )
        crop_specs: dict[
            str,
            tuple[tuple[int, int, int, int], tuple[tuple[float, float], ...]],
        ] = {}
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
            screen_polygon = _associated_screen(entries, screen_polygons, person_boxes)
            if not person_boxes or screen_polygon is None:
                continue
            crop_specs[track_id] = (
                _joint_crop_box(
                    person_boxes,
                    screen_polygon,
                    frame_width=width,
                    frame_height=height,
                ),
                screen_polygon,
            )
        targets: dict[int, list[tuple[str, int, float, Mapping[str, Any]]]] = {}
        for track_id, entries in selected_by_track:
            if track_id not in crop_specs:
                continue
            for order, (frame_index, time_sec, entry) in enumerate(entries):
                targets.setdefault(frame_index, []).append(
                    (track_id, order, time_sec, entry)
                )

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
                crop_box, screen_polygon = crop_specs[track_id]
                evidence = build_capture_frame(
                    frame,
                    crop_box=crop_box,
                    person_box=person_box,
                    screen_polygon=screen_polygon,
                )
                transported = io.BytesIO()
                evidence.save(transported, format="JPEG", quality=92)
                transported.seek(0)
                with Image.open(transported) as reopened:
                    final_panel = reopened.convert("RGB").copy()
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
    timeline: list[Any], *, fps: float, frame_count: int
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
        window = [item for item in ordered if abs(item[1] - center_time) <= 2.5]
        if len(window) > CAPTURE_FRAME_COUNT:
            positions = [
                round(index * (len(window) - 1) / (CAPTURE_FRAME_COUNT - 1))
                for index in range(CAPTURE_FRAME_COUNT)
            ]
            window = [window[position] for position in positions]
        selected.append((track_id, tuple(window)))
    return tuple(selected)


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


def _associated_screen(
    entries: Sequence[tuple[int, float, Mapping[str, Any]]],
    screens: Mapping[str, tuple[tuple[float, float], ...]],
    person_boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[float, float], ...] | None:
    for _frame_index, _time_sec, entry in sorted(
        entries, key=lambda item: item[2].get("alarm") is not True
    ):
        screen_id = entry.get("screen_id")
        if isinstance(screen_id, str) and screen_id in screens:
            return screens[screen_id]
    if not screens or not person_boxes:
        return None
    left, top, right, bottom = person_boxes[len(person_boxes) // 2]
    person_center = ((left + right) / 2, (top + bottom) / 2)

    def distance(polygon: Sequence[tuple[float, float]]) -> float:
        center_x = sum(point[0] for point in polygon) / len(polygon)
        center_y = sum(point[1] for point in polygon) / len(polygon)
        return (center_x - person_center[0]) ** 2 + (center_y - person_center[1]) ** 2

    return min(screens.values(), key=distance)


def _joint_crop_box(
    person_boxes: Sequence[tuple[int, int, int, int]],
    screen_polygon: Sequence[tuple[float, float]],
    *,
    frame_width: int,
    frame_height: int,
    margin_ratio: float = 0.15,
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
