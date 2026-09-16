"""Chronological full-scene evidence for offline capture review."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CAPTURE_EVIDENCE_REVISION = "scene-person-phone-span5s-16f-jpeg92-v1"


@dataclass(frozen=True)
class CaptureSequence:
    track_id: str
    frames: tuple[Any, ...]
    source_frame_indices: tuple[int, ...]
    times: tuple[float, ...]


def build_capture_panel(
    full_frame: Any,
    *,
    person_box: tuple[int, int, int, int],
    phone_box: tuple[int, int, int, int] | None,
    screen_polygons: Sequence[Sequence[tuple[float, float]]],
    occluder_polygons: Sequence[Sequence[tuple[float, float]]],
    image_size: int = 448,
) -> Any:
    from PIL import Image, ImageDraw

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    frame = full_frame.convert("RGB")
    panel = Image.new("RGB", (image_size * 2, image_size), (128, 128, 128))

    scene_scale = min(image_size / frame.width, image_size / frame.height)
    scene_size = (
        max(1, int(round(frame.width * scene_scale))),
        max(1, int(round(frame.height * scene_scale))),
    )
    scene = frame.resize(scene_size, Image.Resampling.LANCZOS)
    scene_origin = (
        (image_size - scene.width) // 2,
        (image_size - scene.height) // 2,
    )
    panel.paste(scene, scene_origin)
    draw = ImageDraw.Draw(panel)
    for polygons, color in (
        (screen_polygons, (255, 0, 0)),
        (occluder_polygons, (0, 255, 0)),
    ):
        for polygon in polygons:
            points = [
                (
                    int(round(x * scene_scale)) + scene_origin[0],
                    int(round(y * scene_scale)) + scene_origin[1],
                )
                for x, y in polygon
            ]
            if len(points) >= 3:
                draw.line(points + [points[0]], fill=color, width=3)

    person = _crop(frame, person_box, "person")
    person_scale = min(1.0, image_size / person.width, image_size / person.height)
    person_size = (
        max(1, int(round(person.width * person_scale))),
        max(1, int(round(person.height * person_scale))),
    )
    if person_size != person.size:
        person = person.resize(person_size, Image.Resampling.LANCZOS)
    person_origin = (
        image_size + (image_size - person.width) // 2,
        (image_size - person.height) // 2,
    )
    panel.paste(person, person_origin)

    if phone_box is not None:
        focus = _phone_focus_box(phone_box, frame.width, frame.height)
        phone = _crop(frame, focus, "phone")
        tile_size = max(64, min(144, image_size // 3))
        phone.thumbnail((tile_size - 8, tile_size - 8), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), (96, 96, 96))
        tile.paste(
            phone,
            ((tile_size - phone.width) // 2, (tile_size - phone.height) // 2),
        )
        ImageDraw.Draw(tile).rectangle(
            (0, 0, tile_size - 1, tile_size - 1), fill=None, outline=(255, 255, 0), width=3
        )
        panel.paste(tile, (image_size * 2 - tile_size - 8, image_size - tile_size - 8))
    return panel


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
        screen_polygons, occluder_polygons = _visibility_polygons(
            visibility, video_width=width, video_height=height
        )
        targets: dict[int, list[tuple[str, int, float, Mapping[str, Any]]]] = {}
        for track_id, entries in selected_by_track:
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
                phone_box = _largest_phone_box(
                    entry,
                    video_width=width,
                    video_height=height,
                )
                panel = build_capture_panel(
                    frame,
                    person_box=person_box,
                    phone_box=phone_box,
                    screen_polygons=screen_polygons,
                    occluder_polygons=occluder_polygons,
                )
                transported = io.BytesIO()
                panel.save(transported, format="JPEG", quality=92)
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


def _phone_focus_box(
    phone_box: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = phone_box
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("invalid phone crop")
    side = max(48, int(round(max(width, height) * 2.75)))
    side = min(side, frame_width, frame_height)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    focus_left = min(max(0, int(round(center_x - side / 2))), frame_width - side)
    focus_top = min(max(0, int(round(center_y - side / 2))), frame_height - side)
    return focus_left, focus_top, focus_left + side, focus_top + side


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
        if len(window) > 16:
            positions = [round(index * (len(window) - 1) / 15) for index in range(16)]
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


def _largest_phone_box(
    entry: Mapping[str, Any], *, video_width: int, video_height: int
) -> tuple[int, int, int, int] | None:
    raw_boxes = entry.get("phone_boxes")
    if not isinstance(raw_boxes, list):
        return None
    boxes = []
    for raw in raw_boxes:
        if not isinstance(raw, Mapping) or raw.get("accepted") is not True:
            continue
        box = _scaled_box(
            raw.get("box"), entry, video_width=video_width, video_height=video_height
        )
        if box is not None:
            boxes.append(box)
    return max(
        boxes,
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        default=None,
    )


def _visibility_polygons(
    visibility: Any, *, video_width: int, video_height: int
) -> tuple[tuple[tuple[tuple[float, float], ...], ...], tuple[tuple[tuple[float, float], ...], ...]]:
    if not isinstance(visibility, Mapping):
        return (), ()
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

    screens = []
    raw_screens = visibility.get("screens")
    if isinstance(raw_screens, list):
        for screen in raw_screens:
            if isinstance(screen, Mapping):
                polygon = scaled(screen.get("screen_polygon", screen.get("screen_poly")))
                if polygon is not None:
                    screens.append(polygon)
    occluders = []
    raw_occluders = visibility.get("occluders")
    if isinstance(raw_occluders, list):
        for occluder in raw_occluders:
            if isinstance(occluder, Mapping):
                polygon = scaled(occluder.get("polygon"))
                if polygon is not None:
                    occluders.append(polygon)
    return tuple(screens), tuple(occluders)
