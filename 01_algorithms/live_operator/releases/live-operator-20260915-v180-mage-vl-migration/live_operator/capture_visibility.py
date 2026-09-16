"""Camera-specific visibility relationships for screen-capture review."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


VisibilityState = Literal["possible", "blocked", "unknown"]


@dataclass(frozen=True)
class VisibilityRelation:
    screen_id: str
    state: VisibilityState
    zone_id: str | None
    occluder_id: str | None


@dataclass(frozen=True)
class CaptureVisibilityConfig:
    camera: str
    frame_width: int
    frame_height: int
    screens: dict[str, "ScreenVisibility"]
    occluders: dict[str, "Occluder"]


Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class VisibilityZone:
    zone_id: str
    polygon: Polygon
    occluder_id: str | None = None


@dataclass(frozen=True)
class ScreenVisibility:
    screen_id: str
    capture_position_zones: tuple[VisibilityZone, ...]
    blocked_position_zones: tuple[VisibilityZone, ...]
    unknown_position_zones: tuple[VisibilityZone, ...]


@dataclass(frozen=True)
class Occluder:
    occluder_id: str
    polygon: Polygon


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"invalid {label}")
    return value


def _positive_dimension(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"invalid {label}")
    return value


def _polygon(value: Any, width: int, height: int, label: str) -> Polygon:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{label} must contain three distinct points")
    points: list[Point] = []
    for raw_point in value:
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise ValueError(f"invalid point in {label}")
        x, y = raw_point
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
        ):
            raise ValueError(f"non-finite point in {label}")
        point = (float(x), float(y))
        if not (0.0 <= point[0] <= width and 0.0 <= point[1] <= height):
            raise ValueError(f"point outside frame in {label}")
        points.append(point)
    if len(set(points)) < 3:
        raise ValueError(f"{label} must contain three distinct points")
    return tuple(points)


def _zones(
    value: Any,
    *,
    width: int,
    height: int,
    label: str,
    blocked: bool,
    seen_ids: set[str],
) -> tuple[VisibilityZone, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    output = []
    for index, raw_zone in enumerate(value):
        zone = _object(raw_zone, f"{label}[{index}]")
        zone_id = _identifier(zone.get("zone_id"), "zone_id")
        if zone_id in seen_ids:
            raise ValueError(f"duplicate zone_id: {zone_id}")
        seen_ids.add(zone_id)
        occluder_id = (
            _identifier(zone.get("occluder_id"), "occluder_id") if blocked else None
        )
        output.append(
            VisibilityZone(
                zone_id=zone_id,
                polygon=_polygon(zone.get("polygon"), width, height, label),
                occluder_id=occluder_id,
            )
        )
    return tuple(output)


def load_visibility_sidecar(path: Path) -> CaptureVisibilityConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid visibility sidecar JSON") from error
    root = _object(payload, "visibility sidecar")
    if root.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    camera = _identifier(root.get("camera"), "camera")
    width = _positive_dimension(root.get("frame_width"), "frame_width")
    height = _positive_dimension(root.get("frame_height"), "frame_height")

    raw_occluders = root.get("occluders")
    if not isinstance(raw_occluders, list):
        raise ValueError("occluders must be an array")
    occluders: dict[str, Occluder] = {}
    for index, raw_occluder in enumerate(raw_occluders):
        item = _object(raw_occluder, f"occluders[{index}]")
        occluder_id = _identifier(item.get("occluder_id"), "occluder_id")
        if occluder_id in occluders:
            raise ValueError(f"duplicate occluder_id: {occluder_id}")
        occluders[occluder_id] = Occluder(
            occluder_id=occluder_id,
            polygon=_polygon(item.get("polygon"), width, height, "occluder polygon"),
        )

    raw_screens = root.get("screens")
    if not isinstance(raw_screens, list) or not raw_screens:
        raise ValueError("screens must be a non-empty array")
    screens: dict[str, ScreenVisibility] = {}
    for index, raw_screen in enumerate(raw_screens):
        item = _object(raw_screen, f"screens[{index}]")
        screen_id = _identifier(item.get("screen_id"), "screen_id")
        if screen_id in screens:
            raise ValueError(f"duplicate screen_id: {screen_id}")
        zone_ids: set[str] = set()
        capture = _zones(
            item.get("capture_position_zones"),
            width=width,
            height=height,
            label="capture_position_zones",
            blocked=False,
            seen_ids=zone_ids,
        )
        blocked = _zones(
            item.get("blocked_position_zones"),
            width=width,
            height=height,
            label="blocked_position_zones",
            blocked=True,
            seen_ids=zone_ids,
        )
        unknown = _zones(
            item.get("unknown_position_zones"),
            width=width,
            height=height,
            label="unknown_position_zones",
            blocked=False,
            seen_ids=zone_ids,
        )
        for zone in blocked:
            if zone.occluder_id not in occluders:
                raise ValueError(f"unknown occluder_id: {zone.occluder_id}")
        screens[screen_id] = ScreenVisibility(
            screen_id=screen_id,
            capture_position_zones=capture,
            blocked_position_zones=blocked,
            unknown_position_zones=unknown,
        )
    return CaptureVisibilityConfig(camera, width, height, screens, occluders)


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-9:
        return False
    return min(ax, bx) - 1e-9 <= px <= max(ax, bx) + 1e-9 and min(
        ay, by
    ) - 1e-9 <= py <= max(ay, by) + 1e-9


def _point_in_polygon(point: Point, polygon: Polygon) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        x1, y1 = previous
        x2, y2 = current
        if (y1 > point[1]) != (y2 > point[1]):
            crossing_x = (x2 - x1) * (point[1] - y1) / (y2 - y1) + x1
            if point[0] < crossing_x:
                inside = not inside
        previous = current
    return inside


def evaluate_visibility(
    config: CaptureVisibilityConfig,
    *,
    screen_id: str,
    anchor: tuple[float, float],
) -> VisibilityRelation:
    screen = config.screens.get(screen_id)
    if screen is None:
        raise ValueError(f"unknown screen_id: {screen_id}")
    for zone in screen.blocked_position_zones:
        if _point_in_polygon(anchor, zone.polygon):
            return VisibilityRelation(screen_id, "blocked", zone.zone_id, zone.occluder_id)
    for zone in screen.capture_position_zones:
        if _point_in_polygon(anchor, zone.polygon):
            return VisibilityRelation(screen_id, "possible", zone.zone_id, None)
    for zone in screen.unknown_position_zones:
        if _point_in_polygon(anchor, zone.polygon):
            return VisibilityRelation(screen_id, "unknown", zone.zone_id, None)
    return VisibilityRelation(screen_id, "unknown", None, None)
