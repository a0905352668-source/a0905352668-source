#!/usr/bin/env python3
"""Static phone suppression helpers for JianKong surveillance inference.

The detector is still phone-only. This module only decides whether a detected
phone looks like a stationary desk/box/stand object near a tracked person.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

Point = tuple[float, float]
BBox = tuple[float, float, float, float]
Polygon = list[Point]


@dataclass
class StaticPhoneConfig:
    enabled: bool = True
    window_seconds: float = 1.5
    max_disp_ratio: float = 0.03
    risk_multiplier: float = 0.2
    min_abs_disp_px: float = 6.0
    min_bbox_iou: float = 0.45
    max_bbox_size_change_ratio: float = 0.35
    lower_person_start_ratio: float = 0.55
    wrist_follow_min_motion_px: float = 10.0
    wrist_follow_cosine: float = 0.65


@dataclass
class StaticPhoneObservation:
    camera_id: str
    person_id: int
    frame_id: int
    timestamp: float
    phone_center: Point
    phone_bbox: BBox
    person_bbox: BBox
    nearest_wrist: Optional[Point] = None
    risk_score: float = 0.0
    candidate: bool = False


@dataclass
class StaticPhoneResult:
    phone_static: bool = False
    phone_static_duration: float = 0.0
    phone_motion_px: float = 0.0
    phone_in_desk_zone: bool = False
    static_suppressed: bool = False
    static_risk_multiplier: float = 1.0
    phone_follow_wrist: bool = False
    bbox_stable: bool = False
    in_static_area: bool = False
    reason: str = ""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _looks_like_point(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2 and _is_number(value[0]) and _is_number(value[1])


def _looks_like_polygon(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 3 and all(_looks_like_point(p) for p in value)


def _normalise_polygon(poly: Any) -> Optional[Polygon]:
    if not _looks_like_polygon(poly):
        return None
    return [(float(p[0]), float(p[1])) for p in poly]


def iter_polygon_specs(value: Any) -> Iterable[Polygon]:
    if value is None:
        return
    if isinstance(value, dict):
        for key in ("polygon", "points", "poly"):
            if key in value:
                yield from iter_polygon_specs(value[key])
                return
        for key in ("zones", "desk_static_zones"):
            if key in value:
                yield from iter_polygon_specs(value[key])
                return
        return
    poly = _normalise_polygon(value)
    if poly is not None:
        yield poly
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_polygon_specs(item)


def _scale_polygon(poly: Polygon, sx: float, sy: float) -> Polygon:
    return [(float(x) * sx, float(y) * sy) for x, y in poly]


def load_desk_static_zones_from_data(data: dict[str, Any], frame_width: int, frame_height: int, field: str = "desk_static_zones") -> list[Polygon]:
    if "frame_size" in data:
        ref_w, ref_h = data["frame_size"]
    else:
        ref_w = data.get("image_width", frame_width)
        ref_h = data.get("image_height", frame_height)
    sx = frame_width / float(ref_w or frame_width or 1)
    sy = frame_height / float(ref_h or frame_height or 1)

    raw_items: list[Any] = []
    keys = []
    for key in (field, "desk_static_zones", "desk_static_zone"):
        if key not in keys:
            keys.append(key)
    for key in keys:
        if key in data:
            raw_items.append(data[key])
    for screen in data.get("screens", []):
        if not isinstance(screen, dict):
            continue
        for key in keys:
            if key in screen:
                raw_items.append(screen[key])

    zones: list[Polygon] = []
    for raw in raw_items:
        for poly in iter_polygon_specs(raw):
            if len(poly) >= 3:
                zones.append(_scale_polygon(poly, sx, sy))
    return zones


def load_desk_static_zones_from_calibration(path: Union[str, Path], frame_width: int, frame_height: int, field: str = "desk_static_zones") -> list[Polygon]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return load_desk_static_zones_from_data(data, frame_width, frame_height, field)


def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        on_y = (yi > y) != (yj > y)
        if on_y:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x <= x_at_y:
                inside = not inside
        j = i
    return inside


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1.0, area_a + area_b - inter)


def _bbox_size_change(a: BBox, b: BBox) -> float:
    aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
    bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    return max(abs(aw - bw) / max(aw, bw), abs(ah - bh) / max(ah, bh))


def _person_height(person_bbox: BBox) -> float:
    return max(1.0, person_bbox[3] - person_bbox[1])


def _in_lower_person_area(center: Point, person_bbox: BBox, start_ratio: float) -> bool:
    x, y = center
    x1, y1, x2, y2 = person_bbox
    w = max(1.0, x2 - x1)
    margin = w * 0.15
    return (x1 - margin) <= x <= (x2 + margin) and y >= y1 + (y2 - y1) * start_ratio


def _motion_px(points: list[Point]) -> float:
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _follows_wrist(observations: list[StaticPhoneObservation], cfg: StaticPhoneConfig) -> bool:
    wrist_obs = [o for o in observations if o.nearest_wrist is not None]
    if len(wrist_obs) < 2:
        return False
    first, last = wrist_obs[0], wrist_obs[-1]
    wx = last.nearest_wrist[0] - first.nearest_wrist[0]  # type: ignore[index]
    wy = last.nearest_wrist[1] - first.nearest_wrist[1]  # type: ignore[index]
    px = last.phone_center[0] - first.phone_center[0]
    py = last.phone_center[1] - first.phone_center[1]
    wrist_motion = math.hypot(wx, wy)
    phone_motion = math.hypot(px, py)
    person_h = _person_height(last.person_bbox)
    min_wrist_motion = max(cfg.wrist_follow_min_motion_px, 0.04 * person_h)
    if wrist_motion < min_wrist_motion or phone_motion < cfg.min_abs_disp_px:
        return False
    denom = max(1e-6, wrist_motion * phone_motion)
    cosine = (wx * px + wy * py) / denom
    similar_magnitude = phone_motion >= wrist_motion * 0.35
    return cosine >= cfg.wrist_follow_cosine and similar_magnitude


def _continuous_enough(observations: list[StaticPhoneObservation], window_seconds: float) -> bool:
    if len(observations) < 2:
        return False
    duration = observations[-1].timestamp - observations[0].timestamp
    if duration < window_seconds:
        return False
    gaps = [b.timestamp - a.timestamp for a, b in zip(observations, observations[1:])]
    max_allowed_gap = max(0.35, window_seconds * 0.60)
    return not gaps or max(gaps) <= max_allowed_gap


class StaticPhoneSuppressionState:
    def __init__(self, config: Optional[StaticPhoneConfig] = None) -> None:
        self.config = config or StaticPhoneConfig()
        self._history: dict[tuple[str, int], deque[StaticPhoneObservation]] = defaultdict(lambda: deque(maxlen=120))

    def update(self, observation: StaticPhoneObservation, desk_static_zones: Optional[list[Polygon]] = None) -> StaticPhoneResult:
        cfg = self.config
        key = (observation.camera_id, int(observation.person_id))
        hist = self._history[key]
        hist.append(observation)
        cutoff = observation.timestamp - max(cfg.window_seconds * 2.5, cfg.window_seconds + 1.0)
        while hist and hist[0].timestamp < cutoff:
            hist.popleft()

        result = StaticPhoneResult(static_risk_multiplier=1.0)
        zones = desk_static_zones or []
        result.phone_in_desk_zone = any(point_in_polygon(observation.phone_center, poly) for poly in zones)
        lower_person = _in_lower_person_area(observation.phone_center, observation.person_bbox, cfg.lower_person_start_ratio)
        result.in_static_area = result.phone_in_desk_zone or lower_person

        analysis_window = max(cfg.window_seconds, min(2.0, cfg.window_seconds + 0.5))
        recent = [o for o in hist if observation.timestamp - o.timestamp <= analysis_window + 1e-6]
        result.phone_static_duration = max(0.0, recent[-1].timestamp - recent[0].timestamp) if len(recent) >= 2 else 0.0
        result.phone_motion_px = _motion_px([o.phone_center for o in recent])
        max_disp = max(cfg.min_abs_disp_px, cfg.max_disp_ratio * _person_height(observation.person_bbox))

        if recent:
            ious = [bbox_iou(observation.phone_bbox, o.phone_bbox) for o in recent]
            size_changes = [_bbox_size_change(observation.phone_bbox, o.phone_bbox) for o in recent]
            result.bbox_stable = min(ious) >= cfg.min_bbox_iou and max(size_changes) <= cfg.max_bbox_size_change_ratio
        result.phone_follow_wrist = _follows_wrist(recent, cfg)

        continuous = _continuous_enough(recent, cfg.window_seconds)
        motion_stable = result.phone_motion_px < max_disp
        result.phone_static = bool(continuous and motion_stable and result.bbox_stable and result.in_static_area and not result.phone_follow_wrist)
        result.static_suppressed = bool(cfg.enabled and result.phone_static)
        result.static_risk_multiplier = cfg.risk_multiplier if result.static_suppressed else 1.0
        if not continuous:
            result.reason = "not_enough_static_history"
        elif not motion_stable:
            result.reason = "phone_moving"
        elif not result.bbox_stable:
            result.reason = "bbox_unstable"
        elif not result.in_static_area:
            result.reason = "not_static_area"
        elif result.phone_follow_wrist:
            result.reason = "follows_wrist"
        else:
            result.reason = "static_phone"
        return result
