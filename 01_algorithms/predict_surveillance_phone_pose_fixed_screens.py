#!/usr/bin/env python3
"""Surveillance anti-filming inference v2.1.

This version follows the v2.1 design:
screen static zones + person dynamic zones + local phone detection +
zone/hand/angle joint scoring. It intentionally disables the old
near-screen override and occlusion-only alert path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


FONT_SCALE_MULT = 1.5
MIN_SEARCH_AREA = 160 * 160


@dataclass
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def diag(self) -> float:
        return math.hypot(self.w, self.h)

    def as_int(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def clamp(self, width: int, height: int) -> "Rect":
        return Rect(
            max(0.0, min(width - 1.0, self.x1)),
            max(0.0, min(height - 1.0, self.y1)),
            max(0.0, min(width - 1.0, self.x2)),
            max(0.0, min(height - 1.0, self.y2)),
        )


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]]
    weight: float = 1.0


@dataclass
class ScreenConfig:
    screen_id: str
    screen_poly: list[tuple[float, float]]
    near_zone: list[tuple[float, float]]
    danger_zones: list[Zone]
    ignore_zones: list[Zone]
    params: dict[str, Any]


@dataclass
class PersonPose:
    index: int
    nose: tuple[float, float] | None
    l_shoulder: tuple[float, float] | None
    r_shoulder: tuple[float, float] | None
    l_elbow: tuple[float, float] | None
    r_elbow: tuple[float, float] | None
    l_wrist: tuple[float, float] | None
    r_wrist: tuple[float, float] | None
    box: Rect
    scale: float
    conf: float = 0.0
    track_id: int = -1

    def visible_points(self) -> list[tuple[float, float]]:
        pts = [self.nose, self.l_shoulder, self.r_shoulder, self.l_elbow, self.r_elbow, self.l_wrist, self.r_wrist]
        return [p for p in pts if p is not None]


@dataclass
class PhoneCandidate:
    box: Rect
    conf: float
    source: str
    screen_id: str
    roi: Rect
    phone_id: int = -1
    person_id: int = -1

    @property
    def center(self) -> tuple[float, float]:
        return self.box.cx, self.box.cy


@dataclass
class RoiJob:
    roi: Rect
    screen_id: str
    source: str
    person_id: int = -1


@dataclass
class CandidateEval:
    phone: PhoneCandidate
    screen: ScreenConfig
    person: PersonPose | None
    static_zone_score: float
    zone_reason: str
    person_match_score: float
    hand_link_score: float
    aim_score: float
    source_score: float
    risk_score: float
    level: str
    reject_reason: str
    candidate_reason: str
    best_angle: float
    best_ray_hit: bool
    person_expand_roi: Rect | None = None
    hand_rois: list[Rect] = field(default_factory=list)
    corridor_bbox: Rect | None = None
    effective_search_roi: Rect | None = None
    track_id: int = -1
    phone_score: float = 0.0
    phone_hand_score: float = 0.0
    screen_relation_score: float = 0.0
    pose_score: float = 0.0
    temporal_score: float = 0.0
    nearest_wrist: str = ""
    phone_to_left_wrist_dist: float = -1.0
    phone_to_right_wrist_dist: float = -1.0
    phone_to_screen_dist: float = -1.0
    phone_in_person_roi: bool = False
    person_screen_relation: str = ""
    person_state: str = ""
    person_stable_count: int = 0
    person_window_hits: int = 0
    person_window_size: int = 0
    person_alarm: bool = False

    @property
    def accepted(self) -> bool:
        return not self.reject_reason


@dataclass
class PersonTrackState:
    track_id: int
    bbox_history: Any = field(default_factory=lambda: deque(maxlen=30))
    phone_history: Any = field(default_factory=lambda: deque(maxlen=30))
    risk_history: Any = field(default_factory=lambda: deque(maxlen=30))
    candidate_history: Any = field(default_factory=lambda: deque(maxlen=30))
    state: str = "S0_CLEAR"
    stable_count: int = 0
    window_hits: int = 0
    alarm_triggered: bool = False
    last_seen: int = 0
    last_bbox: Rect | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-model", required=True, type=Path)
    parser.add_argument("--phone-model", required=True, type=Path)
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--debug-csv", type=Path)
    parser.add_argument("--pose-imgsz", type=int, default=1280)
    parser.add_argument("--phone-imgsz", type=int, default=1280)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--kp-conf", type=float, default=0.35)
    parser.add_argument("--phone-candidate-conf", type=float, default=0.25)
    parser.add_argument("--roi-batch-size", type=int, default=4)
    parser.add_argument("--phone-roi-mode", choices=["person", "person_screen"], default="person")
    parser.add_argument("--draw-static-zones", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--draw-debug", action="store_true", default=True)
    parser.add_argument("--draw-mode", choices=["full", "compact"], default="full")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def default_params() -> dict[str, Any]:
    return {
        "enable_dynamic_person_zone": True,
        "person_expand_x": 0.20,
        "person_expand_y": 0.12,
        "near_zone_detect_interval": 10,
        "phone_valid_conf_person_roi": 0.35,
        "phone_valid_conf_near_zone": 0.50,
        "phone_strong_conf": 0.55,
        "ignore_center_inside": True,
        "angle_thresh": 90.0,
        "angle_relaxed_thresh": 110.0,
        "hand_radius_ratio": 0.18,
        "hand_radius_min": 40.0,
        "corridor_width_ratio": 0.35,
        "corridor_width_min": 80.0,
        "alert_counter_threshold": 10.0,
        "person_state_window": 30,
        "person_state_min_hits": 16,
        "person_state_risk_threshold": 0.65,
        "person_state_alarm_risk": 0.90,
        "occluded_enable": False,
        "near_screen_override": False,
    }


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_deg(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_conf(conf: float, low: float = 0.35, high: float = 0.70) -> float:
    return clamp((conf - low) / max(1e-6, high - low), 0.0, 1.0)


def scale_poly(poly: list[list[float]] | list[tuple[float, float]], sx: float, sy: float) -> list[tuple[float, float]]:
    return [(float(x) * sx, float(y) * sy) for x, y in poly]


def bbox_from_polygon(poly: list[tuple[float, float]]) -> Rect:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return Rect(min(xs), min(ys), max(xs), max(ys))


def rect_to_poly(rect: Rect) -> list[tuple[float, float]]:
    return [(rect.x1, rect.y1), (rect.x2, rect.y1), (rect.x2, rect.y2), (rect.x1, rect.y2)]


def expand_rect(rect: Rect, x_ratio: float, y_ratio: float, width: int, height: int) -> Rect:
    dx = rect.w * x_ratio
    dy = rect.h * y_ratio
    return Rect(rect.x1 - dx, rect.y1 - dy, rect.x2 + dx, rect.y2 + dy).clamp(width, height)


def square_around(pt: tuple[float, float], radius: float, width: int, height: int) -> Rect:
    return Rect(pt[0] - radius, pt[1] - radius, pt[0] + radius, pt[1] + radius).clamp(width, height)


def union_boxes(boxes: list[Rect]) -> Rect | None:
    boxes = [b for b in boxes if b.area > 1]
    if not boxes:
        return None
    return Rect(min(b.x1 for b in boxes), min(b.y1 for b in boxes), max(b.x2 for b in boxes), max(b.y2 for b in boxes))


def intersect_boxes(a: Rect, b: Rect) -> Rect | None:
    out = Rect(max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2))
    if out.x2 <= out.x1 or out.y2 <= out.y1:
        return None
    return out


def boxes_intersect(a: Rect, b: Rect) -> bool:
    return not (a.x2 <= b.x1 or b.x2 <= a.x1 or a.y2 <= b.y1 or b.y2 <= a.y1)


def point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    return cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32), pt, False) >= 0


def point_in_box(pt: tuple[float, float], rect: Rect) -> bool:
    return rect.x1 <= pt[0] <= rect.x2 and rect.y1 <= pt[1] <= rect.y2


def box_iou(a: Rect, b: Rect) -> float:
    inter = intersect_boxes(a, b)
    if inter is None:
        return 0.0
    return inter.area / max(1.0, a.area + b.area - inter.area)


def point_to_rect_distance(pt: tuple[float, float], rect: Rect) -> float:
    x, y = pt
    return math.hypot(max(rect.x1 - x, 0.0, x - rect.x2), max(rect.y1 - y, 0.0, y - rect.y2))


def box_to_poly_distance(box: Rect, poly: list[tuple[float, float]]) -> float:
    poly_box = bbox_from_polygon(poly)
    if boxes_intersect(box, poly_box):
        return 0.0
    points = [(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2), (box.cx, box.cy)]
    return min(abs(cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32), p, True)) for p in points)




def resize_history(values: Any, window_size: int) -> Any:
    if getattr(values, "maxlen", None) == window_size:
        return values
    return deque(values, maxlen=window_size)


def new_track_state(track_id: int, window_size: int) -> PersonTrackState:
    state = PersonTrackState(track_id)
    ensure_track_window(state, window_size)
    return state


def ensure_track_window(state: PersonTrackState, window_size: int) -> None:
    state.bbox_history = resize_history(state.bbox_history, window_size)
    state.phone_history = resize_history(state.phone_history, window_size)
    state.risk_history = resize_history(state.risk_history, window_size)
    state.candidate_history = resize_history(state.candidate_history, window_size)


def assign_person_track_ids(people: list[PersonPose], states: dict[int, PersonTrackState], next_track_id: int, frame_id: int, window_size: int) -> int:
    assigned: set[int] = set()
    stale_after = max(window_size * 2, 60)
    for person in people:
        best_id = -1
        best_score = 0.0
        for track_id, state in states.items():
            if track_id in assigned or state.last_bbox is None or frame_id - state.last_seen > stale_after:
                continue
            iou = box_iou(person.box, state.last_bbox)
            center_dist = dist((person.box.cx, person.box.cy), (state.last_bbox.cx, state.last_bbox.cy))
            center_score = clamp(1.0 - center_dist / max(person.scale, state.last_bbox.diag, 1.0), 0.0, 1.0)
            score = max(iou, center_score * 0.5)
            if score > best_score:
                best_score = score
                best_id = track_id
        if best_id < 0 or best_score < 0.20:
            best_id = next_track_id
            next_track_id += 1
            states[best_id] = new_track_state(best_id, window_size)
        state = states[best_id]
        ensure_track_window(state, window_size)
        person.track_id = best_id
        state.last_seen = frame_id
        state.last_bbox = person.box
        assigned.add(best_id)
    return next_track_id


def prune_person_tracks(states: dict[int, PersonTrackState], frame_id: int, window_size: int) -> None:
    stale_after = max(window_size * 3, 90)
    for track_id in list(states):
        if frame_id - states[track_id].last_seen > stale_after:
            del states[track_id]


def candidate_level_from_risk(risk_score: float) -> str:
    if risk_score < 0.40:
        return "ignore"
    if risk_score < 0.65:
        return "weak"
    if risk_score < 0.80:
        return "normal"
    return "strong"


def phone_reliability_score(phone: PhoneCandidate, required_conf: float) -> float:
    conf_score = normalize_conf(phone.conf, low=required_conf, high=max(required_conf + 0.25, 0.75))
    aspect = phone.box.w / max(1.0, phone.box.h)
    aspect_score = 1.0 if 0.25 <= aspect <= 4.0 else 0.5
    size_score = 1.0 if phone.box.area >= 80.0 else clamp(phone.box.area / 80.0, 0.0, 0.7)
    return clamp(0.70 * conf_score + 0.20 * size_score + 0.10 * aspect_score, 0.0, 1.0)


def wrist_debug(phone: PhoneCandidate, person: PersonPose) -> tuple[str, float, float]:
    left = dist(phone.center, person.l_wrist) if person.l_wrist is not None else -1.0
    right = dist(phone.center, person.r_wrist) if person.r_wrist is not None else -1.0
    choices = []
    if left >= 0:
        choices.append(("left", left))
    if right >= 0:
        choices.append(("right", right))
    nearest = min(choices, key=lambda x: x[1])[0] if choices else ""
    return nearest, left, right


def pose_support_score(phone: PhoneCandidate, person: PersonPose, aim_score: float, hand_score: float) -> float:
    rel_y = (phone.box.cy - person.box.y1) / max(1.0, person.box.h)
    upper_body_score = clamp((0.90 - rel_y) / 0.65, 0.0, 1.0)
    wrists = [w for w in (person.l_wrist, person.r_wrist) if w is not None]
    if wrists:
        raised_score = 1.0 if min(w[1] for w in wrists) <= person.box.y1 + person.box.h * 0.78 else 0.35
    else:
        raised_score = 0.0
    return clamp(0.45 * upper_body_score + 0.35 * raised_score + 0.15 * hand_score + 0.05 * aim_score, 0.0, 1.0)


def compute_risk_score(phone_score: float, phone_hand_score: float, screen_relation_score: float, pose_score: float, temporal_score: float) -> float:
    return clamp(
        0.30 * phone_score
        + 0.20 * phone_hand_score
        + 0.20 * screen_relation_score
        + 0.15 * pose_score
        + 0.15 * temporal_score,
        0.0,
        1.0,
    )


def compute_temporal_score(state: PersonTrackState | None, risk_threshold: float) -> float:
    if state is None or not state.risk_history:
        return 0.0
    risks = list(state.risk_history)
    candidates = list(state.candidate_history)
    hit_ratio = sum(1 for r in risks if r >= risk_threshold) / max(1, len(risks))
    avg_risk = sum(risks) / max(1, len(risks))
    candidate_ratio = sum(candidates) / max(1, len(candidates))
    return clamp(0.50 * hit_ratio + 0.35 * avg_risk + 0.15 * candidate_ratio, 0.0, 1.0)


def update_eval_risk(ev: CandidateEval) -> None:
    ev.risk_score = compute_risk_score(ev.phone_score, ev.phone_hand_score, ev.screen_relation_score, ev.pose_score, ev.temporal_score)
    ev.level = candidate_level_from_risk(ev.risk_score)
    if ev.person is not None and ev.reject_reason in ("", "low_joint_score", "low_risk_score"):
        if ev.risk_score >= 0.40 and ev.static_zone_score > 0.0 and ev.person_match_score >= 0.35:
            ev.reject_reason = ""
        else:
            ev.reject_reason = "low_risk_score"
    ev.candidate_reason = (
        f"{ev.level}|{ev.zone_reason}|phone={ev.phone_score:.2f}|hand={ev.phone_hand_score:.2f}|"
        f"screen={ev.screen_relation_score:.2f}|pose={ev.pose_score:.2f}|temporal={ev.temporal_score:.2f}|risk={ev.risk_score:.2f}"
    )


def apply_temporal_scores(evals: list[CandidateEval], states: dict[int, PersonTrackState], risk_threshold: float) -> None:
    for ev in evals:
        if ev.person is None:
            continue
        ev.track_id = ev.person.track_id
        ev.temporal_score = compute_temporal_score(states.get(ev.track_id), risk_threshold)
        update_eval_risk(ev)


def trailing_candidate_count(values: Any) -> int:
    count = 0
    for item in reversed(list(values)):
        if not item:
            break
        count += 1
    return count


def update_person_states(
    people: list[PersonPose],
    evals: list[CandidateEval],
    states: dict[int, PersonTrackState],
    frame_id: int,
    window_size: int,
    risk_threshold: float,
    min_hits: int,
) -> None:
    best_by_track: dict[int, CandidateEval] = {}
    for ev in evals:
        if ev.person is None or ev.track_id < 0:
            continue
        prev = best_by_track.get(ev.track_id)
        if prev is None or ev.risk_score > prev.risk_score:
            best_by_track[ev.track_id] = ev
    for person in people:
        if person.track_id < 0:
            continue
        state = states.setdefault(person.track_id, new_track_state(person.track_id, window_size))
        ensure_track_window(state, window_size)
        ev = best_by_track.get(person.track_id)
        risk = ev.risk_score if ev is not None else 0.0
        phone_seen = ev is not None and ev.phone_score > 0.0
        candidate = ev is not None and ev.accepted and ev.risk_score >= risk_threshold
        state.bbox_history.append(person.box)
        state.phone_history.append(1 if phone_seen else 0)
        state.risk_history.append(risk)
        state.candidate_history.append(1 if candidate else 0)
        state.window_hits = int(sum(state.candidate_history))
        state.stable_count = trailing_candidate_count(state.candidate_history)
        state.alarm_triggered = state.window_hits >= min_hits
        if state.alarm_triggered:
            state.state = "S4_ALARM"
        elif state.window_hits >= max(3, min_hits // 2):
            state.state = "S3_SUSTAINED_RISK"
        elif candidate:
            state.state = "S2_RISK"
        elif phone_seen:
            state.state = "S1_PHONE"
        else:
            state.state = "S0_CLEAR"
        state.last_seen = frame_id
        state.last_bbox = person.box


def sync_eval_state_fields(evals: list[CandidateEval], states: dict[int, PersonTrackState]) -> None:
    for ev in evals:
        if ev.track_id < 0:
            continue
        state = states.get(ev.track_id)
        if state is None:
            continue
        ev.person_state = state.state
        ev.person_stable_count = state.stable_count
        ev.person_window_hits = state.window_hits
        ev.person_window_size = len(state.candidate_history)
        ev.person_alarm = state.alarm_triggered


def ray_hits_box(start: tuple[float, float], direction: tuple[float, float], rect: Rect, max_t: float = 4.0) -> bool:
    sx, sy = start
    dx, dy = direction
    if math.hypot(dx, dy) < 1e-6:
        return False
    tmin, tmax = 0.0, max_t
    for s, d, lo, hi in ((sx, dx, rect.x1, rect.x2), (sy, dy, rect.y1, rect.y2)):
        if abs(d) < 1e-6:
            if s < lo or s > hi:
                return False
            continue
        t1 = (lo - s) / d
        t2 = (hi - s) / d
        t1, t2 = min(t1, t2), max(t1, t2)
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmax < tmin:
            return False
    return tmax >= max(0.0, tmin)


def corridor_bbox(start: tuple[float, float], end: tuple[float, float], width: float, frame_w: int, frame_h: int) -> Rect:
    return Rect(
        min(start[0], end[0]) - width * 0.5,
        min(start[1], end[1]) - width * 0.5,
        max(start[0], end[0]) + width * 0.5,
        max(start[1], end[1]) + width * 0.5,
    ).clamp(frame_w, frame_h)


def old_screen_to_v21(item: dict[str, Any], idx: int, width: int, height: int) -> ScreenConfig:
    params = default_params()
    if "bbox_xyxy" in item:
        x1, y1, x2, y2 = [float(v) for v in item["bbox_xyxy"]]
        screen_poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        screen_poly = [(float(x), float(y)) for x, y in item["points"]]
    screen_box = bbox_from_polygon(screen_poly)
    near = expand_rect(screen_box, 1.2, 0.9, width, height)
    front = expand_rect(screen_box, 0.65, 0.55, width, height)
    left = Rect(screen_box.x1 - screen_box.w * 0.9, screen_box.y1 - screen_box.h * 0.35, screen_box.x1 + screen_box.w * 0.35, screen_box.y2 + screen_box.h * 0.45).clamp(width, height)
    right = Rect(screen_box.x2 - screen_box.w * 0.35, screen_box.y1 - screen_box.h * 0.35, screen_box.x2 + screen_box.w * 0.9, screen_box.y2 + screen_box.h * 0.45).clamp(width, height)
    return ScreenConfig(
        screen_id=str(item.get("screen_id") or item.get("id") or f"screen_{idx:02d}"),
        screen_poly=screen_poly,
        near_zone=rect_to_poly(near),
        danger_zones=[
            Zone("front", rect_to_poly(front), 1.0),
            Zone("left_side", rect_to_poly(left), 0.8),
            Zone("right_side", rect_to_poly(right), 0.8),
        ],
        ignore_zones=[],
        params=params,
    )


def load_calibration(path: Path, width: int, height: int) -> tuple[str, list[ScreenConfig]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "frame_size" in data:
        ref_w, ref_h = data["frame_size"]
    else:
        ref_w, ref_h = data.get("image_width", width), data.get("image_height", height)
    sx, sy = width / float(ref_w), height / float(ref_h)
    screens: list[ScreenConfig] = []
    for idx, item in enumerate(data.get("screens", []), start=1):
        if "screen_poly" not in item:
            sc = old_screen_to_v21(item, idx, int(ref_w), int(ref_h))
        else:
            params = default_params()
            params.update(item.get("params", {}))
            sc = ScreenConfig(
                screen_id=str(item.get("screen_id") or item.get("id") or f"screen_{idx:02d}"),
                screen_poly=[(float(x), float(y)) for x, y in item["screen_poly"]],
                near_zone=[(float(x), float(y)) for x, y in item["near_zone"]],
                danger_zones=[Zone(str(z.get("name", "danger")), [(float(x), float(y)) for x, y in z["polygon"]], float(z.get("weight", 1.0))) for z in item.get("danger_zones", [])],
                ignore_zones=[Zone(str(z.get("name", "ignore")), [(float(x), float(y)) for x, y in z["polygon"]], 0.0) for z in item.get("ignore_zones", [])],
                params=params,
            )
        sc.screen_poly = scale_poly(sc.screen_poly, sx, sy)
        sc.near_zone = scale_poly(sc.near_zone, sx, sy)
        sc.danger_zones = [Zone(z.name, scale_poly(z.polygon, sx, sy), z.weight) for z in sc.danger_zones]
        sc.ignore_zones = [Zone(z.name, scale_poly(z.polygon, sx, sy), z.weight) for z in sc.ignore_zones]
        screens.append(sc)
    return str(data.get("camera_id", path.stem)), screens


def write_upgraded_calibration(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    width = int(data.get("image_width") or data.get("frame_size", [0, 0])[0])
    height = int(data.get("image_height") or data.get("frame_size", [0, 0])[1])
    camera_id = str(data.get("camera_id", src.stem))
    screens = []
    for idx, item in enumerate(data.get("screens", []), start=1):
        sc = old_screen_to_v21(item, idx, width, height) if "screen_poly" not in item else item
        if isinstance(sc, ScreenConfig):
            screens.append(
                {
                    "screen_id": sc.screen_id,
                    "screen_poly": [[round(x, 2), round(y, 2)] for x, y in sc.screen_poly],
                    "near_zone": [[round(x, 2), round(y, 2)] for x, y in sc.near_zone],
                    "danger_zones": [
                        {"name": z.name, "polygon": [[round(x, 2), round(y, 2)] for x, y in z.polygon], "weight": z.weight}
                        for z in sc.danger_zones
                    ],
                    "ignore_zones": [],
                    "params": sc.params,
                }
            )
        else:
            screens.append(sc)
    out = {
        "version": "2.1",
        "camera_id": camera_id,
        "frame_size": [width, height],
        "description": f"Auto-upgraded v2.1 zones from {src.name}. Please manually refine near/danger/ignore zones.",
        "screens": screens,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def is_decode_gray(frame) -> bool:
    b, g, r = cv2.split(frame)
    lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    neutral = (np.abs(b.astype(np.int16) - g.astype(np.int16)) < 7) & (np.abs(g.astype(np.int16) - r.astype(np.int16)) < 7)
    mid = (lum > 90) & (lum < 175)
    frac = float((neutral & mid).mean())
    mean = float(lum.mean())
    std = float(lum.std())
    return (frac > 0.68 and 100 < mean < 165) or (std < 12 and 95 < mean < 170)


def get_point(kp, idx: int, conf: float) -> tuple[float, float] | None:
    if idx >= kp.shape[0] or float(kp[idx][2]) < conf:
        return None
    return float(kp[idx][0]), float(kp[idx][1])


def extract_people(pose_result, kp_conf: float, width: int, height: int) -> list[PersonPose]:
    if pose_result.keypoints is None or pose_result.keypoints.data is None:
        return []
    people = []
    boxes_xyxy = []
    boxes_conf = []
    if pose_result.boxes is not None and pose_result.boxes.xyxy is not None:
        boxes_xyxy = pose_result.boxes.xyxy.cpu().numpy().tolist()
        if pose_result.boxes.conf is not None:
            boxes_conf = pose_result.boxes.conf.cpu().numpy().tolist()
    for i, kp in enumerate(pose_result.keypoints.data.cpu().numpy()):
        pts_all = [
            get_point(kp, 0, kp_conf),
            get_point(kp, 5, kp_conf),
            get_point(kp, 6, kp_conf),
            get_point(kp, 7, kp_conf),
            get_point(kp, 8, kp_conf),
            get_point(kp, 9, kp_conf),
            get_point(kp, 10, kp_conf),
        ]
        pts = [p for p in pts_all if p is not None]
        if len(pts) < 2:
            continue
        if i < len(boxes_xyxy):
            box = Rect(*[float(v) for v in boxes_xyxy[i]]).clamp(width, height)
        else:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            box = Rect(min(xs), min(ys), max(xs), max(ys)).clamp(width, height)
        shoulder_width = 0.0
        if pts_all[1] is not None and pts_all[2] is not None:
            shoulder_width = dist(pts_all[1], pts_all[2])
        scale = max(box.w, box.h, shoulder_width * 3.0, 80.0)
        person_conf = float(boxes_conf[i]) if i < len(boxes_conf) else 0.0
        people.append(PersonPose(i, pts_all[0], pts_all[1], pts_all[2], pts_all[3], pts_all[4], pts_all[5], pts_all[6], box, scale, person_conf))
    return people


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.55) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale *= FONT_SCALE_MULT
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + baseline + 6, y)
    x = max(0, min(frame.shape[1] - tw - 12, x))
    cv2.rectangle(frame, (x, y - th - baseline - 8), (x + tw + 10, y + 4), color, -1)
    cv2.putText(frame, text, (x + 5, y - baseline - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_poly(frame, poly: list[tuple[float, float]], color: tuple[int, int, int], thickness: int, label: str | None = None) -> None:
    pts = np.asarray(poly, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    if label and len(pts):
        draw_label(frame, label, int(pts[0][0]), int(pts[0][1]), color, scale=0.34)


def draw_rect(frame, rect: Rect, color: tuple[int, int, int], thickness: int, label: str | None = None) -> None:
    x1, y1, x2, y2 = rect.as_int()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        draw_label(frame, label, x1, y1, color, scale=0.36)


def person_related_to_screen(person: PersonPose, screen: ScreenConfig, width: int, height: int) -> bool:
    p = screen.params
    person_roi = expand_rect(person.box, p["person_expand_x"], p["person_expand_y"], width, height)
    near_bbox = bbox_from_polygon(screen.near_zone)
    if boxes_intersect(person_roi, near_bbox):
        return True
    return box_to_poly_distance(person.box, screen.screen_poly) <= person.scale * 0.50


def person_anchor(person: PersonPose) -> tuple[float, float]:
    if person.l_shoulder is not None and person.r_shoulder is not None:
        return ((person.l_shoulder[0] + person.r_shoulder[0]) * 0.5, (person.l_shoulder[1] + person.r_shoulder[1]) * 0.5)
    return person.box.cx, person.box.cy


def build_dynamic_zones(person: PersonPose, screen: ScreenConfig, width: int, height: int) -> tuple[Rect, list[Rect], Rect]:
    p = screen.params
    person_roi = expand_rect(person.box, p["person_expand_x"], p["person_expand_y"], width, height)
    radius = max(person.scale * p["hand_radius_ratio"], p["hand_radius_min"])
    hand_rois = [square_around(w, radius, width, height) for w in (person.l_wrist, person.r_wrist) if w is not None]
    screen_center = bbox_from_polygon(screen.screen_poly).cx, bbox_from_polygon(screen.screen_poly).cy
    cw = max(person.scale * p["corridor_width_ratio"], p["corridor_width_min"])
    corridor = corridor_bbox(person_anchor(person), screen_center, cw, width, height)
    return person_roi, hand_rois, corridor


def build_effective_search_roi(person: PersonPose, screen: ScreenConfig, width: int, height: int) -> tuple[Rect | None, Rect, list[Rect], Rect]:
    p = screen.params
    person_roi, hand_rois, corridor = build_dynamic_zones(person, screen, width, height)
    dynamic = union_boxes([person_roi, corridor] + hand_rois)
    near_bbox = bbox_from_polygon(screen.near_zone)
    roi = intersect_boxes(near_bbox, dynamic) if dynamic else None
    if roi is None or roi.area < MIN_SEARCH_AREA:
        fallback = expand_rect(person.box, 0.50, 0.30, width, height)
        roi = intersect_boxes(near_bbox, fallback)
    if roi is None or roi.area < 16 * 16:
        return None, person_roi, hand_rois, corridor
    return roi.clamp(width, height), person_roi, hand_rois, corridor


def detect_phone_in_roi(model: YOLO, names: dict[int, str], frame, roi: Rect, conf: float, imgsz: int, device: str, screen_id: str, source: str) -> list[PhoneCandidate]:
    x1, y1, x2, y2 = roi.as_int()
    if x2 <= x1 or y2 <= y1:
        return []
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    result = model.predict(crop, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    phones = []
    if result.boxes is None:
        return phones
    for raw in result.boxes:
        cls_id = int(raw.cls.item())
        name = names.get(cls_id, str(cls_id)).lower()
        pconf = float(raw.conf.item())
        if name != "phone" or pconf < conf:
            continue
        bx1, by1, bx2, by2 = [float(v) for v in raw.xyxy[0].tolist()]
        box = Rect(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
        phones.append(PhoneCandidate(box=box, conf=pconf, source=source, screen_id=screen_id, roi=roi))
    return phones


def detect_phone_in_rois(model: YOLO, names: dict[int, str], frame, jobs: list[RoiJob], conf: float, imgsz: int, device: str, batch_size: int) -> list[PhoneCandidate]:
    valid_jobs: list[RoiJob] = []
    crops = []
    for job in jobs:
        x1, y1, x2, y2 = job.roi.as_int()
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        valid_jobs.append(job)
        crops.append(crop)
    if not crops:
        return []
    phones: list[PhoneCandidate] = []
    batch_size = max(1, batch_size)
    for start in range(0, len(crops), batch_size):
        batch_crops = crops[start : start + batch_size]
        batch_jobs = valid_jobs[start : start + batch_size]
        results = model.predict(batch_crops, imgsz=imgsz, conf=conf, device=device, verbose=False)
        for job, result in zip(batch_jobs, results):
            ox, oy, _, _ = job.roi.as_int()
            if result.boxes is None:
                continue
            for raw in result.boxes:
                cls_id = int(raw.cls.item())
                name = names.get(cls_id, str(cls_id)).lower()
                pconf = float(raw.conf.item())
                if name != "phone" or pconf < conf:
                    continue
                bx1, by1, bx2, by2 = [float(v) for v in raw.xyxy[0].tolist()]
                phones.append(
                    PhoneCandidate(
                        box=Rect(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
                        conf=pconf,
                        source=job.source,
                        screen_id=job.screen_id,
                        roi=job.roi,
                        person_id=job.person_id,
                    )
                )
    return phones


def nms_phone_candidates(phones: list[PhoneCandidate], iou_thresh: float = 0.5) -> list[PhoneCandidate]:
    ordered = sorted(phones, key=lambda p: p.conf, reverse=True)
    keep: list[PhoneCandidate] = []
    for p in ordered:
        if all(box_iou(p.box, k.box) <= iou_thresh for k in keep):
            keep.append(p)
    for i, p in enumerate(keep):
        p.phone_id = i
    return keep


def get_static_zone_score(phone: PhoneCandidate, screen: ScreenConfig) -> tuple[float, str]:
    c = phone.center
    for z in screen.ignore_zones:
        if point_in_polygon(c, z.polygon):
            return 0.0, f"ignore:{z.name}"
    for dz in screen.danger_zones:
        if point_in_polygon(c, dz.polygon):
            return float(dz.weight), f"danger:{dz.name}"
    if point_in_polygon(c, screen.near_zone):
        return 0.5, "near_zone"
    if phone.source == "person_roi":
        screen_box = bbox_from_polygon(screen.screen_poly)
        screen_dist = point_to_rect_distance(c, screen_box)
        if screen_dist <= max(screen_box.diag * 0.80, phone.box.diag * 3.0):
            return 0.5, "screen_distance"
    return 0.0, "outside"


def get_hand_link_score(phone: PhoneCandidate, person: PersonPose, screen: ScreenConfig) -> float:
    wrists = [w for w in (person.l_wrist, person.r_wrist) if w is not None]
    if not wrists:
        return 0.0
    hand_thresh = max(phone.box.diag * 0.85, person.scale * 0.10)
    best = min(dist(phone.center, w) for w in wrists)
    if best <= hand_thresh:
        return 1.0
    if best <= hand_thresh * 1.5:
        return 0.5
    return 0.0


def get_person_zone_score(phone: PhoneCandidate, person: PersonPose, screen: ScreenConfig, width: int, height: int) -> tuple[float, Rect, list[Rect], Rect, bool]:
    person_roi, hand_rois, corridor = build_dynamic_zones(person, screen, width, height)
    c = phone.center
    person_box_score = 0.8 if point_in_box(c, person_roi) else 0.0
    wrist_score = get_hand_link_score(phone, person, screen)
    corridor_hit = point_in_box(c, corridor)
    corridor_score = 0.7 if corridor_hit else 0.0
    return max(person_box_score, wrist_score, corridor_score), person_roi, hand_rois, corridor, corridor_hit


def iter_arm_vectors(person: PersonPose, phone: PhoneCandidate):
    phone_center = phone.center
    for side, shoulder, elbow, wrist in (("L", person.l_shoulder, person.l_elbow, person.l_wrist), ("R", person.r_shoulder, person.r_elbow, person.r_wrist)):
        if wrist is None:
            continue
        if elbow is not None:
            yield f"{side}_forearm", elbow, (wrist[0] - elbow[0], wrist[1] - elbow[1])
        if shoulder is not None:
            yield f"{side}_upper_to_wrist", shoulder, (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
        yield f"{side}_wrist_phone", wrist, (phone_center[0] - wrist[0], phone_center[1] - wrist[1])


def get_aim_score(phone: PhoneCandidate, person: PersonPose, screen: ScreenConfig, static_zone_score: float, hand_link_score: float) -> tuple[float, float, bool, str]:
    screen_box = bbox_from_polygon(screen.screen_poly)
    screen_center = (screen_box.cx, screen_box.cy)
    target_box = bbox_from_polygon(screen.screen_poly)
    best_angle, best_ray_hit, best_reason = 180.0, False, "none"
    for reason, start, vec in iter_arm_vectors(person, phone):
        to_screen = (screen_center[0] - start[0], screen_center[1] - start[1])
        angle = angle_deg(vec, to_screen)
        ray_hit = ray_hits_box(start, vec, target_box, max_t=4.0)
        if ray_hit or angle < best_angle:
            best_angle, best_ray_hit, best_reason = angle, ray_hit, reason
    angle_thresh = float(screen.params["angle_thresh"])
    relaxed = float(screen.params["angle_relaxed_thresh"])
    if best_ray_hit or best_angle <= angle_thresh:
        return 1.0, best_angle, best_ray_hit, best_reason
    if best_angle <= relaxed and phone.conf >= 0.55 and hand_link_score >= 0.8 and static_zone_score >= 0.8:
        return 0.7, best_angle, best_ray_hit, f"relaxed:{best_reason}"
    return 0.0, best_angle, best_ray_hit, best_reason


def match_phone_to_person(phone: PhoneCandidate, persons: list[PersonPose], screen: ScreenConfig, width: int, height: int) -> tuple[PersonPose | None, float, float, Rect | None, list[Rect], Rect | None, bool]:
    best = (None, 0.0, 0.0, None, [], None, False)
    for person in persons:
        person_zone_score, person_roi, hand_rois, corridor, corridor_hit = get_person_zone_score(phone, person, screen, width, height)
        hand_score = get_hand_link_score(phone, person, screen)
        corridor_score = 0.7 if corridor_hit else 0.0
        score = 0.45 * hand_score + 0.35 * person_zone_score + 0.20 * corridor_score
        if score > best[1]:
            best = (person, score, hand_score, person_roi, hand_rois, corridor, corridor_hit)
    if best[1] < 0.35:
        return None, best[1], best[2], best[3], best[4], best[5], best[6]
    return best


def classify_candidate(risk_score: float) -> str:
    return candidate_level_from_risk(risk_score)




def evaluate_phone(phone: PhoneCandidate, screen: ScreenConfig, persons: list[PersonPose], width: int, height: int) -> CandidateEval:
    required_conf = float(screen.params["phone_valid_conf_person_roi"] if phone.source in ("person_roi", "person_screen_roi") else screen.params["phone_valid_conf_near_zone"])
    static_score, zone_reason = get_static_zone_score(phone, screen)
    source_score = 1.0 if phone.source in ("person_roi", "person_screen_roi") else 0.6
    reject = ""
    person = None
    person_match_score = 0.0
    hand_score = 0.0
    aim_score = 0.0
    risk = 0.0
    level = "ignore"
    best_angle = 180.0
    best_ray_hit = False
    candidate_reason = ""
    person_roi = None
    hand_rois: list[Rect] = []
    corridor = None
    corridor_hit = False
    phone_score = 0.0
    screen_relation_score = clamp(static_score, 0.0, 1.0)
    pose_score = 0.0
    temporal_score = 0.0
    aim_reason = "none"
    if zone_reason.startswith("ignore"):
        reject = "ignore_zone"
    elif phone.conf < required_conf:
        reject = "low_conf_for_source"
    elif static_score <= 0:
        reject = "outside_zone"
    else:
        related_persons = [p for p in persons if person_related_to_screen(p, screen, width, height)]
        if phone.person_id >= 0:
            preferred = [p for p in related_persons if p.index == phone.person_id]
            related_persons = preferred or related_persons
        person, person_match_score, hand_score, person_roi, hand_rois, corridor, corridor_hit = match_phone_to_person(phone, related_persons, screen, width, height)
        if person is None:
            reject = "no_matched_person"
        else:
            aim_score, best_angle, best_ray_hit, aim_reason = get_aim_score(phone, person, screen, static_score, hand_score)
            phone_score = phone_reliability_score(phone, required_conf)
            pose_score = pose_support_score(phone, person, aim_score, hand_score)
            risk = compute_risk_score(phone_score, hand_score, screen_relation_score, pose_score, temporal_score)
            level = classify_candidate(risk)
            if risk < 0.40 or person_match_score < 0.35:
                reject = "low_risk_score"
            candidate_reason = (
                f"{level}|{zone_reason}|aim={aim_reason}|phone={phone_score:.2f}|hand={hand_score:.2f}|"
                f"screen={screen_relation_score:.2f}|pose={pose_score:.2f}|risk={risk:.2f}"
            )
    ev = CandidateEval(phone, screen, person, static_score, zone_reason, person_match_score, hand_score, aim_score, source_score, risk, level, reject, candidate_reason, best_angle, best_ray_hit, person_roi, hand_rois, corridor)
    ev.phone_score = phone_score
    ev.phone_hand_score = hand_score
    ev.screen_relation_score = screen_relation_score
    ev.pose_score = pose_score
    ev.temporal_score = temporal_score
    ev.phone_to_screen_dist = box_to_poly_distance(phone.box, screen.screen_poly)
    if person is not None:
        ev.track_id = person.track_id
        ev.nearest_wrist, ev.phone_to_left_wrist_dist, ev.phone_to_right_wrist_dist = wrist_debug(phone, person)
        ev.phone_in_person_roi = person_roi is not None and point_in_box(phone.center, person_roi)
        ev.person_screen_relation = "corridor" if corridor_hit else zone_reason
    update_eval_risk(ev)
    return ev


def candidate_delta(level: str) -> float:
    if level == "strong":
        return 2.0
    if level == "normal":
        return 1.0
    if level == "weak":
        return 0.3
    return 0.0


def write_debug_row(writer: csv.DictWriter, frame_id: int, ev: CandidateEval, alert_counter: float, stable_alert: bool, effective_roi: Rect | None) -> None:
    p = ev.person
    phone = ev.phone
    writer.writerow(
        {
            "frame_id": frame_id,
            "screen_id": ev.screen.screen_id,
            "person_id": "" if p is None else p.track_id,
            "pose_person_index": "" if p is None else p.index,
            "person_conf": "" if p is None else f"{p.conf:.4f}",
            "phone_id": phone.phone_id,
            "phone_source": phone.source,
            "phone_conf": f"{phone.conf:.4f}",
            "phone_bbox": [round(phone.box.x1, 1), round(phone.box.y1, 1), round(phone.box.x2, 1), round(phone.box.y2, 1)],
            "phone_box": [round(phone.box.x1, 1), round(phone.box.y1, 1), round(phone.box.x2, 1), round(phone.box.y2, 1)],
            "phone_center_x": f"{phone.box.cx:.1f}",
            "phone_center_y": f"{phone.box.cy:.1f}",
            "nearest_wrist": ev.nearest_wrist,
            "phone_to_left_wrist_dist": "" if ev.phone_to_left_wrist_dist < 0 else f"{ev.phone_to_left_wrist_dist:.1f}",
            "phone_to_right_wrist_dist": "" if ev.phone_to_right_wrist_dist < 0 else f"{ev.phone_to_right_wrist_dist:.1f}",
            "phone_to_screen_dist": f"{ev.phone_to_screen_dist:.1f}",
            "phone_in_person_roi": ev.phone_in_person_roi,
            "person_screen_relation": ev.person_screen_relation,
            "in_near_zone": point_in_polygon(phone.center, ev.screen.near_zone),
            "in_danger_zone": ev.zone_reason.startswith("danger"),
            "danger_zone_name": ev.zone_reason,
            "in_ignore_zone": ev.zone_reason.startswith("ignore"),
            "static_zone_score": f"{ev.static_zone_score:.3f}",
            "person_related_to_screen": p is not None,
            "person_box": "" if p is None else [round(p.box.x1, 1), round(p.box.y1, 1), round(p.box.x2, 1), round(p.box.y2, 1)],
            "person_bbox": "" if p is None else [round(p.box.x1, 1), round(p.box.y1, 1), round(p.box.x2, 1), round(p.box.y2, 1)],
            "person_expand_roi": "" if ev.person_expand_roi is None else [round(ev.person_expand_roi.x1, 1), round(ev.person_expand_roi.y1, 1), round(ev.person_expand_roi.x2, 1), round(ev.person_expand_roi.y2, 1)],
            "effective_search_roi": "" if effective_roi is None else [round(effective_roi.x1, 1), round(effective_roi.y1, 1), round(effective_roi.x2, 1), round(effective_roi.y2, 1)],
            "phone_score": f"{ev.phone_score:.3f}",
            "phone_hand_score": f"{ev.phone_hand_score:.3f}",
            "screen_relation_score": f"{ev.screen_relation_score:.3f}",
            "pose_score": f"{ev.pose_score:.3f}",
            "temporal_score": f"{ev.temporal_score:.3f}",
            "hand_link_score": f"{ev.hand_link_score:.3f}",
            "person_match_score": f"{ev.person_match_score:.3f}",
            "person_screen_corridor_hit": ev.corridor_bbox is not None and point_in_box(phone.center, ev.corridor_bbox),
            "best_angle": f"{ev.best_angle:.2f}",
            "best_ray_hit": ev.best_ray_hit,
            "aim_score": f"{ev.aim_score:.3f}",
            "candidate_level": ev.level,
            "risk_score": f"{ev.risk_score:.3f}",
            "state": ev.person_state,
            "is_alarm": ev.person_alarm,
            "person_stable_count": ev.person_stable_count,
            "person_window_hits": ev.person_window_hits,
            "person_window_size": ev.person_window_size,
            "alert_counter": f"{alert_counter:.2f}",
            "stable_alert": stable_alert,
            "reject_reason": ev.reject_reason,
            "candidate_reason": ev.candidate_reason,
        }
    )


def draw_outputs(frame, screens: list[ScreenConfig], people: list[PersonPose], evals: list[CandidateEval], person_states: dict[int, PersonTrackState], alert_counter: float, stable_alert: bool, draw_static_zones: bool, draw_mode: str = "full") -> None:
    compact = draw_mode == "compact"
    for screen in screens:
        if draw_static_zones and not compact:
            draw_poly(frame, screen.near_zone, (0, 220, 255), 1, "NEAR")
            for dz in screen.danger_zones:
                draw_poly(frame, dz.polygon, (0, 145, 255), 1, f"DANGER:{dz.name}")
            for iz in screen.ignore_zones:
                draw_poly(frame, iz.polygon, (0, 0, 255), 2, f"IGNORE:{iz.name}")
        draw_poly(frame, screen.screen_poly, (255, 80, 0), 1 if compact else 2, None if compact else "SCREEN")
    for person in people:
        state = person_states.get(person.track_id)
        state_name = state.state[1:] if state is not None and state.state.startswith("S") else (state.state if state is not None else "")
        is_alarm_person = state is not None and state.alarm_triggered
        if compact and not is_alarm_person:
            continue
        person_color = (0, 0, 255) if is_alarm_person else (80, 180, 80)
        person_label = "ALARM" if compact and is_alarm_person else f"P{person.track_id}:{state_name}"
        draw_rect(frame, person.box, person_color, 2 if is_alarm_person else 1, person_label)
        if not compact:
            for pt in (person.l_wrist, person.r_wrist):
                if pt is not None:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (180, 255, 180), -1)
            for elbow, wrist in ((person.l_elbow, person.l_wrist), (person.r_elbow, person.r_wrist)):
                if elbow is not None and wrist is not None:
                    cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 255), 2)
    for ev in evals:
        color = (0, 0, 255) if ev.accepted else (120, 120, 120)
        if ev.accepted and ev.level == "strong":
            color = (0, 0, 255)
        elif ev.accepted:
            color = (0, 165, 255)
        elif ev.phone.source in ("person_roi", "person_screen_roi"):
            color = (180, 0, 180)
        if compact and not ev.accepted:
            continue
        draw_rect(frame, ev.phone.box, color, 3 if ev.accepted else 1, "PHONE" if compact else f"{ev.phone.source}:{ev.phone.conf:.2f}")
        if not compact:
            if ev.person_expand_roi is not None:
                draw_rect(frame, ev.person_expand_roi, (0, 180, 0), 1, "P_ROI")
            for hr in ev.hand_rois:
                draw_rect(frame, hr, (120, 255, 120), 1, "HAND")
            if ev.corridor_bbox is not None:
                draw_rect(frame, ev.corridor_bbox, (255, 255, 0), 1, "CORRIDOR")
            if ev.effective_search_roi is not None:
                draw_rect(frame, ev.effective_search_roi, (180, 0, 180), 1, "SEARCH")
            label = ev.candidate_reason if ev.accepted else ev.reject_reason
            draw_label(frame, label or "reject", int(ev.phone.box.x1), min(int(ev.phone.box.y2) + 26, frame.shape[0] - 5), color, scale=0.38)
        if ev.accepted and ev.person is not None and not compact:
            wrists = [w for w in (ev.person.l_wrist, ev.person.r_wrist) if w is not None]
            if wrists:
                w = min(wrists, key=lambda p: dist(p, ev.phone.center))
                cv2.line(frame, (int(w[0]), int(w[1])), (int(ev.phone.box.cx), int(ev.phone.box.cy)), (255, 255, 255), 2)
                sb = bbox_from_polygon(ev.screen.screen_poly)
                cv2.line(frame, (int(ev.phone.box.cx), int(ev.phone.box.cy)), (int(sb.cx), int(sb.cy)), (255, 255, 255), 1)
    status = "PERSON_ALERT" if stable_alert else "OK"
    color = (0, 0, 255) if stable_alert else (60, 180, 60)
    status_text = ("ALARM" if stable_alert else "OK") if compact else f"v2.2 {status} window_hits={alert_counter:.0f}"
    draw_label(frame, status_text, 12, 38, color, scale=0.55 if compact else 0.7)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    debug_csv = args.debug_csv or args.output.with_suffix(".debug.csv")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {args.output}")

    camera_id, screens = load_calibration(args.screen_config, width, height)
    pose_model = YOLO(str(args.pose_model), task="pose")
    phone_model = YOLO(str(args.phone_model), task="detect")
    phone_names = {int(k): str(v) for k, v in phone_model.names.items()}
    state_window = max(int(s.params.get("person_state_window", 30)) for s in screens) if screens else 30
    state_window = max(1, state_window)
    state_min_hits = max(int(s.params.get("person_state_min_hits", 16)) for s in screens) if screens else 16
    state_min_hits = max(1, min(state_min_hits, state_window))
    state_risk_threshold = max(float(s.params.get("person_state_risk_threshold", 0.65)) for s in screens) if screens else 0.65
    alert_counter = 0.0
    person_states: dict[int, PersonTrackState] = {}
    next_track_id = 1
    previous_alarm_tracks: set[int] = set()
    last_good = None
    stats = {
        "frames": 0,
        "gray_frames_replaced": 0,
        "pose_frames": 0,
        "phone_candidate_frames": 0,
        "phone_candidates_raw": 0,
        "phone_candidates_after_nms": 0,
        "accepted_candidate_frames": 0,
        "accepted_candidates": 0,
        "strong_candidates": 0,
        "normal_candidates": 0,
        "weak_candidates": 0,
        "stable_alert_frames": 0,
        "person_alarm_frames": 0,
        "person_alarm_track_frames": 0,
        "person_alarm_events": 0,
        "max_alert_counter": 0.0,
        "max_person_window_hits": 0,
        "max_risk_score": 0.0,
        "tracked_persons": 0,
        "screen_count": len(screens),
    }
    fields = [
        "frame_id",
        "screen_id",
        "person_id",
        "pose_person_index",
        "person_conf",
        "person_bbox",
        "person_box",
        "phone_id",
        "phone_source",
        "phone_conf",
        "phone_bbox",
        "phone_box",
        "phone_center_x",
        "phone_center_y",
        "nearest_wrist",
        "phone_to_left_wrist_dist",
        "phone_to_right_wrist_dist",
        "phone_to_screen_dist",
        "phone_in_person_roi",
        "person_screen_relation",
        "in_near_zone",
        "in_danger_zone",
        "danger_zone_name",
        "in_ignore_zone",
        "static_zone_score",
        "person_related_to_screen",
        "person_expand_roi",
        "effective_search_roi",
        "phone_score",
        "phone_hand_score",
        "screen_relation_score",
        "pose_score",
        "temporal_score",
        "hand_link_score",
        "person_match_score",
        "person_screen_corridor_hit",
        "best_angle",
        "best_ray_hit",
        "aim_score",
        "candidate_level",
        "risk_score",
        "state",
        "is_alarm",
        "person_stable_count",
        "person_window_hits",
        "person_window_size",
        "alert_counter",
        "stable_alert",
        "reject_reason",
        "candidate_reason",
    ]
    debug_csv.parent.mkdir(parents=True, exist_ok=True)
    with debug_csv.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fields)
        csv_writer.writeheader()
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if args.max_frames > 0 and frame_id > args.max_frames:
                break
            if is_decode_gray(frame) and last_good is not None:
                frame = last_good.copy()
                stats["gray_frames_replaced"] += 1
            else:
                last_good = frame.copy()

            pose_result = pose_model.predict(frame, imgsz=args.pose_imgsz, conf=args.pose_conf, device=args.device, verbose=False)[0]
            people = extract_people(pose_result, args.kp_conf, width, height)
            next_track_id = assign_person_track_ids(people, person_states, next_track_id, frame_id, state_window)
            roi_jobs: list[RoiJob] = []
            if args.phone_roi_mode == "person":
                p = default_params()
                for person in people:
                    person_roi = expand_rect(person.box, p["person_expand_x"], p["person_expand_y"], width, height)
                    roi_jobs.append(RoiJob(roi=person_roi, screen_id="", source="person_roi", person_id=person.index))
            else:
                for screen in screens:
                    related = [p for p in people if person_related_to_screen(p, screen, width, height)]
                    for person in related:
                        roi, person_roi, hand_rois, corridor = build_effective_search_roi(person, screen, width, height)
                        if roi is None:
                            continue
                        roi_jobs.append(RoiJob(roi=roi, screen_id=screen.screen_id, source="person_screen_roi", person_id=person.index))
                    interval = max(1, int(screen.params["near_zone_detect_interval"]))
                    if frame_id % interval == 0:
                        near_roi = bbox_from_polygon(screen.near_zone).clamp(width, height)
                        roi_jobs.append(RoiJob(roi=near_roi, screen_id=screen.screen_id, source="near_zone"))

            all_phones = detect_phone_in_rois(phone_model, phone_names, frame, roi_jobs, args.phone_candidate_conf, args.phone_imgsz, args.device, args.roi_batch_size)
            phones = nms_phone_candidates(all_phones, iou_thresh=0.5)
            evals: list[CandidateEval] = []
            for phone in phones:
                if phone.screen_id:
                    candidate_screens = [s for s in screens if s.screen_id == phone.screen_id]
                else:
                    candidate_screens = screens
                phone_evals: list[CandidateEval] = []
                for screen in candidate_screens:
                    ev = evaluate_phone(phone, screen, people, width, height)
                    ev.effective_search_roi = phone.roi if phone.source in ("person_roi", "person_screen_roi") else None
                    phone_evals.append(ev)
                accepted_evals = [e for e in phone_evals if e.accepted]
                if accepted_evals:
                    evals.append(max(accepted_evals, key=lambda e: (candidate_delta(e.level), e.risk_score)))
                elif phone_evals:
                    evals.append(max(phone_evals, key=lambda e: e.risk_score))

            apply_temporal_scores(evals, person_states, state_risk_threshold)
            accepted = [e for e in evals if e.accepted]
            update_person_states(people, evals, person_states, frame_id, state_window, state_risk_threshold, state_min_hits)
            sync_eval_state_fields(evals, person_states)
            visible_track_ids = {p.track_id for p in people if p.track_id >= 0}
            active_alarm_tracks = {tid for tid in visible_track_ids if tid in person_states and person_states[tid].alarm_triggered}
            new_alarm_tracks = active_alarm_tracks - previous_alarm_tracks
            alert_counter = float(max((person_states[tid].window_hits for tid in visible_track_ids if tid in person_states), default=0))
            stable_alert = bool(active_alarm_tracks)
            prune_person_tracks(person_states, frame_id, state_window)

            for ev in evals:
                write_debug_row(csv_writer, frame_id, ev, alert_counter, stable_alert, ev.effective_search_roi)

            annotated = frame.copy()
            draw_outputs(annotated, screens, people, evals, person_states, alert_counter, stable_alert, args.draw_static_zones, args.draw_mode)
            writer.write(annotated)


            stats["frames"] += 1
            stats["pose_frames"] += int(bool(people))
            stats["phone_candidate_frames"] += int(bool(phones))
            stats["phone_candidates_raw"] += len(all_phones)
            stats["phone_candidates_after_nms"] += len(phones)
            stats["accepted_candidate_frames"] += int(bool(accepted))
            stats["accepted_candidates"] += len(accepted)
            stats["strong_candidates"] += sum(1 for e in accepted if e.level == "strong")
            stats["normal_candidates"] += sum(1 for e in accepted if e.level == "normal")
            stats["weak_candidates"] += sum(1 for e in accepted if e.level == "weak")
            stats["stable_alert_frames"] += int(stable_alert)
            stats["person_alarm_frames"] += int(stable_alert)
            stats["person_alarm_track_frames"] += len(active_alarm_tracks)
            stats["person_alarm_events"] += len(new_alarm_tracks)
            stats["max_alert_counter"] = max(stats["max_alert_counter"], alert_counter)
            stats["max_person_window_hits"] = max(stats["max_person_window_hits"], int(alert_counter))
            stats["max_risk_score"] = max(stats["max_risk_score"], max((e.risk_score for e in evals), default=0.0))
            stats["tracked_persons"] = max(stats["tracked_persons"], len(person_states))
            previous_alarm_tracks = active_alarm_tracks
            if frame_id % 100 == 0 or frame_id == total:
                print(
                    f"[{frame_id}/{total}] raw={stats['phone_candidates_raw']} nms={stats['phone_candidates_after_nms']} "
                    f"accepted_frames={stats['accepted_candidate_frames']} stable={stats['stable_alert_frames']} "
                    f"window_hits={alert_counter:.0f} gray_fixed={stats['gray_frames_replaced']}",
                    flush=True,
                )


    cap.release()
    writer.release()
    summary = {
        "version": "v2.2",
        "camera_id": camera_id,
        "pose_model": str(args.pose_model),
        "phone_model": str(args.phone_model),
        "screen_config": str(args.screen_config),
        "video": str(args.video),
        "output": str(args.output),
        "debug_csv": str(debug_csv),
        "width": width,
        "height": height,
        "fps": fps,
        "settings": vars(args),
        "v2_2_state_machine": {
            "person_state_window": state_window,
            "person_state_min_hits": state_min_hits,
            "person_state_risk_threshold": state_risk_threshold,
            "risk_weights": {
                "phone_score": 0.30,
                "phone_hand_score": 0.20,
                "screen_relation_score": 0.20,
                "pose_score": 0.15,
                "temporal_score": 0.15,
            },
        },
        "stats": stats,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] video={args.output}")
    print(f"[DONE] debug_csv={debug_csv}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
