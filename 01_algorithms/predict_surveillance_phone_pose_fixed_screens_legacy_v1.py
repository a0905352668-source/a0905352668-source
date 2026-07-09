#!/usr/bin/env python3
"""Run phone + pose surveillance aiming logic with fixed screen calibration."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


FONT_SCALE_MULT = 1.5


@dataclass
class Box:
    name: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def w(self) -> float:
        return max(1.0, float(self.x2 - self.x1))

    @property
    def h(self) -> float:
        return max(1.0, float(self.y2 - self.y1))

    @property
    def diag(self) -> float:
        return math.hypot(self.w, self.h)


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

    def visible_points(self) -> list[tuple[float, float]]:
        pts = [self.nose, self.l_shoulder, self.r_shoulder, self.l_elbow, self.r_elbow, self.l_wrist, self.r_wrist]
        return [p for p in pts if p is not None]

    @property
    def scale(self) -> float:
        pts = self.visible_points()
        if len(pts) < 2:
            return 100.0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(80.0, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))


@dataclass
class AimMatch:
    phone: Box
    screen: Box
    person_index: int
    hand: tuple[float, float]
    angle: float
    hand_dist: float
    screen_dist: float
    ray_hit: bool
    reason: str
    level: str = "normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-model", required=True, type=Path)
    parser.add_argument("--phone-model", required=True, type=Path)
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pose-imgsz", type=int, default=1280)
    parser.add_argument("--phone-imgsz", type=int, default=1280)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--phone-conf", type=float, default=0.25)
    parser.add_argument("--screen-conf", type=float, default=1.0)
    parser.add_argument("--kp-conf", type=float, default=0.35)
    parser.add_argument("--aim-rule", choices=["v1", "tight"], default="v1")
    parser.add_argument("--angle-thresh", type=float, default=140.0)
    parser.add_argument("--screen-expand", type=float, default=0.20)
    parser.add_argument("--hand-phone-scale", type=float, default=0.85)
    parser.add_argument("--hand-person-scale", type=float, default=0.10)
    parser.add_argument("--screen-near-scale", type=float, default=1.00)
    parser.add_argument("--near-screen-override-scale", type=float, default=0.10)
    parser.add_argument("--alert-frames", type=int, default=5)
    parser.add_argument("--occluded-alert-frames", type=int, default=12)
    parser.add_argument("--alert-decay", type=int, default=1)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def is_decode_gray(frame) -> bool:
    b, g, r = cv2.split(frame)
    lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    neutral = (np.abs(b.astype(np.int16) - g.astype(np.int16)) < 7) & (
        np.abs(g.astype(np.int16) - r.astype(np.int16)) < 7
    )
    mid = (lum > 90) & (lum < 175)
    frac = float((neutral & mid).mean())
    mean = float(lum.mean())
    std = float(lum.std())
    return (frac > 0.68 and 100 < mean < 165) or (std < 12 and 95 < mean < 170)


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_deg(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def expand_box(box: Box, scale: float, width: int, height: int) -> tuple[float, float, float, float]:
    dx = box.w * scale
    dy = box.h * scale
    return max(0.0, box.x1 - dx), max(0.0, box.y1 - dy), min(width - 1.0, box.x2 + dx), min(height - 1.0, box.y2 + dy)


def point_to_box_distance(pt: tuple[float, float], box_xyxy: tuple[float, float, float, float]) -> float:
    x, y = pt
    x1, y1, x2, y2 = box_xyxy
    return math.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2))


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, a.w * a.h)
    area_b = max(1, b.w * b.h)
    return inter / max(1, area_a + area_b - inter)


def ray_hits_box(start: tuple[float, float], direction: tuple[float, float], box_xyxy, max_t: float) -> bool:
    sx, sy = start
    dx, dy = direction
    if math.hypot(dx, dy) < 1e-6:
        return False
    x1, y1, x2, y2 = box_xyxy
    tmin, tmax = 0.0, max_t
    for s, d, lo, hi in ((sx, dx, x1, x2), (sy, dy, y1, y2)):
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


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.65) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale *= FONT_SCALE_MULT
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + baseline + 6, y)
    x = max(0, min(frame.shape[1] - tw - 12, x))
    cv2.rectangle(frame, (x, y - th - baseline - 8), (x + tw + 10, y + 4), color, -1)
    cv2.putText(frame, text, (x + 5, y - baseline - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def get_point(kp, idx: int, conf: float) -> tuple[float, float] | None:
    if idx >= kp.shape[0] or float(kp[idx][2]) < conf:
        return None
    return float(kp[idx][0]), float(kp[idx][1])


def extract_people(pose_result, kp_conf: float) -> list[PersonPose]:
    if pose_result.keypoints is None or pose_result.keypoints.data is None:
        return []
    people = []
    for i, kp in enumerate(pose_result.keypoints.data.cpu().numpy()):
        person = PersonPose(
            index=i,
            nose=get_point(kp, 0, kp_conf),
            l_shoulder=get_point(kp, 5, kp_conf),
            r_shoulder=get_point(kp, 6, kp_conf),
            l_elbow=get_point(kp, 7, kp_conf),
            r_elbow=get_point(kp, 8, kp_conf),
            l_wrist=get_point(kp, 9, kp_conf),
            r_wrist=get_point(kp, 10, kp_conf),
        )
        if person.visible_points():
            people.append(person)
    return people


def load_fixed_screens(path: Path, width: int, height: int) -> list[Box]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    ref_w = float(cfg["image_width"])
    ref_h = float(cfg["image_height"])
    sx = width / ref_w
    sy = height / ref_h
    screens = []
    for item in cfg["screens"]:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        screens.append(
            Box(
                name="screen",
                conf=1.0,
                x1=int(round(x1 * sx)),
                y1=int(round(y1 * sy)),
                x2=int(round(x2 * sx)),
                y2=int(round(y2 * sy)),
            )
        )
    return screens


def extract_phone_boxes(result, names: dict[int, str], width: int, height: int, conf_th: float) -> list[Box]:
    boxes = []
    if result.boxes is None:
        return boxes
    for raw in result.boxes:
        cls_id = int(raw.cls.item())
        name = names.get(cls_id, str(cls_id)).lower()
        conf = float(raw.conf.item())
        if name != "phone" or conf < conf_th:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in raw.xyxy[0].tolist()]
        boxes.append(Box("phone", conf, max(0, min(width - 1, x1)), max(0, min(height - 1, y1)), max(0, min(width - 1, x2)), max(0, min(height - 1, y2))))
    return boxes


def iter_arm_candidates(person: PersonPose, phone: Box) -> Iterable[tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]]]:
    phone_center = (phone.cx, phone.cy)
    for side, shoulder, elbow, wrist in (("L", person.l_shoulder, person.l_elbow, person.l_wrist), ("R", person.r_shoulder, person.r_elbow, person.r_wrist)):
        if wrist is None:
            continue
        if elbow is not None:
            yield f"{side}_forearm", elbow, wrist, (wrist[0] - elbow[0], wrist[1] - elbow[1])
        if shoulder is not None:
            yield f"{side}_upper_to_wrist", shoulder, wrist, (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
        yield f"{side}_wrist_phone", wrist, phone_center, (phone_center[0] - wrist[0], phone_center[1] - wrist[1])


def person_near_box(person: PersonPose, box_xyxy: tuple[float, float, float, float], max_dist: float) -> bool:
    return any(point_to_box_distance(point, box_xyxy) <= max_dist for point in person.visible_points())


def match_suspicious_aim(phones: list[Box], screens: list[Box], people: list[PersonPose], width: int, height: int, args) -> list[AimMatch]:
    matches = []
    if not phones or not screens or not people:
        return matches
    for phone in phones:
        phone_center = (phone.cx, phone.cy)
        for screen in screens:
            expanded = expand_box(screen, args.screen_expand, width, height)
            screen_dist = point_to_box_distance(phone_center, expanded)
            near_thresh = max(screen.diag * args.screen_near_scale, phone.diag * 1.5)
            very_near_thresh = max(screen.diag * args.near_screen_override_scale * 0.5, phone.diag * 0.45)
            for person in people:
                hand_points = [p for p in (person.l_wrist, person.r_wrist) if p is not None]
                nearest_hand = min(hand_points, key=lambda p: dist(p, phone_center)) if hand_points else phone_center
                hand_dist = dist(nearest_hand, phone_center) if hand_points else 999999.0
                hand_thresh = max(phone.diag * args.hand_phone_scale, person.scale * args.hand_person_scale)
                person_screen_dist = max(screen.diag * 0.25, person.scale * 0.50)
                phone_screen_iou = box_iou(phone, screen)
                phone_too_screen_like = phone_screen_iou > 0.15 or (phone.w * phone.h) > (screen.w * screen.h) * 0.30
                occluded_ok = (
                    not hand_points
                    and screen_dist <= very_near_thresh
                    and person_near_box(person, expanded, person_screen_dist)
                    and not phone_too_screen_like
                )
                if screen_dist > near_thresh and not occluded_ok:
                    continue
                if hand_dist > hand_thresh:
                    if occluded_ok:
                        matches.append(AimMatch(phone, screen, person.index, phone_center, 180.0, hand_dist, screen_dist, False, "occluded_phone_screen", "occluded"))
                    continue
                screen_center = (screen.cx, screen.cy)
                best_angle, best_ray_hit, best_reason = 180.0, False, ""
                wrist_phone_angle, wrist_phone_ray_hit = 180.0, False
                for reason, start, _end, vec in iter_arm_candidates(person, phone):
                    to_screen = (screen_center[0] - start[0], screen_center[1] - start[1])
                    angle = angle_deg(vec, to_screen)
                    ray_hit = ray_hits_box(start, vec, expanded, max_t=4.0)
                    if ray_hit or angle < best_angle:
                        best_angle, best_ray_hit, best_reason = angle, ray_hit, reason
                    if reason.endswith("wrist_phone"):
                        wrist_phone_angle = min(wrist_phone_angle, angle)
                        wrist_phone_ray_hit = wrist_phone_ray_hit or ray_hit
                if args.aim_rule == "v1":
                    near_screen_override = screen_dist <= very_near_thresh
                    rule_ok = best_ray_hit or best_angle <= args.angle_thresh or near_screen_override
                    final_angle, final_ray_hit = best_angle, best_ray_hit
                    if near_screen_override and not (best_ray_hit or best_angle <= args.angle_thresh):
                        final_reason = "near_screen_override"
                    else:
                        final_reason = best_reason
                else:
                    wrist_phone_ok = wrist_phone_ray_hit or wrist_phone_angle <= args.angle_thresh
                    forearm_ok = (best_ray_hit or best_angle <= args.angle_thresh) and screen_dist <= 1e-3 and hand_dist <= phone.diag * 0.45
                    rule_ok = wrist_phone_ok or forearm_ok
                    final_angle = min(best_angle, wrist_phone_angle)
                    final_ray_hit = best_ray_hit or wrist_phone_ray_hit
                    final_reason = "wrist_phone" if wrist_phone_ok else best_reason
                if rule_ok:
                    matches.append(AimMatch(phone, screen, person.index, nearest_hand, final_angle, hand_dist, screen_dist, final_ray_hit, final_reason))
    return matches


def draw_outputs(frame, phones: list[Box], screens: list[Box], people: list[PersonPose], matches: list[AimMatch]) -> None:
    matched_phone_ids = {id(m.phone) for m in matches}
    matched_screen_ids = {id(m.screen) for m in matches}
    for screen in screens:
        color = (0, 0, 255) if id(screen) in matched_screen_ids else (255, 120, 0)
        thick = 4 if id(screen) in matched_screen_ids else 2
        cv2.rectangle(frame, (screen.x1, screen.y1), (screen.x2, screen.y2), color, thick)
        draw_label(frame, "SCREEN_FIXED", screen.x1, screen.y1, color, scale=0.42)
    for phone in phones:
        color = (0, 0, 255) if id(phone) in matched_phone_ids else (0, 180, 0)
        thick = 4 if id(phone) in matched_phone_ids else 2
        cv2.rectangle(frame, (phone.x1, phone.y1), (phone.x2, phone.y2), color, thick)
        draw_label(frame, f"PHONE {phone.conf:.2f}", phone.x1, phone.y1, color)
    for person in people:
        for pt, color in ((person.l_wrist, (0, 255, 0)), (person.r_wrist, (0, 220, 255)), (person.l_elbow, (180, 255, 0)), (person.r_elbow, (180, 255, 0))):
            if pt is not None:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)
        for elbow, wrist in ((person.l_elbow, person.l_wrist), (person.r_elbow, person.r_wrist)):
            if elbow is not None and wrist is not None:
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 255), 2)
    for match in matches:
        pc = (int(match.phone.cx), int(match.phone.cy))
        sc = (int(match.screen.cx), int(match.screen.cy))
        hand = (int(match.hand[0]), int(match.hand[1]))
        color = (0, 0, 255) if match.level == "normal" else (0, 120, 255)
        cv2.line(frame, hand, pc, color, 3)
        cv2.line(frame, pc, sc, color, 2)
        prefix = "AIM_ALERT" if match.level == "normal" else "OCCLUDED_ALERT"
        draw_label(frame, f"{prefix} angle={match.angle:.0f} hand={match.hand_dist:.0f} {match.reason}", match.phone.x1, min(match.phone.y2 + 28, frame.shape[0] - 5), color, scale=0.58)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
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

    pose_model = YOLO(str(args.pose_model))
    phone_model = YOLO(str(args.phone_model))
    phone_names = {int(k): str(v) for k, v in phone_model.names.items()}
    screens = load_fixed_screens(args.screen_config, width, height)
    alert_counter = 0
    occluded_alert_counter = 0
    last_good = None
    stats = {
        "frames": 0,
        "gray_frames_replaced": 0,
        "pose_frames": 0,
        "phone_frames": 0,
        "raw_phone_boxes": 0,
        "fixed_screen_count": len(screens),
        "candidate_aim_frames": 0,
        "candidate_aim_matches": 0,
        "occluded_candidate_frames": 0,
        "occluded_candidate_matches": 0,
        "stable_alert_frames": 0,
        "stable_occluded_alert_frames": 0,
        "max_alert_counter": 0,
        "max_occluded_alert_counter": 0,
    }

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if is_decode_gray(frame) and last_good is not None:
            frame = last_good.copy()
            stats["gray_frames_replaced"] += 1
        else:
            last_good = frame.copy()

        pose_result = pose_model.predict(frame, imgsz=args.pose_imgsz, conf=args.pose_conf, device=args.device, verbose=False)[0]
        people = extract_people(pose_result, args.kp_conf)
        phone_result = phone_model.predict(frame, imgsz=args.phone_imgsz, conf=args.phone_conf, device=args.device, verbose=False)[0]
        phones = extract_phone_boxes(phone_result, phone_names, width, height, args.phone_conf)
        matches = match_suspicious_aim(phones, screens, people, width, height, args)
        normal_matches = [match for match in matches if match.level == "normal"]
        occluded_matches = [match for match in matches if match.level == "occluded"]
        if normal_matches:
            alert_counter = min(args.alert_frames, alert_counter + 1)
            stats["candidate_aim_frames"] += 1
            stats["candidate_aim_matches"] += len(normal_matches)
        else:
            alert_counter = max(0, alert_counter - args.alert_decay)
        if occluded_matches:
            occluded_alert_counter = min(args.occluded_alert_frames, occluded_alert_counter + 1)
            stats["occluded_candidate_frames"] += 1
            stats["occluded_candidate_matches"] += len(occluded_matches)
        else:
            occluded_alert_counter = max(0, occluded_alert_counter - args.alert_decay)
        stable_normal_alert = alert_counter >= args.alert_frames
        stable_occluded_alert = occluded_alert_counter >= args.occluded_alert_frames
        stable_alert = stable_normal_alert or stable_occluded_alert

        annotated = frame.copy()
        draw_outputs(annotated, phones, screens, people, matches)
        status_color = (0, 0, 255) if stable_alert else ((0, 180, 255) if matches else (60, 180, 60))
        if stable_normal_alert:
            status = "STABLE_ALERT"
        elif stable_occluded_alert:
            status = "STABLE_OCCLUDED_ALERT"
        elif normal_matches:
            status = "CANDIDATE_AIM"
        elif occluded_matches:
            status = "CANDIDATE_OCCLUDED"
        else:
            status = "OK"
        draw_label(annotated, f"{status} aim={alert_counter}/{args.alert_frames} occ={occluded_alert_counter}/{args.occluded_alert_frames} phones={len(phones)} screens={len(screens)} people={len(people)}", 12, 34, status_color, scale=0.75)
        writer.write(annotated)

        stats["frames"] += 1
        stats["pose_frames"] += int(bool(people))
        stats["phone_frames"] += int(bool(phones))
        stats["raw_phone_boxes"] += len(phones)
        stats["stable_alert_frames"] += int(stable_alert)
        stats["stable_occluded_alert_frames"] += int(stable_occluded_alert)
        stats["max_alert_counter"] = max(stats["max_alert_counter"], alert_counter)
        stats["max_occluded_alert_counter"] = max(stats["max_occluded_alert_counter"], occluded_alert_counter)
        if index % 100 == 0 or index == total:
            print(f"[{index}/{total}] phone_frames={stats['phone_frames']} pose_frames={stats['pose_frames']} candidate={stats['candidate_aim_frames']} occluded={stats['occluded_candidate_frames']} stable={stats['stable_alert_frames']} gray_fixed={stats['gray_frames_replaced']}", flush=True)

    cap.release()
    writer.release()
    summary = {
        "pose_model": str(args.pose_model),
        "phone_model": str(args.phone_model),
        "screen_config": str(args.screen_config),
        "video": str(args.video),
        "output": str(args.output),
        "width": width,
        "height": height,
        "fps": fps,
        "settings": {
            "pose_imgsz": args.pose_imgsz,
            "phone_imgsz": args.phone_imgsz,
            "pose_conf": args.pose_conf,
            "phone_conf": args.phone_conf,
            "kp_conf": args.kp_conf,
            "aim_rule": args.aim_rule,
            "angle_thresh": args.angle_thresh,
            "screen_expand": args.screen_expand,
            "hand_phone_scale": args.hand_phone_scale,
            "hand_person_scale": args.hand_person_scale,
            "screen_near_scale": args.screen_near_scale,
            "near_screen_override_scale": args.near_screen_override_scale,
            "alert_frames": args.alert_frames,
            "occluded_alert_frames": args.occluded_alert_frames,
            "alert_decay": args.alert_decay,
        },
        "stats": stats,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] video={args.output}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
