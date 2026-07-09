#!/usr/bin/env python3
"""Convert LabelMe screen polygons into JianKong v2.1 screen calibration JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT = Path("/media/boshi/Data/JianKong")
LABELME_DIR = PROJECT / "02_configs/surveillance/recalibration_20260706"
CONFIG_DIR = PROJECT / "02_configs/surveillance"

VIEWS = [
    ("dianqi1", "camera_01", "camera_01_screen_calibration_v21.json"),
    ("dianqi2", "camera_02", "camera_02_screen_calibration_v21.json"),
    ("jixie1", "camera_mechanical_01", "camera_mechanical_01_screen_calibration_v21.json"),
    ("jixie2", "camera_mechanical_02", "camera_mechanical_02_screen_calibration_v21.json"),
    ("ruanjian1", "camera_software_01", "camera_software_01_screen_calibration_v21.json"),
    ("ruanjian2", "camera_software_02", "camera_software_02_screen_calibration_v21.json"),
    ("zoulang", "camera_corridor", "camera_corridor_screen_calibration_v21.json"),
]

DEFAULT_PARAMS: dict[str, Any] = {
    "enable_dynamic_person_zone": True,
    "person_expand_x": 0.35,
    "person_expand_y": 0.20,
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def rect_poly(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)], [round(x1, 2), round(y2, 2)]]


def expand_box(box: tuple[float, float, float, float], xr: float, yr: float, width: int, height: int) -> list[list[float]]:
    x1, y1, x2, y2 = box
    dx = max(1.0, x2 - x1) * xr
    dy = max(1.0, y2 - y1) * yr
    return rect_poly(
        clamp(x1 - dx, 0, width - 1),
        clamp(y1 - dy, 0, height - 1),
        clamp(x2 + dx, 0, width - 1),
        clamp(y2 + dy, 0, height - 1),
    )


def old_params(filename: str) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    path = CONFIG_DIR / filename
    if not path.exists():
        return params
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return params
    for screen in data.get("screens", []):
        if screen.get("params"):
            params.update(screen["params"])
            return params
    return params


def shape_points(shape: dict[str, Any]) -> list[list[float]]:
    points = shape.get("points", [])
    if shape.get("shape_type") == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        return rect_poly(float(x1), float(y1), float(x2), float(y2))
    out = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            out.append([round(float(point[0]), 2), round(float(point[1]), 2)])
    return out


def is_screen_shape(shape: dict[str, Any]) -> bool:
    label = str(shape.get("label", "")).strip().lower()
    return label in {"screen", "monitor", "display", "屏幕"} or "screen" in label or "屏幕" in label


def read_labelme(view: str) -> tuple[int, int, list[list[list[float]]], str]:
    image_path = LABELME_DIR / f"{view}_middle_frame.jpg"
    json_path = LABELME_DIR / f"{view}_middle_frame.json"
    width = height = 0
    if image_path.exists():
        with Image.open(image_path) as image:
            width, height = image.size
    if not json_path.exists():
        return width or 2560, height or 1440, [], "missing_json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(data.get("imageWidth") or width or 2560)
    height = int(data.get("imageHeight") or height or 1440)
    polys = []
    ignored = 0
    for shape in data.get("shapes", []):
        if not is_screen_shape(shape):
            ignored += 1
            continue
        points = shape_points(shape)
        if len(points) >= 4:
            polys.append(points)
        else:
            ignored += 1
    note = f"ignored_{ignored}" if ignored else "ok"
    return width, height, polys, note


def build_config(view: str, camera_id: str, filename: str) -> tuple[dict[str, Any], dict[str, Any]]:
    width, height, polygons, note = read_labelme(view)
    params = old_params(filename)
    screens = []
    for idx, points in enumerate(polygons, start=1):
        b = bbox(points)
        front = expand_box(b, 0.65, 0.55, width, height)
        fx1, fy1, fx2, fy2 = bbox(front)
        left = rect_poly(fx1, fy1 + (fy2 - fy1) * 0.15, fx1 + (fx2 - fx1) * 0.55, fy2 - (fy2 - fy1) * 0.05)
        right = rect_poly(fx1 + (fx2 - fx1) * 0.45, fy1 + (fy2 - fy1) * 0.15, fx2, fy2 - (fy2 - fy1) * 0.05)
        screens.append(
            {
                "screen_id": f"screen_{idx:02d}",
                "screen_poly": points,
                "near_zone": expand_box(b, 1.20, 0.90, width, height),
                "danger_zones": [
                    {"name": "front", "polygon": front, "weight": 1.0},
                    {"name": "left_side", "polygon": left, "weight": 0.8},
                    {"name": "right_side", "polygon": right, "weight": 0.8},
                ],
                "ignore_zones": [],
                "params": params,
            }
        )
    config = {
        "version": "2.1",
        "camera_id": camera_id,
        "frame_size": [width, height],
        "description": f"Recalibrated from LabelMe file {view}_middle_frame.json on 2026-07-06. Empty screens means this view has no fixed monitor target.",
        "screens": screens,
    }
    summary = {"view": view, "filename": filename, "frame_size": [width, height], "screens": len(screens), "note": note}
    return config, summary


def draw_preview(view: str, config: dict[str, Any]) -> str:
    src = LABELME_DIR / f"{view}_middle_frame.jpg"
    out = LABELME_DIR / f"{view}_labelme_v21_preview.jpg"
    if not src.exists():
        return ""
    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for screen in config.get("screens", []):
        near_poly = [tuple(p) for p in screen["near_zone"]]
        screen_poly = [tuple(p) for p in screen["screen_poly"]]
        draw.polygon(near_poly, outline=(255, 180, 0, 255), fill=(255, 180, 0, 32))
        for zone in screen.get("danger_zones", []):
            draw.polygon([tuple(p) for p in zone["polygon"]], outline=(255, 0, 255, 210), fill=(255, 0, 255, 20))
        draw.polygon(screen_poly, outline=(255, 40, 40, 255), fill=(255, 40, 40, 55))
    image.save(out, quality=92)
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write official calibration files")
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary = []
    generated_dir = LABELME_DIR / f"generated_v21_from_labelme_{timestamp}"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for view, camera_id, filename in VIEWS:
        config, item_summary = build_config(view, camera_id, filename)
        generated_path = generated_dir / filename
        generated_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        item_summary["generated_path"] = str(generated_path)
        item_summary["preview_path"] = draw_preview(view, config)
        if args.apply:
            official = CONFIG_DIR / filename
            if official.exists():
                backup = CONFIG_DIR / f"{filename}.bak_before_labelme_{timestamp}"
                shutil.copy2(official, backup)
                item_summary["backup_path"] = str(backup)
            shutil.copy2(generated_path, official)
            item_summary["official_path"] = str(official)
        summary.append(item_summary)
    summary_path = generated_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"generated_dir": str(generated_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
