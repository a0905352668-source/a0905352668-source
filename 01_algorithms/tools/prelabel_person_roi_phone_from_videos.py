#!/usr/bin/env python3
"""Mine phone-positive Person ROI crops from surveillance videos as LabelMe data."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path("/media/boshi/Data/JianKong")
CORE_SCRIPT = PROJECT_ROOT / "01_algorithms/predict_surveillance_phone_pose_fixed_screens.py"
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "03_raw_videos_and_frames/2026-07-01"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "04_labelme_datasets/Person标注/20260703_20260701_7videos_person_roi_phone_prelabel_10fps_trt_2fps_pid3s"
)
DEFAULT_CALIB_DIR = PROJECT_ROOT / "02_configs/surveillance"
DEFAULT_POSE_MODEL = PROJECT_ROOT / "07_models/pose_models/yolo11s-pose.engine"
DEFAULT_PHONE_MODEL = (
    PROJECT_ROOT
    / "07_models/phone_models/TRUE_PERSON_ROI_PHONE_yolo11s_network_roi_plus_own_roi_20260701_best_epoch36.engine"
)


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("surveillance_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import core script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--video", type=Path, action="append", help="Video path. Defaults to all mp4 under --video-root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--phone-model", type=Path, default=DEFAULT_PHONE_MODEL)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--infer-fps", type=float, default=10.0)
    parser.add_argument("--save-fps", type=float, default=2.0)
    parser.add_argument("--per-id-seconds", type=float, default=3.0)
    parser.add_argument("--pose-imgsz", type=int, default=960)
    parser.add_argument("--phone-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--kp-conf", type=float, default=0.35)
    parser.add_argument("--phone-conf", type=float, default=0.25)
    parser.add_argument("--roi-batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def screen_config_for_video(video: Path, calib_dir: Path) -> Path | None:
    name = video.stem
    mapping = [
        ("软件1", "camera_software_01_screen_calibration_v21.json"),
        ("软件2", "camera_software_02_screen_calibration_v21.json"),
        ("机械1", "camera_mechanical_01_screen_calibration_v21.json"),
        ("机械2", "camera_mechanical_02_screen_calibration_v21.json"),
        ("电气1", "camera_01_screen_calibration_v21.json"),
        ("电气2", "camera_02_screen_calibration_v21.json"),
        ("software1", "camera_software_01_screen_calibration_v21.json"),
        ("software2", "camera_software_02_screen_calibration_v21.json"),
        ("mechanical1", "camera_mechanical_01_screen_calibration_v21.json"),
        ("mechanical2", "camera_mechanical_02_screen_calibration_v21.json"),
        ("electrical1", "camera_01_screen_calibration_v21.json"),
        ("electrical2", "camera_02_screen_calibration_v21.json"),
    ]
    for needle, filename in mapping:
        if needle in name:
            return calib_dir / filename
    return None


def safe_slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in "-_"):
            out.append(ch)
        else:
            code = f"u{ord(ch):04x}" if not ch.isascii() else "_"
            out.append(code)
    return "".join(out).strip("_") or "video"


def chinese_video_slug(stem: str) -> str:
    replacements = {
        "软件": "software",
        "机械": "mechanical",
        "电气": "electrical",
        "走廊": "corridor",
    }
    slug = stem
    for src, dst in replacements.items():
        slug = slug.replace(src, dst)
    return safe_slug(slug)


def make_labelme_json(image_name: str, width: int, height: int, shapes: list[dict[str, Any]], description: str) -> dict[str, Any]:
    return {
        "version": "2.4.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": description,
    }


def clamp_box_to_crop(phone_box, roi, crop_w: int, crop_h: int) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(crop_w), phone_box.x1 - roi.x1))
    y1 = max(0.0, min(float(crop_h), phone_box.y1 - roi.y1))
    x2 = max(0.0, min(float(crop_w), phone_box.x2 - roi.x1))
    y2 = max(0.0, min(float(crop_h), phone_box.y2 - roi.y1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def class_is_phone(names: dict[int, str], cls_id: int) -> bool:
    name = names.get(cls_id, str(cls_id)).lower()
    return name == "phone" or (len(names) == 1 and cls_id == 0)


def detect_phone_in_rois(core, model: YOLO, names: dict[int, str], frame, jobs: list[Any], conf: float, imgsz: int, device: str, batch_size: int) -> list[Any]:
    valid_jobs = []
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

    phones = []
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
                pconf = float(raw.conf.item())
                if pconf < conf or not class_is_phone(names, cls_id):
                    continue
                bx1, by1, bx2, by2 = [float(v) for v in raw.xyxy[0].tolist()]
                phones.append(
                    core.PhoneCandidate(
                        box=core.Rect(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
                        conf=pconf,
                        source=job.source,
                        screen_id=job.screen_id,
                        roi=job.roi,
                        person_id=job.person_id,
                    )
                )
    return phones


def nms_by_person(core, phones: list[Any]) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for phone in phones:
        grouped[int(phone.person_id)].append(phone)
    return {pid: core.nms_phone_candidates(items, iou_thresh=0.5) for pid, items in grouped.items()}


def should_process_time(timestamp: float, next_time: float, interval: float) -> tuple[bool, float]:
    if timestamp + 1e-6 < next_time:
        return False, next_time
    while next_time <= timestamp + 1e-6:
        next_time += interval
    return True, next_time


def write_outputs(
    out_dir: Path,
    image_prefix: str,
    frame,
    roi,
    phones: list[Any],
    person_id: int,
    frame_id: int,
    timestamp: float,
    source_video: Path,
    jpeg_quality: int,
) -> tuple[Path, Path, int, int]:
    x1, y1, x2, y2 = roi.as_int()
    crop = frame[y1:y2, x1:x2].copy()
    h, w = crop.shape[:2]
    shapes = []
    for phone in phones:
        rel = clamp_box_to_crop(phone.box, roi, w, h)
        if rel is None:
            continue
        px1, py1, px2, py2 = rel
        shapes.append(
            {
                "label": "phone",
                "points": [[px1, py1], [px2, py1], [px2, py2], [px1, py2]],
                "group_id": None,
                "description": f"conf={phone.conf:.4f} source={phone.source} person_id={person_id} t={timestamp:.3f}s",
                "shape_type": "rectangle",
                "flags": {},
            }
        )
    if not shapes:
        raise ValueError("No usable phone shapes inside crop")

    image_name = f"{image_prefix}.jpg"
    json_name = f"{image_prefix}.json"
    image_path = out_dir / image_name
    json_path = out_dir / json_name
    ok = cv2.imwrite(str(image_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {image_path}")
    description = (
        f"Person ROI crop from {source_video.stem}, person_id={person_id}, frame_id={frame_id}, "
        f"time={timestamp:.3f}s, 3s_bucket={int(timestamp // 3)}, "
        f"roi_xyxy_full_frame={[x1, y1, x2, y2]}"
    )
    label = make_labelme_json(image_name, w, h, shapes, description)
    json_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, json_path, w, h


def mine_video(args: argparse.Namespace, core, pose_model: YOLO, phone_model: YOLO, video: Path, csv_writer: csv.DictWriter) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    screen_config = screen_config_for_video(video, args.calib_dir)
    camera_id = ""
    if screen_config is not None and screen_config.exists():
        camera_id, _screens = core.load_calibration(screen_config, width, height)

    params = core.default_params()
    state_window = 30
    person_states: dict[int, Any] = {}
    next_track_id = 1
    infer_interval = 1.0 / max(0.001, args.infer_fps)
    save_interval = 1.0 / max(0.001, args.save_fps)
    next_infer_t = 0.0
    next_save_t = 0.0
    last_save_by_track: dict[int, float] = {}
    infer_idx = 0
    frame_id = 0
    stats = {
        "video": str(video),
        "screen_config": "" if screen_config is None else str(screen_config),
        "camera_id": camera_id,
        "native_fps": native_fps,
        "width": width,
        "height": height,
        "frames_total": total,
        "frames_read": 0,
        "frames_inferred": 0,
        "frames_save_slots": 0,
        "person_roi_jobs": 0,
        "phone_detections": 0,
        "saved_crops": 0,
        "skipped_per_id_throttle": 0,
        "started_at": time.strftime("%F %T"),
    }
    phone_names = {int(k): str(v) for k, v in phone_model.names.items()}
    video_slug = chinese_video_slug(video.stem)
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        stats["frames_read"] += 1
        if args.max_frames > 0 and frame_id > args.max_frames:
            break
        timestamp = (frame_id - 1) / native_fps if native_fps > 0 else 0.0
        do_infer, next_infer_t = should_process_time(timestamp, next_infer_t, infer_interval)
        if not do_infer:
            continue
        infer_idx += 1
        stats["frames_inferred"] += 1

        pose_result = pose_model.predict(frame, imgsz=args.pose_imgsz, conf=args.pose_conf, device=args.device, verbose=False)[0]
        people = core.extract_people(pose_result, args.kp_conf, width, height)
        next_track_id = core.assign_person_track_ids(people, person_states, next_track_id, infer_idx, state_window)
        core.prune_person_tracks(person_states, infer_idx, state_window)

        roi_jobs = []
        roi_by_track = {}
        for person in people:
            if person.track_id < 0:
                continue
            person_roi = core.expand_rect(person.box, params["person_expand_x"], params["person_expand_y"], width, height)
            roi_jobs.append(core.RoiJob(roi=person_roi, screen_id="", source="person_roi", person_id=person.track_id))
            roi_by_track[person.track_id] = person_roi
        stats["person_roi_jobs"] += len(roi_jobs)
        if not roi_jobs:
            continue

        phones = detect_phone_in_rois(
            core,
            phone_model,
            phone_names,
            frame,
            roi_jobs,
            args.phone_conf,
            args.phone_imgsz,
            args.device,
            args.roi_batch_size,
        )
        phones_by_track = nms_by_person(core, phones)
        stats["phone_detections"] += sum(len(items) for items in phones_by_track.values())
        if not phones_by_track:
            continue

        do_save_slot, next_save_t = should_process_time(timestamp, next_save_t, save_interval)
        if not do_save_slot:
            continue
        stats["frames_save_slots"] += 1

        for track_id, track_phones in sorted(phones_by_track.items()):
            roi = roi_by_track.get(track_id)
            if roi is None:
                continue
            last_t = last_save_by_track.get(track_id, -1e9)
            if timestamp - last_t < args.per_id_seconds - 1e-6:
                stats["skipped_per_id_throttle"] += 1
                continue
            bucket = int(math.floor(timestamp / max(args.per_id_seconds, 1e-6)))
            prefix = f"20260701_{video_slug}_pid{track_id}_b{bucket:04d}_f{frame_id:06d}"
            try:
                image_path, json_path, crop_w, crop_h = write_outputs(
                    args.output_dir,
                    prefix,
                    frame,
                    roi,
                    track_phones,
                    track_id,
                    frame_id,
                    timestamp,
                    video,
                    args.jpeg_quality,
                )
            except ValueError:
                continue
            last_save_by_track[track_id] = timestamp
            stats["saved_crops"] += 1
            csv_writer.writerow(
                {
                    "video": str(video),
                    "video_stem": video.stem,
                    "camera_id": camera_id,
                    "screen_config": "" if screen_config is None else str(screen_config),
                    "frame_id": frame_id,
                    "time_sec": f"{timestamp:.3f}",
                    "person_id": track_id,
                    "phone_count": len(track_phones),
                    "max_phone_conf": f"{max(p.conf for p in track_phones):.4f}",
                    "roi_xyxy": list(roi.as_int()),
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                    "image": str(image_path),
                    "json": str(json_path),
                }
            )

        if stats["frames_inferred"] % 100 == 0:
            elapsed = max(1e-6, time.time() - t0)
            print(
                f"[{video.name}] inferred={stats['frames_inferred']} saved={stats['saved_crops']} "
                f"read={stats['frames_read']}/{total} speed={stats['frames_inferred']/elapsed:.2f} infer_fps_wall",
                flush=True,
            )

    cap.release()
    stats["ended_at"] = time.strftime("%F %T")
    stats["elapsed_sec"] = round(time.time() - t0, 3)
    print(f"[DONE] {video.name} inferred={stats['frames_inferred']} saved={stats['saved_crops']}", flush=True)
    return stats


def main() -> None:
    args = parse_args()
    core = load_core(CORE_SCRIPT)
    videos = args.video or sorted(args.video_root.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found under {args.video_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = args.output_dir / "manifest.csv"
    summary_json = args.output_dir / "summary.json"

    print(f"[LOAD] pose_model={args.pose_model}", flush=True)
    pose_model = YOLO(str(args.pose_model), task="pose")
    print(f"[LOAD] phone_model={args.phone_model}", flush=True)
    phone_model = YOLO(str(args.phone_model), task="detect")

    fields = [
        "video",
        "video_stem",
        "camera_id",
        "screen_config",
        "frame_id",
        "time_sec",
        "person_id",
        "phone_count",
        "max_phone_conf",
        "roi_xyxy",
        "crop_w",
        "crop_h",
        "image",
        "json",
    ]
    summaries = []
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for video in videos:
            summaries.append(mine_video(args, core, pose_model, phone_model, video, writer))
            f.flush()
    summary = {
        "version": "person_roi_phone_prelabel_v1",
        "created_at": time.strftime("%F %T"),
        "settings": {
            "infer_fps": args.infer_fps,
            "save_fps": args.save_fps,
            "per_id_seconds": args.per_id_seconds,
            "pose_imgsz": args.pose_imgsz,
            "phone_imgsz": args.phone_imgsz,
            "pose_conf": args.pose_conf,
            "kp_conf": args.kp_conf,
            "phone_conf": args.phone_conf,
            "roi_batch_size": args.roi_batch_size,
            "device": args.device,
            "pose_model": str(args.pose_model),
            "phone_model": str(args.phone_model),
        },
        "videos": summaries,
        "total_saved_crops": sum(int(s["saved_crops"]) for s in summaries),
        "manifest_csv": str(manifest_csv),
        "output_dir": str(args.output_dir),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] output_dir={args.output_dir}", flush=True)
    print(f"[DONE] manifest={manifest_csv}", flush=True)
    print(f"[DONE] summary={summary_json}", flush=True)


if __name__ == "__main__":
    main()
