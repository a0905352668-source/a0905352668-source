#!/usr/bin/env python3
"""Parallel 7-view Person ROI phone prelabel mining for LabelMe."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path("/media/boshi/Data/JianKong")
CORE_SCRIPT = PROJECT_ROOT / "01_algorithms/predict_surveillance_phone_pose_fixed_screens.py"


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("jk_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load core script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-script", type=Path, default=CORE_SCRIPT)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--phone-model", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--infer-fps", type=float, default=10.0)
    parser.add_argument("--save-fps", type=float, default=2.0)
    parser.add_argument("--per-id-seconds", type=float, default=3.0)
    parser.add_argument("--pose-imgsz", type=int, default=960)
    parser.add_argument("--phone-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--kp-conf", type=float, default=0.35)
    parser.add_argument("--phone-conf", type=float, default=0.25)
    parser.add_argument("--phone-class-ids", default="0")
    parser.add_argument("--pose-batch-size", type=int, default=7)
    parser.add_argument("--phone-batch-size", type=int, default=16)
    parser.add_argument("--dynamic-pose-batch", action="store_true")
    parser.add_argument("--dynamic-phone-batch", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-sampled-frames", type=int, default=0)
    parser.add_argument("--cv-threads", type=int, default=1)
    parser.add_argument("--reader", choices=["cv2", "ffmpeg"], default="cv2")
    return parser.parse_args()


def parse_class_ids(value: str) -> set[int] | None:
    value = (value or "").strip()
    if not value or value.lower() in {"auto", "names"}:
        return None
    out = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def view_key_from_path(path: Path) -> str:
    name = f"{path.parent.name}_{path.name}".lower()
    if "dianqi1" in name or "electrical1" in name:
        return "dianqi1"
    if "dianqi2" in name or "electrical2" in name:
        return "dianqi2"
    if "jixie1" in name or "mechanical1" in name:
        return "jixie1"
    if "jixie2" in name or "mechanical2" in name:
        return "jixie2"
    if "ruanjian1" in name or "software1" in name:
        return "ruanjian1"
    if "ruanjian2" in name or "software2" in name:
        return "ruanjian2"
    if "zoulang" in name or "corridor" in name:
        return "zoulang"
    return "unknown"


def view_sort_key(key: str) -> tuple[int, str]:
    order = {
        "dianqi1": 0,
        "dianqi2": 1,
        "jixie1": 2,
        "jixie2": 3,
        "ruanjian1": 4,
        "ruanjian2": 5,
        "zoulang": 6,
    }
    return (order.get(key, 99), key)


def calib_for_view(calib_dir: Path, view_key: str) -> Path | None:
    mapping = {
        "dianqi1": "camera_01_screen_calibration_v21.json",
        "dianqi2": "camera_02_screen_calibration_v21.json",
        "jixie1": "camera_mechanical_01_screen_calibration_v21.json",
        "jixie2": "camera_mechanical_02_screen_calibration_v21.json",
        "ruanjian1": "camera_software_01_screen_calibration_v21.json",
        "ruanjian2": "camera_software_02_screen_calibration_v21.json",
        "zoulang": "camera_corridor_screen_calibration_v21.json",
    }
    name = mapping.get(view_key)
    return calib_dir / name if name else None


def find_last_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        candidates = [p for p in subdir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}]
        if candidates:
            videos.append(max(candidates, key=lambda p: (p.stat().st_mtime, p.name)))
    return sorted(videos, key=lambda p: view_sort_key(view_key_from_path(p)))


def predict_batches(model: YOLO, images: list[Any], batch_size: int, imgsz: int, conf: float, device: str, pad_to_batch: bool) -> list[Any]:
    if not images:
        return []
    out: list[Any] = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        real_len = len(chunk)
        if pad_to_batch and real_len < batch_size:
            chunk = chunk + [chunk[-1]] * (batch_size - real_len)
        results = model.predict(chunk, imgsz=imgsz, conf=conf, device=device, rect=False, verbose=False)
        out.extend(results[:real_len])
    return out


@dataclass
class Stream:
    idx: int
    video: Path
    view_key: str
    calib_path: Path | None
    cap: Any
    ffmpeg_proc: Any
    frame_bytes: int
    width: int
    height: int
    native_fps: float
    total_frames: int
    sample_indices: list[int]
    camera_id: str = ""
    person_states: dict[int, Any] = field(default_factory=dict)
    next_track_id: int = 1
    state_window: int = 30
    state_min_hits: int = 16
    state_risk_threshold: float = 0.65
    sample_cursor: int = 0
    raw_cursor: int = 0
    last_good: Any = None
    done: bool = False
    next_save_t: float = 0.0
    last_save_by_track: dict[int, float] = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)


def build_streams(args: argparse.Namespace, core: Any) -> list[Stream]:
    streams: list[Stream] = []
    for idx, video in enumerate(find_last_videos(args.root)):
        view_key = view_key_from_path(video)
        calib_path = calib_for_view(args.calib_dir, view_key)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / native_fps if native_fps > 0 else 0.0
        sampled_total = int(math.floor(duration * args.infer_fps)) + 1 if duration > 0 else total_frames
        sample_indices = sorted(
            {
                min(total_frames - 1, int(round(i * native_fps / args.infer_fps)))
                for i in range(max(0, sampled_total))
                if total_frames <= 0 or int(round(i * native_fps / args.infer_fps)) < total_frames
            }
        )
        if args.max_sampled_frames > 0:
            sample_indices = sample_indices[: args.max_sampled_frames]
        ffmpeg_proc = None
        if args.reader == "ffmpeg":
            cap.release()
            cap = None
            fps_expr = f"fps={args.infer_fps:.6f}"
            ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video),
                    "-vf",
                    fps_expr,
                    "-pix_fmt",
                    "bgr24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=width * height * 3 * 2,
            )
        stream = Stream(
            idx=idx,
            video=video,
            view_key=view_key,
            calib_path=calib_path,
            cap=cap,
            ffmpeg_proc=ffmpeg_proc,
            frame_bytes=width * height * 3,
            width=width,
            height=height,
            native_fps=native_fps,
            total_frames=total_frames,
            sample_indices=sample_indices,
        )
        if calib_path is not None and calib_path.exists():
            stream.camera_id, screens = core.load_calibration(calib_path, width, height)
            if screens:
                stream.state_window = max(int(s.params.get("person_state_window", 30)) for s in screens)
                stream.state_min_hits = max(int(s.params.get("person_state_min_hits", 16)) for s in screens)
                stream.state_min_hits = max(1, min(stream.state_min_hits, stream.state_window))
                stream.state_risk_threshold = max(float(s.params.get("person_state_risk_threshold", 0.65)) for s in screens)
        streams.append(stream)
    return streams


core_ref = None


def read_next_sample(stream: Stream):
    if stream.sample_cursor >= len(stream.sample_indices):
        stream.done = True
        return None, None
    target = stream.sample_indices[stream.sample_cursor]
    if stream.ffmpeg_proc is not None:
        assert stream.ffmpeg_proc.stdout is not None
        data = stream.ffmpeg_proc.stdout.read(stream.frame_bytes)
        if len(data) != stream.frame_bytes:
            stream.done = True
            return None, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape((stream.height, stream.width, 3))
        stream.sample_cursor += 1
        return target + 1, frame
    while stream.raw_cursor < target:
        ok = stream.cap.grab()
        if not ok:
            stream.done = True
            return None, None
        stream.raw_cursor += 1
    ok, frame = stream.cap.read()
    if not ok:
        stream.done = True
        return None, None
    stream.raw_cursor += 1
    stream.sample_cursor += 1
    if hasattr(core_ref, "is_decode_gray") and core_ref.is_decode_gray(frame) and stream.last_good is not None:
        frame = stream.last_good.copy()
        stream.stats["gray_frames_replaced"] += 1
    else:
        stream.last_good = frame.copy()
    return target + 1, frame


def class_is_phone(names: dict[int, str], phone_class_ids: set[int] | None, cls_id: int) -> bool:
    if phone_class_ids is not None:
        return cls_id in phone_class_ids
    name = names.get(cls_id, str(cls_id)).lower()
    return name == "phone" or (len(names) == 1 and cls_id == 0)


def detect_phones_batched(
    core: Any,
    model: YOLO,
    names: dict[int, str],
    phone_class_ids: set[int] | None,
    refs: list[tuple[Stream, int, Any, Any]],
    conf: float,
    imgsz: int,
    device: str,
    batch_size: int,
    pad_to_batch: bool,
) -> dict[tuple[int, int], list[Any]]:
    crops = []
    crop_refs: list[tuple[Stream, int, Any, int, int]] = []
    for stream, frame_id, frame, job in refs:
        x1, y1, x2, y2 = job.roi.as_int()
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
        crop_refs.append((stream, frame_id, job, x1, y1))

    phones_by_track: dict[tuple[int, int], list[Any]] = defaultdict(list)
    if not crops:
        return phones_by_track

    results = predict_batches(model, crops, batch_size, imgsz, conf, device, pad_to_batch)
    for (stream, _frame_id, job, ox, oy), result in zip(crop_refs, results):
        if result.boxes is None:
            continue
        for raw in result.boxes:
            cls_id = int(raw.cls.item())
            pconf = float(raw.conf.item())
            if pconf < conf or not class_is_phone(names, phone_class_ids, cls_id):
                continue
            bx1, by1, bx2, by2 = [float(v) for v in raw.xyxy[0].tolist()]
            phone = core.PhoneCandidate(
                box=core.Rect(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
                conf=pconf,
                source=job.source,
                screen_id=job.screen_id,
                roi=job.roi,
                person_id=job.person_id,
            )
            phones_by_track[(stream.idx, int(job.person_id))].append(phone)
    return phones_by_track


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


def safe_slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in "-_"):
            out.append(ch)
        else:
            out.append(f"u{ord(ch):04x}" if not ch.isascii() else "_")
    return "".join(out).strip("_") or "video"


def save_person_roi_crop(
    args: argparse.Namespace,
    stream: Stream,
    frame_id: int,
    frame,
    roi,
    phones: list[Any],
    person_id: int,
    timestamp: float,
    writer: csv.DictWriter,
) -> bool:
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
        return False
    bucket = int(math.floor(timestamp / max(args.per_id_seconds, 1e-6)))
    prefix = f"20260703_{stream.view_key}_{safe_slug(stream.video.stem)}_pid{person_id}_b{bucket:04d}_f{frame_id:06d}"
    image_name = f"{prefix}.jpg"
    json_name = f"{prefix}.json"
    image_path = args.output_dir / image_name
    json_path = args.output_dir / json_name
    ok = cv2.imwrite(str(image_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {image_path}")
    description = (
        f"Person ROI crop from {stream.video.stem}, view={stream.view_key}, person_id={person_id}, "
        f"frame_id={frame_id}, time={timestamp:.3f}s, roi_xyxy_full_frame={[x1, y1, x2, y2]}"
    )
    label = make_labelme_json(image_name, w, h, shapes, description)
    json_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
    writer.writerow(
        {
            "view": stream.view_key,
            "video": str(stream.video),
            "camera_id": stream.camera_id,
            "calib_path": "" if stream.calib_path is None else str(stream.calib_path),
            "frame_id": frame_id,
            "time_sec": f"{timestamp:.3f}",
            "person_id": person_id,
            "phone_count": len(shapes),
            "max_phone_conf": f"{max(p.conf for p in phones):.4f}",
            "roi_xyxy": [x1, y1, x2, y2],
            "crop_w": w,
            "crop_h": h,
            "image": str(image_path),
            "json": str(json_path),
        }
    )
    return True


def main() -> None:
    global core_ref
    args = parse_args()
    cv2.setNumThreads(max(0, args.cv_threads))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = load_core(args.core_script)
    core_ref = core
    streams = build_streams(args, core)
    pose_model = YOLO(str(args.pose_model), task="pose")
    phone_model = YOLO(str(args.phone_model), task="detect")
    phone_class_ids = parse_class_ids(args.phone_class_ids)
    phone_names = {} if phone_class_ids is not None else {int(k): str(v) for k, v in phone_model.names.items()}
    params = core.default_params()

    manifest_path = args.output_dir / "manifest.csv"
    summary_path = args.output_dir / "summary.json"
    fields = [
        "view",
        "video",
        "camera_id",
        "calib_path",
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
    timings = Counter()
    started = time.perf_counter()
    step = 0
    print("[START] parallel multiview person ROI prelabel", flush=True)
    for stream in streams:
        print(
            f"[STREAM] idx={stream.idx} view={stream.view_key} samples={len(stream.sample_indices)} "
            f"video={stream.video} calib={stream.calib_path or 'NONE'}",
            flush=True,
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        while not all(s.done for s in streams):
            active: list[tuple[Stream, int, Any]] = []
            read_t0 = time.perf_counter()
            for stream in streams:
                if stream.done:
                    continue
                frame_id, frame = read_next_sample(stream)
                if frame is not None:
                    active.append((stream, frame_id, frame))
            timings["read_sec"] += time.perf_counter() - read_t0
            if not active:
                break

            pose_t0 = time.perf_counter()
            pose_results = predict_batches(
                pose_model,
                [frame for _, _, frame in active],
                args.pose_batch_size,
                args.pose_imgsz,
                args.pose_conf,
                args.device,
                pad_to_batch=not args.dynamic_pose_batch,
            )
            timings["pose_sec"] += time.perf_counter() - pose_t0

            phone_refs: list[tuple[Stream, int, Any, Any]] = []
            roi_by_track: dict[tuple[int, int], Any] = {}
            frame_by_stream: dict[int, tuple[int, Any]] = {}
            for (stream, frame_id, frame), pose_result in zip(active, pose_results):
                frame_by_stream[stream.idx] = (frame_id, frame)
                people = core.extract_people(pose_result, args.kp_conf, stream.width, stream.height)
                stream.next_track_id = core.assign_person_track_ids(
                    people,
                    stream.person_states,
                    stream.next_track_id,
                    frame_id,
                    stream.state_window,
                )
                core.prune_person_tracks(stream.person_states, frame_id, stream.state_window)
                for person in people:
                    if person.track_id < 0:
                        continue
                    person_roi = core.expand_rect(person.box, params["person_expand_x"], params["person_expand_y"], stream.width, stream.height)
                    job = core.RoiJob(roi=person_roi, screen_id="", source="person_roi", person_id=person.track_id)
                    phone_refs.append((stream, frame_id, frame, job))
                    roi_by_track[(stream.idx, person.track_id)] = person_roi
                stream.stats["frames"] += 1
                stream.stats["pose_people"] += len(people)
                stream.stats["roi_jobs"] += len(people)

            phone_t0 = time.perf_counter()
            phones_by_track = detect_phones_batched(
                core,
                phone_model,
                phone_names,
                phone_class_ids,
                phone_refs,
                args.phone_conf,
                args.phone_imgsz,
                args.device,
                args.phone_batch_size,
                pad_to_batch=not args.dynamic_phone_batch,
            )
            timings["phone_sec"] += time.perf_counter() - phone_t0

            save_t0 = time.perf_counter()
            grouped_by_stream: dict[int, list[tuple[int, list[Any]]]] = defaultdict(list)
            for (stream_idx, track_id), phones in phones_by_track.items():
                grouped_by_stream[stream_idx].append((track_id, core.nms_phone_candidates(phones, iou_thresh=0.5)))
            for stream, frame_id, frame in active:
                timestamp = (frame_id - 1) / stream.native_fps if stream.native_fps else 0.0
                if timestamp + 1e-6 < stream.next_save_t:
                    continue
                while stream.next_save_t <= timestamp + 1e-6:
                    stream.next_save_t += 1.0 / max(args.save_fps, 1e-6)
                for track_id, phones in sorted(grouped_by_stream.get(stream.idx, [])):
                    if not phones:
                        continue
                    stream.stats["phone_tracks"] += 1
                    last_t = stream.last_save_by_track.get(track_id, -1e9)
                    if timestamp - last_t < args.per_id_seconds - 1e-6:
                        stream.stats["skipped_per_id_throttle"] += 1
                        continue
                    roi = roi_by_track.get((stream.idx, track_id))
                    if roi is None:
                        continue
                    if save_person_roi_crop(args, stream, frame_id, frame, roi, phones, track_id, timestamp, writer):
                        stream.last_save_by_track[track_id] = timestamp
                        stream.stats["saved_crops"] += 1
            timings["save_sec"] += time.perf_counter() - save_t0

            step += 1
            if step % 25 == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                total_samples = sum(s.stats["frames"] for s in streams)
                total_saved = sum(s.stats["saved_crops"] for s in streams)
                print(
                    f"[PROGRESS] step={step} total_samples={total_samples} saved={total_saved} aggregate_wall_fps={total_samples/elapsed:.2f}",
                    flush=True,
                )
                manifest_file.flush()

    elapsed = max(time.perf_counter() - started, 1e-9)
    summaries = []
    for stream in streams:
        if stream.cap is not None:
            stream.cap.release()
        if stream.ffmpeg_proc is not None:
            stream.ffmpeg_proc.terminate()
            try:
                stream.ffmpeg_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                stream.ffmpeg_proc.kill()
        summaries.append(
            {
                "idx": stream.idx,
                "view_key": stream.view_key,
                "camera_id": stream.camera_id,
                "video": str(stream.video),
                "calib_path": str(stream.calib_path) if stream.calib_path else "",
                "width": stream.width,
                "height": stream.height,
                "native_fps": stream.native_fps,
                "target_infer_fps": args.infer_fps,
                "total_frames": stream.total_frames,
                "sampled_frames": int(stream.stats["frames"]),
                "duration_sec": stream.total_frames / stream.native_fps if stream.native_fps else None,
                "stats": dict(stream.stats),
            }
        )
    summary = {
        "version": "parallel_multiview_person_roi_prelabel_v1",
        "created_at": time.strftime("%F %T"),
        "root": str(args.root),
        "output_dir": str(args.output_dir),
        "manifest_csv": str(manifest_path),
        "settings": vars(args) | {
            "pose_model": str(args.pose_model),
            "phone_model": str(args.phone_model),
            "core_script": str(args.core_script),
            "calib_dir": str(args.calib_dir),
        },
        "elapsed_wall_sec": elapsed,
        "total_sampled_frames": sum(s["sampled_frames"] for s in summaries),
        "aggregate_wall_fps": sum(s["sampled_frames"] for s in summaries) / elapsed,
        "total_saved_crops": sum(int(s["stats"].get("saved_crops", 0)) for s in summaries),
        "timings_sec": dict(timings),
        "streams": summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[DONE] summary=" + str(summary_path), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
