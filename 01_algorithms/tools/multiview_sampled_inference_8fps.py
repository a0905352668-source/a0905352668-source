#!/usr/bin/env python3
"""Run sampled 7-view JianKong inference with per-view calibration.

This wrapper keeps the fixed-screen person ROI phone logic in
predict_surveillance_phone_pose_fixed_screens.py, but samples each input at a
target FPS and batches active streams in one process.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def load_core(script_path: Path):
    spec = importlib.util.spec_from_file_location("jk_core", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load core script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-script", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--phone-model", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--infer-fps", type=float, default=8.0)
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
    parser.add_argument("--draw-mode", choices=["full", "compact"], default="compact")
    parser.add_argument("--max-sampled-frames", type=int, default=0)
    parser.add_argument("--write-debug-csv", action="store_true")
    parser.add_argument("--cv-threads", type=int, default=1)
    parser.add_argument("--reader", choices=["cv2", "ffmpeg"], default="cv2")
    parser.add_argument("--no-output-video", action="store_true")
    parser.add_argument("--skip-gray-check", action="store_true")
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
        candidates = [
            p
            for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}
        ]
        if not candidates:
            continue
        videos.append(max(candidates, key=lambda p: (p.stat().st_mtime, p.name)))
    return sorted(videos, key=lambda p: view_sort_key(view_key_from_path(p)))


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


def predict_batches(
    model: YOLO,
    images: list[Any],
    batch_size: int,
    imgsz: int,
    conf: float,
    device: str,
    pad_to_batch: bool,
) -> list[Any]:
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


def detect_phones_batched(
    core: Any,
    model: YOLO,
    names: dict[int, str],
    phone_class_ids: set[int] | None,
    refs: list[tuple[int, Any, Any]],
    conf: float,
    imgsz: int,
    device: str,
    batch_size: int,
    pad_to_batch: bool,
) -> dict[int, list[Any]]:
    crops = []
    crop_refs: list[tuple[int, Any, int, int]] = []
    for stream_idx, frame, job in refs:
        x1, y1, x2, y2 = job.roi.as_int()
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
        crop_refs.append((stream_idx, job, x1, y1))

    phones_by_stream: dict[int, list[Any]] = {i: [] for i in range(32)}
    if not crops:
        return phones_by_stream

    results = predict_batches(model, crops, batch_size, imgsz, conf, device, pad_to_batch)
    for (stream_idx, job, ox, oy), result in zip(crop_refs, results):
        if result.boxes is None:
            continue
        for raw in result.boxes:
            cls_id = int(raw.cls.item())
            pconf = float(raw.conf.item())
            is_phone = cls_id in phone_class_ids if phone_class_ids is not None else names.get(cls_id, str(cls_id)).lower() == "phone"
            if not is_phone or pconf < conf:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in raw.xyxy[0].tolist()]
            phones_by_stream.setdefault(stream_idx, []).append(
                core.PhoneCandidate(
                    box=core.Rect(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
                    conf=pconf,
                    source=job.source,
                    screen_id=job.screen_id,
                    roi=job.roi,
                    person_id=job.person_id,
                )
            )
    return phones_by_stream


@dataclass
class Stream:
    idx: int
    video: Path
    view_key: str
    calib_path: Path | None
    cap: Any
    ffmpeg_proc: Any
    frame_bytes: int
    writer: Any
    debug_writer: Any
    csv_file: Any
    width: int
    height: int
    native_fps: float
    total_frames: int
    sample_indices: list[int]
    camera_id: str = ""
    screens: list[Any] = field(default_factory=list)
    state_window: int = 30
    state_min_hits: int = 16
    state_risk_threshold: float = 0.65
    person_states: dict[int, Any] = field(default_factory=dict)
    next_track_id: int = 1
    previous_alarm_tracks: set[int] = field(default_factory=set)
    alert_counter: float = 0.0
    last_good: Any = None
    stats: Counter = field(default_factory=Counter)
    done: bool = False
    sample_cursor: int = 0
    raw_cursor: int = 0


DEBUG_FIELDS = [
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


def build_streams(args: argparse.Namespace, core: Any) -> list[Stream]:
    streams: list[Stream] = []
    videos = find_last_videos(args.root)
    if not videos:
        raise RuntimeError(f"No videos found under {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, video in enumerate(videos):
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

        out_stem = f"{idx:02d}_{view_key}_{video.stem}_8fps"
        out_video = args.output_dir / f"{out_stem}.mp4"
        writer = None
        if not args.no_output_video:
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), args.infer_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to create output video: {out_video}")
        if args.write_debug_csv:
            debug_csv = args.output_dir / f"{out_stem}.debug.csv"
            csv_file = debug_csv.open("w", newline="", encoding="utf-8")
            debug_writer = csv.DictWriter(csv_file, fieldnames=DEBUG_FIELDS)
            debug_writer.writeheader()
        else:
            csv_file = None
            debug_writer = None

        ffmpeg_proc = None
        if args.reader == "ffmpeg":
            cap.release()
            cap = None
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
                    f"fps={args.infer_fps:.6f}",
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
            writer=writer,
            debug_writer=debug_writer,
            csv_file=csv_file,
            width=width,
            height=height,
            native_fps=native_fps,
            total_frames=total_frames,
            sample_indices=sample_indices,
        )
        if calib_path is not None:
            stream.camera_id, stream.screens = core.load_calibration(calib_path, width, height)
            stream.state_window = max(int(s.params.get("person_state_window", 30)) for s in stream.screens) if stream.screens else 30
            stream.state_min_hits = max(int(s.params.get("person_state_min_hits", 16)) for s in stream.screens) if stream.screens else 16
            stream.state_min_hits = max(1, min(stream.state_min_hits, stream.state_window))
            stream.state_risk_threshold = (
                max(float(s.params.get("person_state_risk_threshold", 0.65)) for s in stream.screens)
                if stream.screens
                else 0.65
            )
        streams.append(stream)
    return streams


def read_next_sample(args: argparse.Namespace, stream: Stream):
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
        if not args.skip_gray_check and core_ref.is_decode_gray(frame) and stream.last_good is not None:
            frame = stream.last_good.copy()
            stream.stats["gray_frames_replaced"] += 1
        elif not args.skip_gray_check:
            stream.last_good = frame.copy()
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
    if not args.skip_gray_check and core_ref.is_decode_gray(frame) and stream.last_good is not None:
        frame = stream.last_good.copy()
        stream.stats["gray_frames_replaced"] += 1
    elif not args.skip_gray_check:
        stream.last_good = frame.copy()
    return target + 1, frame


core_ref = None


def main() -> None:
    global core_ref
    args = parse_args()
    cv2.setNumThreads(max(0, args.cv_threads))
    core = load_core(args.core_script)
    core_ref = core

    streams = build_streams(args, core)
    pose_model = YOLO(str(args.pose_model), task="pose")
    phone_model = YOLO(str(args.phone_model), task="detect")
    phone_class_ids = parse_class_ids(args.phone_class_ids)
    phone_names = {} if phone_class_ids is not None else {int(k): str(v) for k, v in phone_model.names.items()}
    params = core.default_params()

    started = time.perf_counter()
    timings = Counter()
    step = 0
    print("[START] sampled multiview inference", flush=True)
    for stream in streams:
        print(
            f"[STREAM] idx={stream.idx} view={stream.view_key} video={stream.video} "
            f"native_fps={stream.native_fps:.3f} frames={stream.total_frames} samples={len(stream.sample_indices)} "
            f"calib={stream.calib_path or 'NONE'}",
            flush=True,
        )

    while not all(s.done for s in streams):
        active: list[tuple[Stream, int, Any]] = []
        read_t0 = time.perf_counter()
        for stream in streams:
            if stream.done:
                continue
            frame_id, frame = read_next_sample(args, stream)
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

        per_stream_people: dict[int, list[Any]] = {}
        phone_refs: list[tuple[int, Any, Any]] = []
        for (stream, frame_id, frame), pose_result in zip(active, pose_results):
            people = core.extract_people(pose_result, args.kp_conf, stream.width, stream.height)
            stream.next_track_id = core.assign_person_track_ids(
                people,
                stream.person_states,
                stream.next_track_id,
                frame_id,
                stream.state_window,
            )
            per_stream_people[stream.idx] = people
            for person in people:
                person_roi = core.expand_rect(person.box, params["person_expand_x"], params["person_expand_y"], stream.width, stream.height)
                phone_refs.append((stream.idx, frame, core.RoiJob(roi=person_roi, screen_id="", source="person_roi", person_id=person.index)))
            stream.stats["frames"] += 1
            stream.stats["pose_people"] += len(people)

        phone_t0 = time.perf_counter()
        phones_by_stream = detect_phones_batched(
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

        eval_t0 = time.perf_counter()
        for stream, frame_id, frame in active:
            people = per_stream_people.get(stream.idx, [])
            phones = core.nms_phone_candidates(phones_by_stream.get(stream.idx, []), iou_thresh=0.5)
            evals = []
            for phone in phones:
                if not stream.screens:
                    continue
                candidate_screens = [s for s in stream.screens if s.screen_id == phone.screen_id] if phone.screen_id else stream.screens
                phone_evals = []
                for screen in candidate_screens:
                    ev = core.evaluate_phone(phone, screen, people, stream.width, stream.height)
                    ev.effective_search_roi = phone.roi if phone.source in ("person_roi", "person_screen_roi") else None
                    phone_evals.append(ev)
                accepted = [e for e in phone_evals if e.accepted]
                if accepted:
                    evals.append(max(accepted, key=lambda e: (core.candidate_delta(e.level), e.risk_score)))
                elif phone_evals:
                    evals.append(max(phone_evals, key=lambda e: e.risk_score))

            core.apply_temporal_scores(evals, stream.person_states, stream.state_risk_threshold)
            accepted = [e for e in evals if e.accepted]
            core.update_person_states(
                people,
                evals,
                stream.person_states,
                frame_id,
                stream.state_window,
                stream.state_risk_threshold,
                stream.state_min_hits,
            )
            core.sync_eval_state_fields(evals, stream.person_states)
            visible_track_ids = {p.track_id for p in people if p.track_id >= 0}
            active_alarm_tracks = {
                tid
                for tid in visible_track_ids
                if tid in stream.person_states and stream.person_states[tid].alarm_triggered
            }
            new_alarm_tracks = active_alarm_tracks - stream.previous_alarm_tracks
            stream.alert_counter = float(
                max(
                    (stream.person_states[tid].window_hits for tid in visible_track_ids if tid in stream.person_states),
                    default=0,
                )
            )
            stable_alert = bool(active_alarm_tracks)
            core.prune_person_tracks(stream.person_states, frame_id, stream.state_window)

            if stream.debug_writer is not None:
                for ev in evals:
                    core.write_debug_row(stream.debug_writer, frame_id, ev, stream.alert_counter, stable_alert, ev.effective_search_roi)

            if not args.no_output_video and stream.writer is not None:
                annotated = frame.copy()
                core.draw_outputs(
                    annotated,
                    stream.screens,
                    people,
                    evals,
                    stream.person_states,
                    stream.alert_counter,
                    stable_alert,
                    draw_static_zones=False,
                    draw_mode=args.draw_mode,
                )
                if not stream.screens:
                    for phone in phones:
                        core.draw_rect(annotated, phone.box, (180, 0, 180), 1, f"PHONE:{phone.conf:.2f}")
                stream.writer.write(annotated)

            stream.stats["phone_candidates_raw"] += len(phones_by_stream.get(stream.idx, []))
            stream.stats["phone_candidates_after_nms"] += len(phones)
            stream.stats["accepted_candidate_frames"] += int(bool(accepted))
            stream.stats["accepted_candidates"] += len(accepted)
            stream.stats["strong_candidates"] += sum(1 for e in accepted if e.level == "strong")
            stream.stats["normal_candidates"] += sum(1 for e in accepted if e.level == "normal")
            stream.stats["weak_candidates"] += sum(1 for e in accepted if e.level == "weak")
            stream.stats["stable_alert_frames"] += int(stable_alert)
            stream.stats["person_alarm_events"] += len(new_alarm_tracks)
            stream.stats["max_alert_counter"] = max(stream.stats["max_alert_counter"], stream.alert_counter)
            stream.stats["max_person_window_hits"] = max(stream.stats["max_person_window_hits"], int(stream.alert_counter))
            stream.stats["max_risk_score"] = max(stream.stats["max_risk_score"], max((e.risk_score for e in evals), default=0.0))
            stream.stats["tracked_persons"] = max(stream.stats["tracked_persons"], len(stream.person_states))
            stream.previous_alarm_tracks = active_alarm_tracks
        timings["eval_draw_write_sec"] += time.perf_counter() - eval_t0

        step += 1
        if step % 25 == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            total_samples = sum(s.stats["frames"] for s in streams)
            print(
                f"[PROGRESS] step={step} total_samples={total_samples} aggregate_wall_fps={total_samples/elapsed:.2f}",
                flush=True,
            )

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
        if stream.writer is not None:
            stream.writer.release()
        if stream.csv_file is not None:
            stream.csv_file.close()
        out_stem = f"{stream.idx:02d}_{stream.view_key}_{stream.video.stem}_8fps"
        out_video = args.output_dir / f"{out_stem}.mp4"
        debug_csv = args.output_dir / f"{out_stem}.debug.csv" if args.write_debug_csv else None
        summary = {
            "idx": stream.idx,
            "view_key": stream.view_key,
            "camera_id": stream.camera_id,
            "video": str(stream.video),
            "calib_path": str(stream.calib_path) if stream.calib_path else "",
            "output_video": "" if args.no_output_video else str(out_video),
            "debug_csv": str(debug_csv) if debug_csv else "",
            "width": stream.width,
            "height": stream.height,
            "native_fps": stream.native_fps,
            "target_infer_fps": args.infer_fps,
            "total_frames": stream.total_frames,
            "sampled_frames": int(stream.stats["frames"]),
            "duration_sec": stream.total_frames / stream.native_fps if stream.native_fps else None,
            "stats": dict(stream.stats),
        }
        (args.output_dir / f"{out_stem}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append(summary)

    aggregate = {
        "version": "multiview_sampled_inference_8fps_v1",
        "root": str(args.root),
        "output_dir": str(args.output_dir),
        "settings": vars(args) | {
            "pose_model": str(args.pose_model),
            "phone_model": str(args.phone_model),
            "core_script": str(args.core_script),
            "calib_dir": str(args.calib_dir),
        },
        "elapsed_wall_sec": elapsed,
        "total_sampled_frames": sum(s["sampled_frames"] for s in summaries),
        "aggregate_wall_fps": sum(s["sampled_frames"] for s in summaries) / elapsed,
        "meets_7x8fps": (sum(s["sampled_frames"] for s in summaries) / elapsed) >= 56.0,
        "timings_sec": dict(timings),
        "streams": summaries,
    }
    summary_path = args.output_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[DONE] aggregate_summary=" + str(summary_path), flush=True)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
