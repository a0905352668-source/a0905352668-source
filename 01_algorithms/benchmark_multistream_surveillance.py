#!/usr/bin/env python3
"""Multistream benchmark for the JianKong surveillance pipeline.

This keeps the v2.2 person-ROI phone logic, but removes video drawing, mp4
encoding, and debug CSV writes so the result reflects online detection speed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import queue
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


def load_core(script_path: Path):
    spec = importlib.util.spec_from_file_location("jk_core", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load core script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class StreamState:
    stream_id: int
    video: Path
    cap: Any
    width: int
    height: int
    fps: float
    total_frames: int
    frame_id: int = 0
    measured_frames: int = 0
    person_states: dict[int, Any] = field(default_factory=dict)
    next_track_id: int = 1
    previous_alarm_tracks: set[int] = field(default_factory=set)
    alert_counter: float = 0.0
    stats: Counter = field(default_factory=Counter)
    cached_frames: list[Any] | None = None
    reader: Any = None


class ThreadedVideoReader:
    def __init__(self, video: Path, loop_input: bool, queue_size: int = 8):
        self.video = video
        self.loop_input = loop_input
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_size))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        cap = cv2.VideoCapture(str(self.video))
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if not self.loop_input:
                        self.queue.put(None)
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        self.queue.put(None)
                        break
                self.queue.put(frame)
        finally:
            cap.release()

    def read(self):
        return self.queue.get()

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-script", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--phone-model", type=Path, required=True)
    parser.add_argument("--screen-config", type=Path, required=True)
    parser.add_argument("--video", type=Path, nargs="+", required=True)
    parser.add_argument("--streams", type=int, default=7)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--pose-imgsz", type=int, default=960)
    parser.add_argument("--phone-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--kp-conf", type=float, default=0.35)
    parser.add_argument("--phone-candidate-conf", type=float, default=0.25)
    parser.add_argument("--pose-batch-size", type=int, default=7)
    parser.add_argument("--phone-batch-size", type=int, default=16)
    parser.add_argument("--dynamic-pose-batch", action="store_true")
    parser.add_argument("--dynamic-phone-batch", action="store_true")
    parser.add_argument("--screen-related-person-roi-only", action="store_true")
    parser.add_argument("--preload-frames", action="store_true")
    parser.add_argument("--threaded-read", action="store_true")
    parser.add_argument("--read-queue-size", type=int, default=8)
    parser.add_argument("--skip-phone", action="store_true")
    parser.add_argument("--phone-interval", type=int, default=1)
    parser.add_argument("--overlap-phone-prev-roi", action="store_true")
    parser.add_argument("--max-frames-per-stream", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cv-threads", type=int, default=1)
    parser.add_argument("--loop-input", action="store_true")
    return parser.parse_args()


def predict_exact_batches(
    model: YOLO,
    images: list[Any],
    batch_size: int,
    imgsz: int,
    conf: float,
    device: str,
    pad_to_batch: bool = True,
) -> list[Any]:
    if not images:
        return []
    batch_size = max(1, int(batch_size))
    out: list[Any] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        real_len = len(chunk)
        if pad_to_batch and real_len < batch_size:
            chunk = chunk + [chunk[-1]] * (batch_size - real_len)
        results = model.predict(chunk, imgsz=imgsz, conf=conf, device=device, rect=False, verbose=False)
        out.extend(results[:real_len])
    return out


def build_person_roi_jobs(
    core: Any,
    people: list[Any],
    width: int,
    height: int,
    screens: list[Any],
    screen_related_only: bool = False,
) -> list[Any]:
    params = core.default_params()
    jobs = []
    for person in people:
        if screen_related_only and not any(core.person_related_to_screen(person, screen, width, height) for screen in screens):
            continue
        person_roi = core.expand_rect(person.box, params["person_expand_x"], params["person_expand_y"], width, height)
        jobs.append(core.RoiJob(roi=person_roi, screen_id="", source="person_roi", person_id=person.index))
    return jobs


def detect_phones_batched(
    core: Any,
    model: YOLO,
    names: dict[int, str],
    frame_jobs: list[tuple[int, Any, Any]],
    conf: float,
    imgsz: int,
    device: str,
    batch_size: int,
    pad_to_batch: bool,
) -> dict[int, list[Any]]:
    crops = []
    refs: list[tuple[int, Any, int, int]] = []
    for stream_id, frame, job in frame_jobs:
        x1, y1, x2, y2 = job.roi.as_int()
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
        refs.append((stream_id, job, x1, y1))
    phones_by_stream: dict[int, list[Any]] = defaultdict(list)
    if not crops:
        return phones_by_stream
    results = predict_exact_batches(model, crops, batch_size, imgsz, conf, device, pad_to_batch=pad_to_batch)
    for (stream_id, job, ox, oy), result in zip(refs, results):
        if result.boxes is None:
            continue
        for raw in result.boxes:
            cls_id = int(raw.cls.item())
            name = names.get(cls_id, str(cls_id)).lower()
            pconf = float(raw.conf.item())
            if name != "phone" or pconf < conf:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in raw.xyxy[0].tolist()]
            phones_by_stream[stream_id].append(
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


def open_streams(video_paths: list[Path], stream_count: int) -> list[StreamState]:
    streams = []
    for idx in range(stream_count):
        video = video_paths[idx % len(video_paths)]
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        streams.append(StreamState(idx, video, cap, width, height, fps, total))
    return streams


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(max(0, args.cv_threads))
    core = load_core(args.core_script)

    streams = open_streams(args.video, args.streams)
    if args.preload_frames:
        needed = args.max_frames_per_stream + max(0, args.warmup_steps) + 2
        for stream in streams:
            frames = []
            while len(frames) < needed:
                ok, frame = stream.cap.read()
                if not ok:
                    if not args.loop_input:
                        break
                    stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = stream.cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                raise RuntimeError(f"Failed to preload frames for stream {stream.stream_id}: {stream.video}")
            stream.cached_frames = frames
            stream.cap.release()
    elif args.threaded_read:
        for stream in streams:
            if stream.cap is not None and stream.cap.isOpened():
                stream.cap.release()
            stream.reader = ThreadedVideoReader(stream.video, args.loop_input, args.read_queue_size)
    width = streams[0].width
    height = streams[0].height
    camera_id, screens = core.load_calibration(args.screen_config, width, height)
    state_window = max(int(s.params.get("person_state_window", 30)) for s in screens) if screens else 30
    state_window = max(1, state_window)
    state_min_hits = max(int(s.params.get("person_state_min_hits", 16)) for s in screens) if screens else 16
    state_min_hits = max(1, min(state_min_hits, state_window))
    state_risk_threshold = max(float(s.params.get("person_state_risk_threshold", 0.65)) for s in screens) if screens else 0.65

    pose_model = YOLO(str(args.pose_model), task="pose")
    phone_model = YOLO(str(args.phone_model), task="detect")
    phone_names = {int(k): str(v) for k, v in phone_model.names.items()}

    timings = Counter()
    measured_started = False
    measured_start = 0.0
    measured_steps = 0
    warmup_steps = max(0, args.warmup_steps)
    step = 0
    last_roi_jobs_by_stream: dict[int, list[Any]] = defaultdict(list)
    executor = ThreadPoolExecutor(max_workers=2) if args.overlap_phone_prev_roi else None

    while True:
        if all(s.measured_frames >= args.max_frames_per_stream for s in streams):
            break

        read_t0 = time.perf_counter()
        active: list[tuple[StreamState, Any]] = []
        for stream in streams:
            if stream.measured_frames >= args.max_frames_per_stream:
                continue
            if stream.cached_frames is not None:
                if stream.frame_id >= len(stream.cached_frames) and not args.loop_input:
                    continue
                frame = stream.cached_frames[stream.frame_id % len(stream.cached_frames)]
            elif stream.reader is not None:
                frame = stream.reader.read()
                if frame is None:
                    continue
            else:
                ok, frame = stream.cap.read()
                if not ok and args.loop_input:
                    stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = stream.cap.read()
                if not ok:
                    continue
            stream.frame_id += 1
            active.append((stream, frame))
        if not active:
            break

        measure = step >= warmup_steps
        if measure and not measured_started:
            measured_started = True
            measured_start = time.perf_counter()
        if measure:
            timings["read_sec"] += time.perf_counter() - read_t0

        phone_refs: list[tuple[int, Any, Any]] = []
        per_stream_people: dict[int, list[Any]] = {}
        if args.overlap_phone_prev_roi and executor is not None:
            for stream, frame in active:
                run_phone_this_frame = (max(1, args.phone_interval) <= 1) or ((stream.frame_id - 1) % max(1, args.phone_interval) == 0)
                if run_phone_this_frame:
                    for job in last_roi_jobs_by_stream.get(stream.stream_id, []):
                        phone_refs.append((stream.stream_id, frame, job))

            overlap_t0 = time.perf_counter()
            pose_future = executor.submit(
                predict_exact_batches,
                pose_model,
                [frame for _, frame in active],
                args.pose_batch_size,
                args.pose_imgsz,
                args.pose_conf,
                args.device,
                not args.dynamic_pose_batch,
            )
            if args.skip_phone:
                phone_future = None
            else:
                phone_future = executor.submit(
                    detect_phones_batched,
                    core,
                    phone_model,
                    phone_names,
                    phone_refs,
                    args.phone_candidate_conf,
                    args.phone_imgsz,
                    args.device,
                    args.phone_batch_size,
                    not args.dynamic_phone_batch,
                )
            pose_results = pose_future.result()
            pose_done = time.perf_counter()
            phones_by_stream = defaultdict(list) if phone_future is None else phone_future.result()
            overlap_done = time.perf_counter()
            if measure:
                timings["overlap_pose_phone_sec"] += overlap_done - overlap_t0
                timings["pose_wait_sec"] += pose_done - overlap_t0
                timings["phone_wait_after_pose_sec"] += overlap_done - pose_done

            current_roi_jobs_by_stream: dict[int, list[Any]] = defaultdict(list)
            for (stream, _frame), pose_result in zip(active, pose_results):
                people = core.extract_people(pose_result, args.kp_conf, stream.width, stream.height)
                stream.next_track_id = core.assign_person_track_ids(
                    people, stream.person_states, stream.next_track_id, stream.frame_id, state_window
                )
                per_stream_people[stream.stream_id] = people
                current_roi_jobs_by_stream[stream.stream_id] = build_person_roi_jobs(
                    core,
                    people,
                    stream.width,
                    stream.height,
                    screens,
                    args.screen_related_person_roi_only,
                )
                if measure:
                    stream.stats["frames"] += 1
                    stream.stats["pose_people"] += len(people)
                    stream.stats["roi_jobs"] += sum(1 for sid, _frame, _job in phone_refs if sid == stream.stream_id)
                    stream.measured_frames += 1
            last_roi_jobs_by_stream = current_roi_jobs_by_stream
        else:
            pose_t0 = time.perf_counter()
            pose_results = predict_exact_batches(
                pose_model,
                [frame for _, frame in active],
                args.pose_batch_size,
                args.pose_imgsz,
                args.pose_conf,
                args.device,
                pad_to_batch=not args.dynamic_pose_batch,
            )
            if measure:
                timings["pose_sec"] += time.perf_counter() - pose_t0

            for (stream, frame), pose_result in zip(active, pose_results):
                people = core.extract_people(pose_result, args.kp_conf, stream.width, stream.height)
                stream.next_track_id = core.assign_person_track_ids(
                    people, stream.person_states, stream.next_track_id, stream.frame_id, state_window
                )
                per_stream_people[stream.stream_id] = people
                run_phone_this_frame = (max(1, args.phone_interval) <= 1) or ((stream.frame_id - 1) % max(1, args.phone_interval) == 0)
                if run_phone_this_frame:
                    for job in build_person_roi_jobs(
                        core,
                        people,
                        stream.width,
                        stream.height,
                        screens,
                        args.screen_related_person_roi_only,
                    ):
                        phone_refs.append((stream.stream_id, frame, job))
                if measure:
                    stream.stats["frames"] += 1
                    stream.stats["pose_people"] += len(people)
                    stream.stats["roi_jobs"] += sum(1 for sid, _frame, _job in phone_refs if sid == stream.stream_id)
                    stream.measured_frames += 1

            phone_t0 = time.perf_counter()
            if args.skip_phone:
                phones_by_stream = defaultdict(list)
            else:
                phones_by_stream = detect_phones_batched(
                    core,
                    phone_model,
                    phone_names,
                    phone_refs,
                    args.phone_candidate_conf,
                    args.phone_imgsz,
                    args.device,
                    args.phone_batch_size,
                    pad_to_batch=not args.dynamic_phone_batch,
                )
            if measure:
                timings["phone_sec"] += time.perf_counter() - phone_t0

        eval_t0 = time.perf_counter()
        active_by_id = {stream.stream_id: (stream, frame) for stream, frame in active}
        for stream_id, (stream, _frame) in active_by_id.items():
            people = per_stream_people.get(stream_id, [])
            phones = core.nms_phone_candidates(phones_by_stream.get(stream_id, []), iou_thresh=0.5)
            evals = []
            for phone in phones:
                candidate_screens = [s for s in screens if s.screen_id == phone.screen_id] if phone.screen_id else screens
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

            core.apply_temporal_scores(evals, stream.person_states, state_risk_threshold)
            core.update_person_states(people, evals, stream.person_states, stream.frame_id, state_window, state_risk_threshold, state_min_hits)
            core.sync_eval_state_fields(evals, stream.person_states)
            visible_track_ids = {p.track_id for p in people if p.track_id >= 0}
            active_alarm_tracks = {tid for tid in visible_track_ids if tid in stream.person_states and stream.person_states[tid].alarm_triggered}
            stream.alert_counter = float(max((stream.person_states[tid].window_hits for tid in visible_track_ids if tid in stream.person_states), default=0))
            core.prune_person_tracks(stream.person_states, stream.frame_id, state_window)
            if measure:
                accepted_evals = [e for e in evals if e.accepted]
                stream.stats["phone_candidates_after_nms"] += len(phones)
                stream.stats["accepted_candidates"] += len(accepted_evals)
                stream.stats["stable_alert_frames"] += int(bool(active_alarm_tracks))
                stream.stats["tracked_persons_max"] = max(stream.stats["tracked_persons_max"], len(stream.person_states))
            stream.previous_alarm_tracks = active_alarm_tracks
        if measure:
            timings["eval_sec"] += time.perf_counter() - eval_t0
            measured_steps += 1

        step += 1
        if step % 50 == 0:
            measured = sum(s.measured_frames for s in streams)
            elapsed = max(1e-9, time.perf_counter() - measured_start) if measured_started else 0.0
            fps = measured / elapsed if elapsed else 0.0
            print(f"[step={step}] measured_frames={measured} aggregate_fps={fps:.2f}", flush=True)

    measured_wall = max(1e-9, time.perf_counter() - measured_start) if measured_started else 0.0
    total_measured = sum(s.measured_frames for s in streams)
    aggregate_fps = total_measured / measured_wall if measured_wall else 0.0
    per_stream_fps = aggregate_fps / max(1, args.streams)

    for stream in streams:
        if stream.cap is not None and stream.cap.isOpened():
            stream.cap.release()
        if stream.reader is not None:
            stream.reader.close()
    if executor is not None:
        executor.shutdown(wait=True)

    summary = {
        "camera_id": camera_id,
        "stream_count": args.streams,
        "videos": [str(p) for p in args.video],
        "pose_model": str(args.pose_model),
        "phone_model": str(args.phone_model),
        "screen_config": str(args.screen_config),
        "pose_imgsz": args.pose_imgsz,
        "phone_imgsz": args.phone_imgsz,
        "pose_batch_size": args.pose_batch_size,
        "phone_batch_size": args.phone_batch_size,
        "dynamic_pose_batch": args.dynamic_pose_batch,
        "dynamic_phone_batch": args.dynamic_phone_batch,
        "screen_related_person_roi_only": args.screen_related_person_roi_only,
        "preload_frames": args.preload_frames,
        "threaded_read": args.threaded_read,
        "read_queue_size": args.read_queue_size,
        "skip_phone": args.skip_phone,
        "phone_interval": max(1, args.phone_interval),
        "overlap_phone_prev_roi": args.overlap_phone_prev_roi,
        "warmup_steps": warmup_steps,
        "max_frames_per_stream": args.max_frames_per_stream,
        "measured_frames_total": total_measured,
        "measured_wall_sec": measured_wall,
        "aggregate_fps": aggregate_fps,
        "per_stream_fps": per_stream_fps,
        "meets_7x10fps": aggregate_fps >= 70.0,
        "meets_7x8fps": aggregate_fps >= 56.0,
        "timings_sec": dict(timings),
        "timing_share": {k: v / measured_wall for k, v in timings.items()} if measured_wall else {},
        "stream_stats": {
            str(s.stream_id): {
                "video": str(s.video),
                "frames": s.measured_frames,
                "stats": dict(s.stats),
            }
            for s in streams
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
