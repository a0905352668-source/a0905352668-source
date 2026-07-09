from __future__ import annotations

import json
import subprocess
import time
import math
from datetime import datetime
from pathlib import Path

from realtime_sim.config import RuntimeConfig, ViewSpec, default_runtime_config


def create_run_dir(cfg: RuntimeConfig | None = None) -> Path:
    cfg = cfg or default_runtime_config()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.run_root / f"realtime_sim_{stamp}"
    for name in ["incoming", "mini_roots", "processed", "events", "dashboard", "logs"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_source_manifest(run_dir: Path, views: list[ViewSpec], segment_seconds: float) -> Path:
    payload = {
        "segment_seconds": segment_seconds,
        "views": [
            {
                "stream_index": v.stream_index,
                "view_key": v.view_key,
                "view_label": v.view_label,
                "source_folder": str(v.source_folder),
                "source_video": str(v.source_video),
                "calibration_file": str(v.calibration_file),
            }
            for v in views
        ],
    }
    path = run_dir / "source_manifest.json"
    write_json_atomic(path, payload)
    return path


def ffmpeg_cut(source: Path, start: float, duration: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    subprocess.check_call(cmd)
    tmp.replace(target)


def release_batch(run_dir: Path, views: list[ViewSpec], batch_index: int, start_sec: float, duration_sec: float) -> Path:
    batch_dir = run_dir / "incoming" / f"batch_{batch_index:06d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for view in views:
        target = batch_dir / f"stream_{view.stream_index}_{view.view_key}.mp4"
        ffmpeg_cut(view.source_video, start_sec, duration_sec, target)
        files.append(
            {
                "stream_index": view.stream_index,
                "view_key": view.view_key,
                "view_label": view.view_label,
                "source_video": str(view.source_video),
                "segment_file": str(target),
                "global_start_sec": start_sec,
                "duration_sec": duration_sec,
                "calibration_file": str(view.calibration_file),
            }
        )
    write_json_atomic(
        batch_dir / "batch.json",
        {
            "batch_index": batch_index,
            "global_start_sec": start_sec,
            "duration_sec": duration_sec,
            "files": files,
            "complete": True,
        },
    )
    return batch_dir


def produce_batches(run_dir: Path, views: list[ViewSpec], segment_seconds: float, max_batches: int, mode: str) -> None:
    for batch_index in range(max_batches):
        start_sec = batch_index * segment_seconds
        t0 = time.monotonic()
        release_batch(run_dir, views, batch_index, start_sec, segment_seconds)
        if mode == "realtime":
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, segment_seconds - elapsed))


def estimate_video_duration_sec(source: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def batch_count_for_views(views: list[ViewSpec], segment_seconds: float) -> int:
    durations = [estimate_video_duration_sec(v.source_video) for v in views]
    if not durations:
        return 0
    shortest = max(0.0, min(durations))
    return max(1, int(math.ceil(shortest / max(segment_seconds, 1e-6))))
