from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from realtime_sim.segment_producer import write_json_atomic

SCHEMA_VERSION = "jk-runtime-metrics-v1"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_number(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _model_id(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("v20"):
        return parent
    return path.name


def build_batch_metrics(
    *,
    batch: dict,
    elapsed_sec: float,
    worker_summary: dict,
    state: dict,
    infer_fps: float,
    phone_conf: float,
    pose_plan: Path,
    phone_plan: Path,
    no_video: bool,
    clip_build_mode: str,
) -> dict[str, Any]:
    cpp = worker_summary.get("cpp_summary", {}) if isinstance(worker_summary, dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": round(time.time(), 3),
        "batch_index": _int_number(batch.get("batch_index"), -1),
        "batch_name": str(batch.get("batch_name") or batch.get("name") or ""),
        "duration_sec": round(_number(batch.get("duration_sec"), 0.0), 3),
        "elapsed_sec": round(float(elapsed_sec), 3),
        "infer_fps": round(float(infer_fps), 3),
        "phone_conf": round(float(phone_conf), 3),
        "aggregate_fps": round(_number(cpp.get("aggregate_fps"), 0.0), 3),
        "frames": _int_number(cpp.get("frames"), 0),
        "persons": _int_number(cpp.get("persons"), 0),
        "accepted": _int_number(cpp.get("accepted"), 0),
        "alarm_frames": _int_number(cpp.get("alarm_frames"), 0),
        "event_count": _int_number(state.get("event_count"), 0),
        "segment_count": _int_number(state.get("segment_count"), 0),
        "alarm_event_count": _int_number(state.get("alarm_event_count"), 0),
        "realtime_path": "metadata_only" if no_video else "boxed_video",
        "clip_build_mode": clip_build_mode,
        "pose_model_id": _model_id(Path(pose_plan)),
        "phone_model_id": _model_id(Path(phone_plan)),
        "pose_plan": str(pose_plan),
        "phone_plan": str(phone_plan),
    }


def append_batch_metrics(
    *,
    run_dir: Path,
    batch: dict,
    elapsed_sec: float,
    worker_summary: dict,
    state: dict,
    infer_fps: float,
    phone_conf: float,
    pose_plan: Path,
    phone_plan: Path,
    no_video: bool,
    clip_build_mode: str,
) -> dict[str, Any]:
    metric = build_batch_metrics(
        batch=batch,
        elapsed_sec=elapsed_sec,
        worker_summary=worker_summary,
        state=state,
        infer_fps=infer_fps,
        phone_conf=phone_conf,
        pose_plan=pose_plan,
        phone_plan=phone_plan,
        no_video=no_video,
        clip_build_mode=clip_build_mode,
    )
    events_dir = run_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    with (events_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metric, ensure_ascii=False, sort_keys=True) + "\n")
    write_json_atomic(events_dir / "metrics_latest.json", metric)
    return metric
