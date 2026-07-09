from __future__ import annotations

import csv
import json
from pathlib import Path

from realtime_sim.segment_producer import write_json_atomic

RISK_KEYS = ("rule_risk", "risk", "risk_score", "max_risk")


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "alarm"}


def parse_event_rows(processed_dir: Path, batch: dict) -> list[dict]:
    jsonl_rows = parse_jsonl_event_rows(processed_dir, batch)
    if jsonl_rows:
        return jsonl_rows
    rows: list[dict] = []
    for csv_path in sorted(processed_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            for row in reader:
                if not is_interesting_row(row):
                    continue
                rows.append(normalize_row(row, batch))
    return rows


def parse_jsonl_event_rows(processed_dir: Path, batch: dict) -> list[dict]:
    rows: list[dict] = []
    for jsonl_path in sorted(processed_dir.rglob("frame_events.jsonl")):
        with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if is_interesting_json_event(item):
                    rows.append(normalize_json_event(item, batch))
    return rows


def is_interesting_json_event(item: dict) -> bool:
    if int(item.get("accepted_count", 0) or 0) > 0:
        return True
    if int(item.get("alarm_track_count", 0) or 0) > 0:
        return True
    if float(item.get("max_risk", 0.0) or 0.0) >= 0.55:
        return True
    for person in item.get("persons", []):
        if person.get("suspect") or person.get("risk") or person.get("alarm"):
            return True
        if float(person.get("risk_score", 0.0) or 0.0) >= 0.55:
            return True
    for phone in item.get("phones", []):
        if phone.get("accepted") or phone.get("alarm"):
            return True
    return False


def normalize_json_event(item: dict, batch: dict) -> dict:
    stream_index = int(item.get("stream_index", 0) or 0)
    frame_index = int(item.get("frame_index", item.get("frame_id", 0)) or 0)
    batch_start = float(batch.get("global_start_sec", 0.0))
    batch_index = int(batch.get("batch_index", -1) or -1)
    batch_name = str(batch.get("batch_name") or batch.get("_batch_name") or batch.get("name") or (f"batch_{batch_index:06d}" if batch_index >= 0 else ""))
    local_time = float(item.get("time_sec", 0.0) or 0.0)
    risk = float(item.get("max_risk", 0.0) or 0.0)
    best_track = -1
    best_track_risk = -1.0
    alarm = int(item.get("alarm_track_count", 0) or 0) > 0
    for person in item.get("persons", []):
        person_risk = float(person.get("risk_score", 0.0) or 0.0)
        risk = max(risk, person_risk)
        if person.get("alarm"):
            alarm = True
        if person.get("suspect") or person.get("risk") or person.get("alarm") or person_risk > best_track_risk:
            if person_risk >= best_track_risk:
                best_track = int(person.get("track_id", -1) or -1)
                best_track_risk = person_risk
    for phone in item.get("phones", []):
        phone_risk = float(phone.get("risk_score", 0.0) or 0.0)
        risk = max(risk, phone_risk)
        if phone.get("alarm"):
            alarm = True
        if phone.get("accepted") and phone_risk >= best_track_risk:
            best_track = int(phone.get("track_id", -1) or -1)
            best_track_risk = phone_risk
    level = "alarm" if alarm or risk >= 0.8 else "risk"
    return {
        "stream_index": stream_index,
        "frame_index": frame_index,
        "local_time": round(local_time, 3),
        "timestamp": round(batch_start + local_time, 3),
        "level": level,
        "risk": round(risk, 4),
        "track_id": best_track,
        "batch_index": batch_index,
        "source_batch": batch_name,
        "output_video": item.get("output_video", ""),
        "raw": item,
    }


def is_interesting_row(row: dict) -> bool:
    text = json.dumps(row, ensure_ascii=False).lower()
    if any(word in text for word in ("alarm", "phone", "拍摄", "warning")):
        return True
    for key in RISK_KEYS:
        try:
            if float(row.get(key, 0) or 0) >= 0.55:
                return True
        except ValueError:
            pass
    return False


def normalize_row(row: dict, batch: dict) -> dict:
    stream_index = int(float(first_present(row, ["stream_index", "stream", "video_index", "view_index"], 0)))
    frame_index = int(float(first_present(row, ["frame_index", "frame", "sample_index"], 0)))
    fps = float(first_present(row, ["infer_fps"], 10.0) or 10.0)
    batch_start = float(batch.get("global_start_sec", 0.0))
    batch_index = int(batch.get("batch_index", -1) or -1)
    batch_name = str(batch.get("batch_name") or batch.get("_batch_name") or batch.get("name") or (f"batch_{batch_index:06d}" if batch_index >= 0 else ""))
    local_time = float(first_present(row, ["time_sec", "time", "local_time"], frame_index / max(fps, 1e-6)))
    timestamp = batch_start + local_time
    risk = 0.0
    for key in RISK_KEYS:
        try:
            risk = max(risk, float(row.get(key, 0) or 0))
        except ValueError:
            pass
    alarm = parse_bool(first_present(row, ["alarm", "is_alarm", "rule_hit"], "0"))
    level = "alarm" if alarm or risk >= 0.8 else "risk"
    return {
        "stream_index": stream_index,
        "frame_index": frame_index,
        "local_time": round(local_time, 3),
        "timestamp": round(timestamp, 3),
        "level": level,
        "risk": round(risk, 4),
        "batch_index": batch_index,
        "source_batch": batch_name,
        "output_video": first_present(row, ["output_video", "boxed_video"], ""),
        "raw": row,
    }


def first_present(row: dict, keys: list[str], default):
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
        if key.lower() in lowered and lowered[key.lower()] not in ("", None):
            return lowered[key.lower()]
    return default


def _append_source_batch(segment: dict, event: dict) -> None:
    source_batch = str(event.get("source_batch") or "")
    if not source_batch:
        return
    batches = segment.setdefault("source_batches", [])
    if source_batch not in batches:
        batches.append(source_batch)


def merge_segments(events: list[dict], gap_seconds: float = 4.0, pad_seconds: float = 3.0) -> list[dict]:
    sorted_events = sorted(events, key=lambda e: (int(e["stream_index"]), float(e["timestamp"])))
    segments: list[dict] = []
    for event in sorted_events:
        stream = int(event["stream_index"])
        t = float(event["timestamp"])
        same_boxed_video = segments and segments[-1].get("boxed_video") == event.get("output_video")
        if segments and segments[-1]["stream_index"] == stream and same_boxed_video and t <= segments[-1]["end"] + gap_seconds:
            seg = segments[-1]
            seg["end"] = max(seg["end"], t + pad_seconds)
            if seg.get("boxed_video") == event.get("output_video"):
                seg["boxed_end"] = max(seg.get("boxed_end", 0.0), float(event.get("local_time", 0.0)) + pad_seconds)
            seg["event_count"] += 1
            seg["max_risk"] = max(seg["max_risk"], float(event.get("risk", 0.0)))
            _append_source_batch(seg, event)
            if event["level"] == "alarm":
                seg["level"] = "alarm"
            continue
        local_time = float(event.get("local_time", 0.0))
        segment = {
            "id": f"event_{len(segments) + 1:06d}",
            "stream_index": stream,
            "start": max(0.0, t - pad_seconds),
            "end": t + pad_seconds,
            "boxed_video": event.get("output_video", ""),
            "boxed_start": max(0.0, local_time - pad_seconds),
            "boxed_end": local_time + pad_seconds,
            "level": event["level"],
            "event_count": 1,
            "max_risk": float(event.get("risk", 0.0)),
            "clip_status": "pending",
            "source_batches": [],
        }
        _append_source_batch(segment, event)
        segments.append(segment)
    return segments


def append_batch_events(run_dir: Path, batch_dir: Path, processed_dir: Path) -> dict:
    batch = dict(load_json(batch_dir / "batch.json", {}))
    batch.setdefault("batch_name", batch_dir.name)
    event_path = run_dir / "events" / "events.json"
    existing = load_json(event_path, {"events": []})
    new_events = parse_event_rows(processed_dir, batch)
    all_events = existing.get("events", []) + new_events
    segments = merge_segments(all_events)
    state = {
        "event_count": len(all_events),
        "segment_count": len(segments),
        "alarm_event_count": sum(1 for e in all_events if e.get("level") == "alarm"),
        "last_batch_index": batch.get("batch_index"),
    }
    write_json_atomic(event_path, {"events": all_events})
    write_json_atomic(run_dir / "events" / "segments.json", {"segments": segments})
    write_json_atomic(run_dir / "events" / "state.json", state)
    return state
