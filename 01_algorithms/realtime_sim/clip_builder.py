from __future__ import annotations

import json
import argparse
import subprocess
from pathlib import Path

from realtime_sim.segment_producer import write_json_atomic


def load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "source_manifest.json").read_text(encoding="utf-8"))


def source_video_for_stream(manifest: dict, stream_index: int) -> Path:
    for view in manifest["views"]:
        if int(view["stream_index"]) == stream_index:
            return Path(view["source_video"])
    raise KeyError(f"stream_index {stream_index} not in source_manifest")


def build_clip(source: Path, start: float, duration: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(tmp),
    ]
    subprocess.check_call(cmd)
    tmp.replace(target)


def build_h264_transcode_command(source: Path, target: Path) -> list[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(source),
        "-map", "0:v:0", "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(target),
    ]


def transcode_to_h264(source: Path, target: Path) -> None:
    tmp = target.with_suffix(".h264.tmp.mp4")
    subprocess.check_call(build_h264_transcode_command(source, tmp))
    tmp.replace(target)


def load_event_details(run_dir: Path) -> list[dict]:
    events: list[dict] = []
    for path in sorted((run_dir / "processed").glob("batch_*/videos/frame_events.jsonl")):
        batch_name = path.parts[-3]
        batch_json = run_dir / "incoming" / batch_name / "batch.json"
        batch_start = 0.0
        if batch_json.exists():
            batch_start = float(json.loads(batch_json.read_text(encoding="utf-8")).get("global_start_sec", 0.0))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                item["_global_time"] = batch_start + float(item.get("time_sec", 0.0))
                events.append(item)
    return events


def rect_points(rect: dict) -> tuple[int, int, int, int]:
    if isinstance(rect, list) and len(rect) >= 4:
        return (
            int(round(float(rect[0]))),
            int(round(float(rect[1]))),
            int(round(float(rect[2]))),
            int(round(float(rect[3]))),
        )
    return (
        int(round(float(rect.get("x1", rect.get("left", 0))))),
        int(round(float(rect.get("y1", rect.get("top", 0))))),
        int(round(float(rect.get("x2", rect.get("right", 0))))),
        int(round(float(rect.get("y2", rect.get("bottom", 0))))),
    )


def draw_label(cv2, frame, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 6, y0 + th + 6), color, -1)
    cv2.putText(frame, text, (x + 3, y0 + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_event_overlay(cv2, frame, event: dict) -> None:
    for screen in event.get("screens", []):
        pts = screen.get("screen_poly", [])
        if len(pts) >= 3:
            poly = []
            for p in pts:
                poly.append([int(round(p[0])), int(round(p[1]))])
            import numpy as np
            arr = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [arr], True, (255, 80, 0), 2)
    for person in event.get("persons", []):
        alarm = bool(person.get("alarm"))
        risk = bool(person.get("risk")) or bool(person.get("suspect")) or int(person.get("window_hits", 0) or 0) > 0
        if not (alarm or risk):
            continue
        x1, y1, x2, y2 = rect_points(person.get("roi", person.get("box", {})))
        color = (0, 0, 255) if alarm else (0, 165, 255)
        label = f"Person{person.get('track_id', person.get('person_index', ''))}"
        label += " ALARM" if alarm else f" RISK {float(person.get('risk_score', 0.0)):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if risk else 4)
        draw_label(cv2, frame, x1, y1, label, color)
    for phone in event.get("phones", []):
        if not (phone.get("accepted") or phone.get("alarm")):
            continue
        x1, y1, x2, y2 = rect_points(phone.get("box", {}))
        color = (0, 0, 255)
        label = f"PHONE {float(phone.get('confidence', 0.0)):.2f}"
        if phone.get("risk_score") is not None:
            label += f" R{float(phone.get('risk_score', 0.0)):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        draw_label(cv2, frame, x1, y1, label, color)


def build_overlay_clip(run_dir: Path, source: Path, start: float, duration: float, stream_index: int, target: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False
    if not hasattr(cv2, "VideoCapture") or not hasattr(cv2, "VideoWriter"):
        return False
    tmp = target.with_suffix(".overlay.tmp.mp4")
    try:
        return _build_overlay_clip_with_cv2(cv2, run_dir, source, start, duration, stream_index, target, tmp)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        return False


def _build_overlay_clip_with_cv2(cv2, run_dir: Path, source: Path, start: float, duration: float, stream_index: int, target: Path, tmp: Path) -> bool:
    events = [
        e for e in load_event_details(run_dir)
        if int(e.get("stream_index", -1)) == stream_index and start - 0.4 <= float(e.get("_global_time", 0.0)) <= start + duration + 0.4
    ]
    if not events:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10.0
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    frame_count = max(1, int(round(duration * fps)))
    for i in range(frame_count):
        t = start + i / fps
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        nearest = min(events, key=lambda e: abs(float(e.get("_global_time", 0.0)) - t))
        if abs(float(nearest.get("_global_time", 0.0)) - t) <= 0.35:
            draw_event_overlay(cv2, frame, nearest)
        writer.write(frame)
    cap.release()
    writer.release()
    transcode_to_h264(tmp, target)
    if tmp.exists():
        tmp.unlink()
    return target.exists() and target.stat().st_size > 0


def merge_clip_updates(processed_payload: dict, latest_payload: dict) -> dict:
    processed_by_id = {seg.get("id"): seg for seg in processed_payload.get("segments", [])}
    merged = {"segments": []}
    for latest_seg in latest_payload.get("segments", []):
        seg_id = latest_seg.get("id")
        processed = processed_by_id.get(seg_id)
        if processed and processed.get("clip_status") in {"ready", "failed"}:
            updated = dict(latest_seg)
            for key in ("clip", "clip_status", "clip_error"):
                if key in processed:
                    updated[key] = processed[key]
                elif key == "clip_error":
                    updated.pop(key, None)
            merged["segments"].append(updated)
        else:
            merged["segments"].append(latest_seg)
    return merged


def build_missing_clips(run_dir: Path) -> int:
    manifest = load_manifest(run_dir)
    segments_path = run_dir / "events" / "segments.json"
    if not segments_path.exists():
        return 0
    payload = json.loads(segments_path.read_text(encoding="utf-8"))
    changed = 0
    for seg in payload.get("segments", []):
        clip_rel = seg.get("clip") or f"clips/{seg['id']}.mp4"
        target = run_dir / "dashboard" / clip_rel
        if target.exists():
            seg["clip"] = clip_rel
            seg["clip_status"] = "ready"
            seg.pop("clip_error", None)
            continue
        try:
            if seg.get("boxed_video"):
                source = Path(seg["boxed_video"])
                start = float(seg.get("boxed_start", 0.0))
                duration = max(2.0, float(seg.get("boxed_end", start + 2.0)) - start)
                build_clip(source, start, duration, target)
            else:
                source = source_video_for_stream(manifest, int(seg["stream_index"]))
                start = float(seg["start"])
                duration = max(2.0, float(seg["end"]) - start)
                if not build_overlay_clip(run_dir, source, start, duration, int(seg["stream_index"]), target):
                    build_clip(source, start, duration, target)
            seg["clip"] = clip_rel
            seg["clip_status"] = "ready"
            seg.pop("clip_error", None)
        except Exception as exc:
            seg["clip_status"] = "failed"
            seg["clip_error"] = str(exc)
        changed += 1
    latest_payload = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else payload
    write_json_atomic(segments_path, merge_clip_updates(payload, latest_payload))
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    changed = build_missing_clips(run_dir)
    from realtime_sim.dashboard_live import write_dashboard_api
    write_dashboard_api(run_dir)
    print(f"clips_changed={changed}")


if __name__ == "__main__":
    main()
