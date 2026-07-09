#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


PROJECT_ROOT = Path("/media/boshi/Data/JianKong")
DEFAULT_RUN_DIR = PROJECT_ROOT / "06_training_runs/cpp_event_dashboard_20260703_third_from_end_7way_20260707_102041"
CV2_AVAILABLE = cv2 is not None and hasattr(cv2, "VideoCapture") and hasattr(cv2, "cvtColor")


VIEW_LABELS = {
    "dianqi1": "dianqi1",
    "dianqi2": "dianqi2",
    "jixie1": "jixie1",
    "jixie2": "jixie2",
    "ruanjian1": "ruanjian1",
    "ruanjian2": "ruanjian2",
    "zoulang": "zoulang",
}


def default_output_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "04_labelme_datasets/Person标注" / f"event_hardcases_person_roi_phone_{stamp}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_slug(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text.strip("._") or "item"


def view_key(path: str) -> str:
    text = path.lower()
    for key in VIEW_LABELS:
        if key in text:
            return VIEW_LABELS[key]
    if "corridor" in text:
        return "zoulang"
    if "software1" in text:
        return "ruanjian1"
    if "software2" in text:
        return "ruanjian2"
    if "mechanical1" in text:
        return "jixie1"
    if "mechanical2" in text:
        return "jixie2"
    return f"stream{abs(hash(path)) % 1000}"


def clamp_rect(box: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))]


def relative_phone_box(phone_box: list[float], roi: list[int], crop_w: int, crop_h: int) -> list[float] | None:
    rx1, ry1, _, _ = roi
    x1, y1, x2, y2 = [float(v) for v in phone_box]
    x1 = max(0.0, min(float(crop_w), x1 - rx1))
    y1 = max(0.0, min(float(crop_h), y1 - ry1))
    x2 = max(0.0, min(float(crop_w), x2 - rx1))
    y2 = max(0.0, min(float(crop_h), y2 - ry1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def labelme_shape(box: list[float], description: str) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    return {
        "label": "phone",
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": description,
        "shape_type": "rectangle",
        "flags": {},
    }


def labelme_doc(image_name: str, width: int, height: int, shapes: list[dict[str, Any]], description: str) -> dict[str, Any]:
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


def find_person(row: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    persons = row.get("persons") or []
    for person in persons:
        if int(person.get("track_id", -1)) == int(track_id):
            return person
    return None


def candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for phone_idx, phone in enumerate(row.get("phones") or []):
            if not phone.get("accepted"):
                continue
            track_id = int(phone.get("track_id", -1))
            person = find_person(row, track_id)
            if person is None:
                continue
            out.append(
                {
                    "row": row,
                    "phone": phone,
                    "person": person,
                    "phone_idx": phone_idx,
                    "stream_index": int(row["stream_index"]),
                    "track_id": track_id,
                    "time_sec": float(row["time_sec"]),
                    "risk_score": float(phone.get("risk_score", row.get("max_risk", 0.0))),
                    "confidence": float(phone.get("confidence", 0.0)),
                    "alarm": bool(phone.get("alarm") or person.get("alarm") or int(row.get("alarm_track_count", 0)) > 0),
                    "view": view_key(row.get("input_video") or row.get("output_video") or ""),
                }
            )
    return out


def select_balanced(candidates: list[dict[str, Any]], max_images: int, min_track_seconds: float) -> list[dict[str, Any]]:
    candidates = sorted(
        candidates,
        key=lambda c: (
            not c["alarm"],
            -c["risk_score"],
            -c["confidence"],
            c["stream_index"],
            c["time_sec"],
        ),
    )
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    last_by_track: dict[tuple[int, int], float] = {}
    for cand in candidates:
        key = (cand["stream_index"], cand["track_id"])
        last = last_by_track.get(key)
        if last is not None and abs(cand["time_sec"] - last) < min_track_seconds:
            continue
        last_by_track[key] = cand["time_sec"]
        buckets[cand["stream_index"]].append(cand)

    selected: list[dict[str, Any]] = []
    stream_ids = sorted(buckets)
    cursor = 0
    while len(selected) < max_images and stream_ids:
        sid = stream_ids[cursor % len(stream_ids)]
        if buckets[sid]:
            selected.append(buckets[sid].pop(0))
        else:
            stream_ids.remove(sid)
            cursor -= 1
        cursor += 1
    return selected


def crop_dhash(image: Image.Image | np.ndarray, hash_size: int = 8) -> int:
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        arr = np.asarray(image)
        if arr.size == 0:
            return 0
        pil_image = Image.fromarray(arr)
    if pil_image.size[0] == 0 or pil_image.size[1] == 0:
        return 0
    gray = pil_image.convert("L")
    resized_a = np.asarray(gray.resize((hash_size, hash_size), Image.Resampling.BILINEAR), dtype=np.uint8)
    resized_h = np.asarray(gray.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR), dtype=np.uint8)
    resized_v = np.asarray(gray.resize((hash_size, hash_size + 1), Image.Resampling.BILINEAR), dtype=np.uint8)
    mean_value = float(resized_a.mean())
    diff = np.concatenate(
        [
            (resized_a >= mean_value).flatten(),
            (resized_h[:, 1:] > resized_h[:, :-1]).flatten(),
            (resized_v[1:, :] > resized_v[:-1, :]).flatten(),
        ]
    )
    value = 0
    for bit in diff:
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(a: int, b: int) -> int:
    value = a ^ b
    if hasattr(value, "bit_count"):
        return int(value.bit_count())
    return bin(value).count("1")


def is_near_duplicate(image_hash: int, prior_hashes: list[int], threshold: int) -> bool:
    return any(hamming_distance(image_hash, prev) <= threshold for prev in prior_hashes)


def read_frame_ffmpeg(video_path: str, time_sec: float) -> Image.Image:
    seek = f"{max(0.0, time_sec):.3f}"
    commands = [
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", seek, "-i", video_path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path, "-ss", seek, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
    ]
    errors: list[str] = []
    for cmd in commands:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and proc.stdout:
            return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
        errors.append(proc.stderr.decode("utf-8", errors="replace").strip())
    raise RuntimeError(f"failed to read frame at {time_sec:.3f}s from {video_path}: {' | '.join(errors)}")


def read_frame_cv2(cap_cache: dict[str, Any], video_path: str, time_sec: float) -> Image.Image:
    if not CV2_AVAILABLE:
        raise RuntimeError("cv2 VideoCapture is not available")
    cap = cap_cache.get(video_path)
    if cap is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        cap_cache[video_path] = cap
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(time_sec * native_fps)))
        ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame at {time_sec:.3f}s from {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def read_frame(cap_cache: dict[str, Any], video_path: str, time_sec: float, reader: str) -> Image.Image:
    if reader == "cv2":
        return read_frame_cv2(cap_cache, video_path, time_sec)
    if reader == "ffmpeg":
        return read_frame_ffmpeg(video_path, time_sec)
    if CV2_AVAILABLE:
        try:
            return read_frame_cv2(cap_cache, video_path, time_sec)
        except Exception:
            pass
    return read_frame_ffmpeg(video_path, time_sec)


def write_sample(out_dir: Path, index: int, cand: dict[str, Any], frame) -> dict[str, Any] | None:
    row = cand["row"]
    person = cand["person"]
    phone = cand["phone"]
    w, h = frame.size
    roi = clamp_rect(person["roi"], w, h)
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    crop = frame.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size
    rel = relative_phone_box(phone["box"], roi, crop_w, crop_h)
    if rel is None:
        return None

    stem = safe_slug(
        f"{index:04d}_{cand['view']}_s{cand['stream_index']}_t{cand['time_sec']:.1f}"
        f"_trk{cand['track_id']}_r{cand['risk_score']:.2f}"
    )
    image_name = f"{stem}.jpg"
    json_name = f"{stem}.json"
    image_path = out_dir / image_name
    json_path = out_dir / json_name
    crop.save(image_path, "JPEG", quality=95)

    shape_desc = (
        f"prelabel=review conf={cand['confidence']:.4f} risk={cand['risk_score']:.4f} "
        f"alarm={int(cand['alarm'])} level={phone.get('level', '')} "
        f"track_id={cand['track_id']} t={cand['time_sec']:.3f}s screen={phone.get('screen_id', '')}"
    )
    description = (
        f"Event hardcase Person ROI crop; source={Path(row.get('input_video', '')).name}; "
        f"stream={cand['stream_index']} view={cand['view']} track_id={cand['track_id']} "
        f"time={cand['time_sec']:.3f}s frame_index={row.get('frame_index')} "
        f"roi_xyxy_full_frame={roi}; note=delete phone box if false positive"
    )
    doc = labelme_doc(image_name, crop_w, crop_h, [labelme_shape(rel, shape_desc)], description)
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "image": str(image_path),
        "json": str(json_path),
        "view": cand["view"],
        "stream_index": cand["stream_index"],
        "time_sec": cand["time_sec"],
        "track_id": cand["track_id"],
        "confidence": cand["confidence"],
        "risk_score": cand["risk_score"],
        "alarm": int(cand["alarm"]),
        "input_video": row.get("input_video", ""),
        "roi_xyxy": roi,
        "phone_xyxy_full": phone["box"],
        "phone_xyxy_roi": rel,
        "image_hash": crop_dhash(crop),
    }


def build_dataset(
    run_dir: Path,
    output_dir: Path,
    max_images: int,
    min_track_seconds: float,
    hash_threshold: int,
    view_hash_threshold: int,
    reader: str,
) -> dict[str, Any]:
    events_path = run_dir / "videos" / "frame_events.jsonl"
    rows = load_jsonl(events_path)
    candidates = candidate_rows(rows)
    selected = select_balanced(candidates, max_images=max_images * 4, min_track_seconds=min_track_seconds)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    hashes_by_track: dict[tuple[int, int], list[int]] = defaultdict(list)
    hashes_by_view: dict[str, list[int]] = defaultdict(list)
    skipped_duplicates = 0
    skipped_invalid = 0
    cap_cache: dict[str, Any] = {}
    for idx, cand in enumerate(selected, start=1):
        if len(manifest_rows) >= max_images:
            break
        video_path = cand["row"].get("input_video") or cand["row"].get("output_video")
        frame = read_frame(cap_cache, video_path, cand["time_sec"], reader)
        sample = write_sample(output_dir, len(manifest_rows) + 1, cand, frame)
        if sample is not None:
            image_hash = int(sample["image_hash"])
            track_key = (int(sample["stream_index"]), int(sample["track_id"]))
            if is_near_duplicate(image_hash, hashes_by_track[track_key], hash_threshold) or is_near_duplicate(
                image_hash, hashes_by_view[str(sample["view"])], view_hash_threshold
            ):
                Path(sample["image"]).unlink(missing_ok=True)
                Path(sample["json"]).unlink(missing_ok=True)
                skipped_duplicates += 1
                continue
            hashes_by_track[track_key].append(image_hash)
            hashes_by_view[str(sample["view"])].append(image_hash)
            manifest_rows.append(sample)
        else:
            skipped_invalid += 1
    for cap in cap_cache.values():
        if hasattr(cap, "release"):
            cap.release()

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "image",
            "json",
            "view",
            "stream_index",
            "time_sec",
            "track_id",
            "confidence",
            "risk_score",
            "alarm",
            "input_video",
            "roi_xyxy",
            "phone_xyxy_full",
            "phone_xyxy_roi",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "event_rows": len(rows),
        "accepted_candidates": len(candidates),
        "selected_candidates": len(selected),
        "written_samples": len(manifest_rows),
        "skipped_duplicates": skipped_duplicates,
        "skipped_invalid": skipped_invalid,
        "max_images": max_images,
        "min_track_seconds": min_track_seconds,
        "hash_threshold": hash_threshold,
        "view_hash_threshold": view_hash_threshold,
        "reader": reader,
        "cv2_available": CV2_AVAILABLE,
        "by_view": {},
        "alarm_samples": sum(int(r["alarm"]) for r in manifest_rows),
    }
    counts: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        counts[str(row["view"])] += 1
    summary["by_view"] = dict(sorted(counts.items()))
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def self_test() -> None:
    assert clamp_rect([1.2, 2.7, 10.1, 12.8], 20, 30) == [1, 2, 11, 13]
    assert clamp_rect([0, 0, 1, 20], 20, 30) is None
    rel = relative_phone_box([110, 220, 135, 250], [100, 200, 300, 500], 200, 300)
    assert rel == [10.0, 20.0, 35.0, 50.0], rel
    rows = [
        {
            "stream_index": 1,
            "time_sec": 1.0,
            "input_video": "demo_jixie1.mp4",
            "persons": [{"track_id": 7, "roi": [0, 0, 100, 100]}],
            "phones": [{"accepted": True, "track_id": 7, "confidence": 0.5, "risk_score": 0.7, "box": [10, 10, 20, 20]}],
        },
        {
            "stream_index": 1,
            "time_sec": 2.0,
            "input_video": "demo_jixie1.mp4",
            "persons": [{"track_id": 7, "roi": [0, 0, 100, 100]}],
            "phones": [{"accepted": True, "track_id": 7, "confidence": 0.6, "risk_score": 0.8, "box": [10, 10, 20, 20]}],
        },
    ]
    cands = candidate_rows(rows)
    assert len(cands) == 2
    selected = select_balanced(cands, max_images=10, min_track_seconds=3.0)
    assert len(selected) == 1, selected
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = 255
    same = image.copy()
    different = np.zeros((32, 32, 3), dtype=np.uint8)
    different[:16, :] = 255
    h1 = crop_dhash(image)
    h2 = crop_dhash(same)
    h3 = crop_dhash(different)
    assert hamming_distance(h1, h2) == 0
    assert is_near_duplicate(h2, [h1], threshold=2)
    assert not is_near_duplicate(h3, [h1], threshold=2)
    assert read_frame is not None
    print("[SELF_TEST] ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--min-track-seconds", type=float, default=15.0)
    parser.add_argument("--hash-threshold", type=int, default=12)
    parser.add_argument("--view-hash-threshold", type=int, default=6)
    parser.add_argument("--reader", choices=["auto", "cv2", "ffmpeg"], default="auto")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    output_dir = args.output_dir or default_output_dir()
    summary = build_dataset(
        args.run_dir,
        output_dir,
        args.max_images,
        args.min_track_seconds,
        args.hash_threshold,
        args.view_hash_threshold,
        args.reader,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
