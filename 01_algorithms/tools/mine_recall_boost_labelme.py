#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


PROJECT_ROOT = Path("/media/boshi/Data/JianKong")
DEFAULT_RUN_DIR = PROJECT_ROOT / "06_training_runs/cpp_prelabel_source_20260703_pick2_7way_20260707_115415"
CV2_AVAILABLE = cv2 is not None and hasattr(cv2, "VideoCapture") and hasattr(cv2, "cvtColor")
DEFAULT_EXISTING_ROOT = PROJECT_ROOT / "04_labelme_datasets/Person标注"


def default_output_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "04_labelme_datasets/Person标注" / f"recall_boost_phone_missed_{stamp}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in "-_."):
            out.append(ch)
        else:
            out.append(f"u{ord(ch):04x}")
    return "".join(out).strip("._") or "sample"


def view_key(path: str) -> str:
    text = path.lower()
    if "dianqi1" in text or "electrical1" in text:
        return "dianqi1"
    if "dianqi2" in text or "electrical2" in text:
        return "dianqi2"
    if "jixie1" in text or "mechanical1" in text:
        return "jixie1"
    if "jixie2" in text or "mechanical2" in text:
        return "jixie2"
    if "ruanjian1" in text or "software1" in text:
        return "ruanjian1"
    if "ruanjian2" in text or "software2" in text:
        return "ruanjian2"
    if "zoulang" in text or "corridor" in text:
        return "zoulang"
    return "unknown"


def clamp_rect(box: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))]


def relative_box(phone_box: list[float], roi: list[int], crop_w: int, crop_h: int) -> list[float] | None:
    rx1, ry1, _, _ = roi
    x1, y1, x2, y2 = [float(v) for v in phone_box]
    x1 = max(0.0, min(float(crop_w), x1 - rx1))
    y1 = max(0.0, min(float(crop_h), y1 - ry1))
    x2 = max(0.0, min(float(crop_w), x2 - rx1))
    y2 = max(0.0, min(float(crop_h), y2 - ry1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def find_person(row: dict[str, Any], phone: dict[str, Any] | None = None, track_id: int | None = None) -> dict[str, Any] | None:
    persons = row.get("persons") or []
    if track_id is None and phone is not None:
        track_id = int(phone.get("track_id", -1))
    if track_id is not None and track_id >= 0:
        for person in persons:
            if int(person.get("track_id", -1)) == int(track_id):
                return person
    if phone is not None:
        person_index = int(phone.get("person_index", -1))
        for person in persons:
            if int(person.get("person_index", -2)) == person_index:
                return person
    return None


def image_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L")
    a = np.asarray(gray.resize((hash_size, hash_size), Image.Resampling.BILINEAR), dtype=np.uint8)
    h = np.asarray(gray.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR), dtype=np.uint8)
    v = np.asarray(gray.resize((hash_size, hash_size + 1), Image.Resampling.BILINEAR), dtype=np.uint8)
    bits = np.concatenate([(a >= float(a.mean())).flatten(), (h[:, 1:] > h[:, :-1]).flatten(), (v[1:, :] > v[:-1, :]).flatten()])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming(a: int, b: int) -> int:
    value = a ^ b
    return int(value.bit_count()) if hasattr(value, "bit_count") else bin(value).count("1")


def read_frame(cap_cache: dict[str, Any], video_path: str, time_sec: float) -> Image.Image:
    if not CV2_AVAILABLE:
        raise RuntimeError("This recall miner requires cv2 VideoCapture; run it with the yolo conda Python.")
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
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def labelme_shape(box: list[float], desc: str) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    return {
        "label": "phone",
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": desc,
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


def collect_low_conf(rows: list[dict[str, Any]], max_conf: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for phone_idx, phone in enumerate(row.get("phones") or []):
            conf = float(phone.get("confidence", 0.0))
            if conf > max_conf:
                continue
            person = find_person(row, phone)
            if person is None:
                continue
            out.append(
                {
                    "kind": "low_conf_candidate",
                    "row": row,
                    "phone": phone,
                    "phone_idx": phone_idx,
                    "person": person,
                    "stream_index": int(row["stream_index"]),
                    "track_id": int(person.get("track_id", phone.get("track_id", -1))),
                    "time_sec": float(row["time_sec"]),
                    "score": conf,
                    "view": view_key(row.get("input_video") or row.get("output_video") or ""),
                    "shape_mode": "phone_prelabel",
                }
            )
    return out


def collect_track_gaps(rows: list[dict[str, Any]], min_gap: float) -> list[dict[str, Any]]:
    by_track: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    accepted_times: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        row_tracks = {int(p.get("track_id", -1)) for p in row.get("persons") or [] if int(p.get("track_id", -1)) >= 0}
        for tid in row_tracks:
            by_track[(int(row["stream_index"]), tid)].append(row)
        for phone in row.get("phones") or []:
            if phone.get("accepted"):
                tid = int(phone.get("track_id", -1))
                if tid >= 0:
                    accepted_times[(int(row["stream_index"]), tid)].append(float(row["time_sec"]))

    out: list[dict[str, Any]] = []
    for key, times in accepted_times.items():
        if len(times) < 2:
            continue
        start_t, end_t = min(times), max(times)
        last_save = -1e9
        accepted_set = {round(t, 1) for t in times}
        for row in sorted(by_track.get(key, []), key=lambda r: float(r["time_sec"])):
            t = float(row["time_sec"])
            if t <= start_t or t >= end_t or t - last_save < min_gap:
                continue
            has_accepted_here = any(phone.get("accepted") and int(phone.get("track_id", -1)) == key[1] for phone in row.get("phones") or [])
            if has_accepted_here or round(t, 1) in accepted_set:
                continue
            person = find_person(row, track_id=key[1])
            if person is None:
                continue
            out.append(
                {
                    "kind": "track_gap_review",
                    "row": row,
                    "phone": None,
                    "phone_idx": -1,
                    "person": person,
                    "stream_index": key[0],
                    "track_id": key[1],
                    "time_sec": t,
                    "score": 0.0,
                    "view": view_key(row.get("input_video") or row.get("output_video") or ""),
                    "shape_mode": "empty_review",
                }
            )
            last_save = t
    return out


def select_candidates(candidates: list[dict[str, Any]], max_images: int, min_track_seconds: float) -> list[dict[str, Any]]:
    priority = {"low_conf_candidate": 0, "track_gap_review": 1}
    candidates = sorted(candidates, key=lambda c: (priority.get(c["kind"], 9), c["stream_index"], c["track_id"], c["time_sec"]))
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    last_by_track: dict[tuple[str, int, int], float] = {}
    for cand in candidates:
        tkey = (cand["kind"], cand["stream_index"], cand["track_id"])
        last = last_by_track.get(tkey)
        if last is not None and abs(cand["time_sec"] - last) < min_track_seconds:
            continue
        last_by_track[tkey] = cand["time_sec"]
        buckets[(cand["kind"], cand["stream_index"])].append(cand)

    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    cursor = 0
    while keys and len(selected) < max_images:
        key = keys[cursor % len(keys)]
        if buckets[key]:
            selected.append(buckets[key].pop(0))
        else:
            keys.remove(key)
            cursor -= 1
        cursor += 1
    return selected


def write_sample(out_dir: Path, index: int, cand: dict[str, Any], frame: Image.Image) -> dict[str, Any] | None:
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
    shapes: list[dict[str, Any]] = []
    rel_phone = None
    if phone is not None:
        rel_phone = relative_box(phone["box"], roi, crop_w, crop_h)
        if rel_phone is not None:
            shape_desc = (
                f"recall_boost=review kind={cand['kind']} conf={float(phone.get('confidence', 0.0)):.4f} "
                f"accepted={int(bool(phone.get('accepted')))} level={phone.get('level', '')} "
                f"track_id={cand['track_id']} t={cand['time_sec']:.3f}s note=keep/adjust if true phone, delete if false"
            )
            shapes.append(labelme_shape(rel_phone, shape_desc))

    stem = safe_slug(f"{index:04d}_{cand['kind']}_{cand['view']}_s{cand['stream_index']}_t{cand['time_sec']:.1f}_trk{cand['track_id']}")
    image_name = f"{stem}.jpg"
    json_name = f"{stem}.json"
    crop.save(out_dir / image_name, "JPEG", quality=95)
    desc = (
        f"Recall boost Person ROI crop; kind={cand['kind']}; source={Path(row.get('input_video', '')).name}; "
        f"stream={cand['stream_index']} view={cand['view']} track_id={cand['track_id']} time={cand['time_sec']:.3f}s "
        f"frame_index={row.get('frame_index')} roi_xyxy_full_frame={roi}; "
        f"instruction={'review prelabel box' if shapes else 'draw phone box if visible, leave empty if no phone'}"
    )
    (out_dir / json_name).write_text(json.dumps(labelme_doc(image_name, crop_w, crop_h, shapes, desc), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "image": str(out_dir / image_name),
        "json": str(out_dir / json_name),
        "kind": cand["kind"],
        "view": cand["view"],
        "stream_index": cand["stream_index"],
        "time_sec": cand["time_sec"],
        "track_id": cand["track_id"],
        "phone_conf": float(phone.get("confidence", 0.0)) if phone else "",
        "phone_accepted": int(bool(phone.get("accepted"))) if phone else "",
        "shape_count": len(shapes),
        "input_video": row.get("input_video", ""),
        "roi_xyxy": roi,
        "phone_xyxy_roi": rel_phone or "",
        "image_hash": image_hash(crop),
    }


def near_duplicate(h: int, prior: list[int], threshold: int) -> bool:
    return any(hamming(h, p) <= threshold for p in prior)


def create_contact_sheet(out_dir: Path, limit: int = 80) -> None:
    thumbs: list[Image.Image] = []
    for jpg in sorted(out_dir.glob("*.jpg"))[:limit]:
        if jpg.name.startswith("contact_sheet"):
            continue
        im = Image.open(jpg).convert("RGB")
        im.thumbnail((180, 140))
        canvas = Image.new("RGB", (180, 160), "white")
        canvas.paste(im, ((180 - im.width) // 2, 0))
        ImageDraw.Draw(canvas).text((4, 143), jpg.stem[:24], fill=(0, 0, 0))
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = 5
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 180, rows * 160), (245, 245, 245))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 180, (i // cols) * 160))
    sheet.save(out_dir / "contact_sheet_first80.jpg", quality=92)


def load_existing_hashes(existing_root: Path, output_dir: Path, max_images_per_dir: int = 500) -> tuple[list[int], dict[str, int]]:
    hashes: list[int] = []
    by_dir: dict[str, int] = {}
    if not existing_root.exists():
        return hashes, by_dir
    output_resolved = output_dir.resolve() if output_dir.exists() else output_dir
    for directory in sorted(p for p in existing_root.iterdir() if p.is_dir()):
        try:
            if directory.resolve() == output_resolved:
                continue
        except Exception:
            pass
        count = 0
        for jpg in sorted(directory.glob("*.jpg")):
            if jpg.name.startswith("contact_sheet"):
                continue
            try:
                with Image.open(jpg) as im:
                    hashes.append(image_hash(im.convert("RGB")))
                    count += 1
            except Exception:
                continue
            if count >= max_images_per_dir:
                break
        if count:
            by_dir[directory.name] = count
    return hashes, by_dir


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.run_dir / "videos" / "frame_events.jsonl")
    low_conf = collect_low_conf(rows, args.low_conf_max)
    gaps = collect_track_gaps(rows, args.gap_seconds)
    selected = select_candidates(low_conf + gaps, args.max_images * 4, args.min_track_seconds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_hashes, existing_hashes_by_dir = load_existing_hashes(
        args.existing_root,
        args.output_dir,
        max_images_per_dir=args.existing_max_images_per_dir,
    )

    manifest: list[dict[str, Any]] = []
    hashes_by_track: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    hashes_by_kind: dict[str, list[int]] = defaultdict(list)
    skipped_duplicates = 0
    skipped_existing_duplicates = 0
    skipped_invalid = 0
    cap_cache: dict[str, Any] = {}
    try:
        for cand in selected:
            if len(manifest) >= args.max_images:
                break
            video_path = cand["row"].get("input_video") or cand["row"].get("output_video")
            frame = read_frame(cap_cache, video_path, cand["time_sec"])
            sample = write_sample(args.output_dir, len(manifest) + 1, cand, frame)
            if sample is None:
                skipped_invalid += 1
                continue
            h = int(sample["image_hash"])
            if near_duplicate(h, existing_hashes, args.existing_hash_threshold):
                Path(sample["image"]).unlink(missing_ok=True)
                Path(sample["json"]).unlink(missing_ok=True)
                skipped_existing_duplicates += 1
                continue
            tkey = (str(sample["kind"]), int(sample["stream_index"]), int(sample["track_id"]))
            if near_duplicate(h, hashes_by_track[tkey], args.hash_threshold) or near_duplicate(h, hashes_by_kind[str(sample["kind"])], args.kind_hash_threshold):
                Path(sample["image"]).unlink(missing_ok=True)
                Path(sample["json"]).unlink(missing_ok=True)
                skipped_duplicates += 1
                continue
            hashes_by_track[tkey].append(h)
            hashes_by_kind[str(sample["kind"])].append(h)
            manifest.append(sample)
    finally:
        for cap in cap_cache.values():
            if hasattr(cap, "release"):
                cap.release()

    fields = ["image", "json", "kind", "view", "stream_index", "time_sec", "track_id", "phone_conf", "phone_accepted", "shape_count", "input_video", "roi_xyxy", "phone_xyxy_roi"]
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)

    by_kind: dict[str, int] = defaultdict(int)
    by_view: dict[str, int] = defaultdict(int)
    for row in manifest:
        by_kind[str(row["kind"])] += 1
        by_view[str(row["view"])] += 1
    summary = {
        "run_dir": str(args.run_dir),
        "output_dir": str(args.output_dir),
        "purpose": "recall_boost_raise_phone_detection",
        "event_rows": len(rows),
        "low_conf_candidates": len(low_conf),
        "track_gap_candidates": len(gaps),
        "selected_candidates": len(selected),
        "written_samples": len(manifest),
        "prelabel_box_samples": sum(1 for r in manifest if int(r["shape_count"]) > 0),
        "empty_review_samples": sum(1 for r in manifest if int(r["shape_count"]) == 0),
        "skipped_duplicates": skipped_duplicates,
        "skipped_existing_duplicates": skipped_existing_duplicates,
        "skipped_invalid": skipped_invalid,
        "existing_root": str(args.existing_root),
        "existing_hash_count": len(existing_hashes),
        "existing_hashes_by_dir": existing_hashes_by_dir,
        "existing_hash_threshold": args.existing_hash_threshold,
        "low_conf_max": args.low_conf_max,
        "min_track_seconds": args.min_track_seconds,
        "gap_seconds": args.gap_seconds,
        "by_kind": dict(sorted(by_kind.items())),
        "by_view": dict(sorted(by_view.items())),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    create_contact_sheet(args.output_dir)
    return summary


def self_test() -> None:
    assert clamp_rect([1.2, 2.2, 10.8, 20.1], 100, 100) == [1, 2, 11, 21]
    assert relative_box([15, 20, 30, 40], [10, 10, 50, 60], 40, 50) == [5.0, 10.0, 20.0, 30.0]
    img = Image.new("RGB", (32, 32), "white")
    h = image_hash(img)
    assert hamming(h, h) == 0
    rows = [
        {"stream_index": 0, "time_sec": 1.0, "input_video": "demo_jixie1.mp4", "persons": [{"track_id": 1, "person_index": 8, "roi": [0, 0, 100, 100]}], "phones": [{"accepted": False, "track_id": 1, "person_index": 8, "confidence": 0.3, "box": [10, 10, 20, 20]}]},
        {"stream_index": 0, "time_sec": 2.0, "input_video": "demo_jixie1.mp4", "persons": [{"track_id": 1, "person_index": 8, "roi": [0, 0, 100, 100]}], "phones": [{"accepted": True, "track_id": 1, "person_index": 8, "confidence": 0.6, "box": [10, 10, 20, 20]}]},
        {"stream_index": 0, "time_sec": 4.0, "input_video": "demo_jixie1.mp4", "persons": [{"track_id": 1, "person_index": 8, "roi": [0, 0, 100, 100]}], "phones": []},
        {"stream_index": 0, "time_sec": 6.0, "input_video": "demo_jixie1.mp4", "persons": [{"track_id": 1, "person_index": 8, "roi": [0, 0, 100, 100]}], "phones": [{"accepted": True, "track_id": 1, "person_index": 8, "confidence": 0.7, "box": [10, 10, 20, 20]}]},
    ]
    assert len(collect_low_conf(rows, 0.45)) == 1
    assert len(collect_track_gaps(rows, 1.0)) == 1
    print("[SELF_TEST] ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=220)
    parser.add_argument("--low-conf-max", type=float, default=0.45)
    parser.add_argument("--gap-seconds", type=float, default=2.0)
    parser.add_argument("--min-track-seconds", type=float, default=8.0)
    parser.add_argument("--hash-threshold", type=int, default=16)
    parser.add_argument("--kind-hash-threshold", type=int, default=8)
    parser.add_argument("--existing-root", type=Path, default=DEFAULT_EXISTING_ROOT)
    parser.add_argument("--existing-hash-threshold", type=int, default=10)
    parser.add_argument("--existing-max-images-per-dir", type=int, default=500)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    args.output_dir = args.output_dir or default_output_dir()
    summary = build_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
