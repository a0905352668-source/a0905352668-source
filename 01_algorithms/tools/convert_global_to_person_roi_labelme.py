#!/usr/bin/env python3
"""Convert full-frame LabelMe phone annotations to Person ROI LabelMe crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


PHONE_LABELS = {"phone", "mobile", "cellphone", "cell phone", "手机"}
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--pose-imgsz", type=int, default=960)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expand-x", type=float, default=0.35)
    parser.add_argument("--expand-y", type=float, default=0.20)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--review", action="store_true")
    return parser.parse_args()


def clamp_box(box: list[float], w: int, h: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return [x1, y1, x2, y2]


def expand_box(box: list[float], x_ratio: float, y_ratio: float, w: int, h: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return clamp_box([x1 - bw * x_ratio, y1 - bh * y_ratio, x2 + bw * x_ratio, y2 + bh * y_ratio], w, h)


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def inter_area(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def point_in_box(pt: tuple[float, float], box: list[float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def shape_to_box(shape: dict, w: int, h: int) -> list[float] | None:
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return clamp_box([min(xs), min(ys), max(xs), max(ys)], w, h)


def find_image(json_path: Path, data: dict) -> Path | None:
    image_path = data.get("imagePath")
    if image_path:
        p = (json_path.parent / image_path).resolve()
        if p.exists():
            return p
    for ext in IMAGE_EXTS:
        p = json_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def safe_tag(path: Path, sources: list[Path]) -> str:
    for source in sources:
        try:
            rel = path.relative_to(source.parent)
            raw = "_".join(rel.parts)
            break
        except ValueError:
            continue
    else:
        raw = "_".join(path.parts[-3:])
    return re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", raw).strip("_") or "source"


def labelme_phone_boxes(data: dict, w: int, h: int) -> list[dict]:
    phones = []
    for idx, shape in enumerate(data.get("shapes", [])):
        label = str(shape.get("label", "")).strip().lower()
        if label not in PHONE_LABELS:
            continue
        box = shape_to_box(shape, w, h)
        if box is None:
            continue
        phones.append({"idx": idx, "box": box, "label": "phone"})
    return phones


def pose_person_boxes(model: YOLO, image, imgsz: int, conf: float, device: str) -> list[dict]:
    result = model.predict(image, imgsz=imgsz, conf=conf, device=device, rect=False, verbose=False)[0]
    people = []
    if result.boxes is None or result.boxes.xyxy is None:
        return people
    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes.conf is not None else [0.0] * len(boxes)
    for i, (box, pconf) in enumerate(zip(boxes, confs)):
        people.append({"idx": i, "box": [float(v) for v in box], "conf": float(pconf)})
    return people


def make_labelme_json(image_name: str, crop_w: int, crop_h: int, shapes: list[dict], description: str) -> dict:
    return {
        "version": "2.4.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": crop_h,
        "imageWidth": crop_w,
        "description": description,
    }


def main() -> None:
    args = parse_args()
    if args.clear_output and args.output.exists():
        # Only remove the explicit generated output directory.
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    review_dir = args.output / "_review"
    if args.review:
        review_dir.mkdir(parents=True, exist_ok=True)

    pose_model = YOLO(str(args.pose_model), task="pose")
    manifest = []
    stats = Counter()

    json_paths: list[Path] = []
    for source in args.source:
        json_paths.extend(sorted(source.rglob("*.json")))
    stats["input_json"] = len(json_paths)

    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            stats["bad_json"] += 1
            continue
        if "shapes" not in data:
            stats["non_labelme_json"] += 1
            continue
        image_path = find_image(json_path, data)
        if image_path is None:
            stats["missing_image"] += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            stats["unreadable_image"] += 1
            continue
        h, w = image.shape[:2]
        phones = labelme_phone_boxes(data, w, h)
        if not phones:
            stats["json_without_phone"] += 1
            continue

        stats["phone_boxes"] += len(phones)
        people = pose_person_boxes(pose_model, image, args.pose_imgsz, args.pose_conf, args.device)
        roi_records = []
        for person in people:
            pbox = clamp_box(person["box"], w, h)
            if pbox is None:
                continue
            roi = expand_box(pbox, args.expand_x, args.expand_y, w, h)
            if roi is None:
                continue
            roi_records.append({"person": person, "person_box": pbox, "roi": roi, "phones": []})
        stats["pose_people"] += len(roi_records)
        if not roi_records:
            stats["phone_frames_without_pose_person"] += 1
            stats["unmatched_phone_boxes"] += len(phones)
            continue

        for phone in phones:
            pbox = phone["box"]
            pc = center(pbox)
            matched = []
            for rec in roi_records:
                roi = rec["roi"]
                inside = point_in_box(pc, roi)
                overlap = inter_area(pbox, roi) / max(1.0, box_area(pbox))
                if inside or overlap >= 0.20:
                    matched.append((overlap, rec))
            if not matched:
                stats["unmatched_phone_boxes"] += 1
                continue
            # Keep all meaningful containing ROIs. In overlaps this gives real crops for each visible person context.
            for _overlap, rec in matched:
                rec["phones"].append(phone)

        source_tag = safe_tag(json_path.parent, args.source)
        for rec_idx, rec in enumerate(roi_records):
            if not rec["phones"]:
                continue
            rx1, ry1, rx2, ry2 = rec["roi"]
            ix1, iy1 = int(math.floor(rx1)), int(math.floor(ry1))
            ix2, iy2 = int(math.ceil(rx2)), int(math.ceil(ry2))
            crop = image[iy1:iy2, ix1:ix2].copy()
            ch, cw = crop.shape[:2]
            if cw < 8 or ch < 8:
                stats["tiny_crop_skip"] += 1
                continue
            shapes = []
            for phone in rec["phones"]:
                x1, y1, x2, y2 = phone["box"]
                lx1, ly1 = max(0.0, x1 - ix1), max(0.0, y1 - iy1)
                lx2, ly2 = min(float(cw), x2 - ix1), min(float(ch), y2 - iy1)
                if lx2 <= lx1 + 1 or ly2 <= ly1 + 1:
                    continue
                shapes.append(
                    {
                        "label": "phone",
                        "points": [[lx1, ly1], [lx2, ly1], [lx2, ly2], [lx1, ly2]],
                        "group_id": None,
                        "description": f"global_phone_idx={phone['idx']} pose_conf={rec['person']['conf']:.4f}",
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                )
            if not shapes:
                stats["empty_after_clip"] += 1
                continue

            stem = f"global2person_{source_tag}_{json_path.stem}_p{rec['person']['idx']:02d}_{rec_idx:02d}"
            img_name = stem + ".jpg"
            json_name = stem + ".json"
            out_img = args.output / img_name
            out_json = args.output / json_name
            cv2.imwrite(str(out_img), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            label = make_labelme_json(
                img_name,
                cw,
                ch,
                shapes,
                f"Converted from full-frame LabelMe {json_path}; roi_xyxy_full_frame={[ix1, iy1, ix2, iy2]}; person_box={rec['person_box']}",
            )
            out_json.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")

            if args.review:
                review = crop.copy()
                for shape in shapes:
                    x1, y1 = [int(round(v)) for v in shape["points"][0]]
                    x2, y2 = [int(round(v)) for v in shape["points"][2]]
                    cv2.rectangle(review, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite(str(review_dir / img_name), review, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            stats["written_pairs"] += 1
            stats["written_phone_shapes"] += len(shapes)
            manifest.append(
                {
                    "source_json": str(json_path),
                    "source_image": str(image_path),
                    "output_image": str(out_img),
                    "output_json": str(out_json),
                    "person_idx": rec["person"]["idx"],
                    "pose_conf": f"{rec['person']['conf']:.4f}",
                    "phone_shapes": len(shapes),
                    "crop_w": cw,
                    "crop_h": ch,
                }
            )

    manifest_path = args.output / "_conversion_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_json", "source_image", "output_image", "output_json", "person_idx", "pose_conf", "phone_shapes", "crop_w", "crop_h"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    validation = Counter()
    for jp in args.output.glob("*.json"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        if int(data.get("imageWidth", 0)) >= 1000 and int(data.get("imageHeight", 0)) >= 700:
            validation["large_frame_like_json"] += 1
        if not data.get("shapes"):
            validation["empty_json"] += 1
        for shape in data.get("shapes", []):
            for x, y in shape.get("points", []):
                if x < 0 or y < 0 or x > data["imageWidth"] or y > data["imageHeight"]:
                    validation["out_of_bounds_points"] += 1
                    break

    summary = {
        "sources": [str(s) for s in args.source],
        "output": str(args.output),
        "pose_model": str(args.pose_model),
        "pose_imgsz": args.pose_imgsz,
        "stats": dict(stats),
        "validation": dict(validation),
        "manifest": str(manifest_path),
    }
    (args.output / "_conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
