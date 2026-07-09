#!/usr/bin/env python3
"""Build a YOLO phone dataset from network YOLO data plus Person ROI LabelMe data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PHONE_LABELS = {"phone", "mobile", "cellphone", "cell phone", "手机"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-yolo", type=Path, required=True)
    parser.add_argument("--person-labelme", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-phone-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


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


def stable_name(prefix: str, path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in path.stem)
    return f"{prefix}_{stem}_{digest}{path.suffix.lower()}"


def split_for_path(path: Path, seed: int, train_ratio: float, val_ratio: float) -> str:
    key = f"{seed}:{path}"
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def shape_to_yolo(shape: dict, width: int, height: int) -> str | None:
    label = str(shape.get("label", "")).strip().lower()
    if label not in PHONE_LABELS:
        return None
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x1 = max(0.0, min(float(width), min(xs)))
    y1 = max(0.0, min(float(height), min(ys)))
    x2 = max(0.0, min(float(width), max(xs)))
    y2 = max(0.0, min(float(height), max(ys)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    vals = [max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh)]
    if vals[2] <= 0.0 or vals[3] <= 0.0:
        return None
    return "0 " + " ".join(f"{v:.6f}" for v in vals)


def rewrite_yolo_label(label_path: Path, phone_class: int) -> tuple[list[str], int]:
    if not label_path.exists():
        return [], 0
    lines = []
    dropped = 0
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if cls_id != phone_class:
            dropped += 1
            continue
        vals = [max(0.0, min(1.0, v)) for v in vals]
        if vals[2] <= 0.0 or vals[3] <= 0.0:
            continue
        lines.append("0 " + " ".join(f"{v:.6f}" for v in vals))
    return lines, dropped


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def write_item(
    output: Path,
    split: str,
    image_path: Path,
    label_lines: list[str],
    out_name: str,
    copy_images: bool,
) -> tuple[Path, Path]:
    out_img = output / split / "images" / out_name
    out_label = output / split / "labels" / (Path(out_name).stem + ".txt")
    link_or_copy(image_path, out_img, copy_images)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    out_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return out_img, out_label


def main() -> None:
    args = parse_args()
    if args.clear_output and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = Counter()
    manifest: list[dict] = []

    for split in ("train", "val", "test"):
        images_root = args.network_yolo / split / "images"
        labels_root = args.network_yolo / split / "labels"
        if not images_root.exists():
            continue
        for image_path in sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            rel = image_path.relative_to(images_root)
            label_path = labels_root / rel.with_suffix(".txt")
            lines, dropped = rewrite_yolo_label(label_path, args.network_phone_class)
            out_name = stable_name(f"net_{split}", image_path)
            out_img, out_label = write_item(args.output, split, image_path, lines, out_name, args.copy_images)
            stats[f"network_{split}_images"] += 1
            stats[f"network_{split}_phone_boxes"] += len(lines)
            stats[f"network_{split}_dropped_non_phone_boxes"] += dropped
            if not lines:
                stats[f"network_{split}_background_images"] += 1
            manifest.append(
                {
                    "source_type": "network_yolo",
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "split": split,
                    "output_image": str(out_img),
                    "output_label": str(out_label),
                    "phone_boxes": len(lines),
                    "dropped_non_phone_boxes": dropped,
                }
            )

    labelme_items = []
    for root in args.person_labelme:
        for json_path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                stats["labelme_bad_json"] += 1
                continue
            shapes = data.get("shapes")
            if not isinstance(shapes, list):
                stats["labelme_non_labelme_json"] += 1
                continue
            image_path = find_image(json_path, data)
            if image_path is None:
                stats["labelme_missing_image"] += 1
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                stats["labelme_unreadable_image"] += 1
                continue
            height, width = img.shape[:2]
            lines = []
            for shape in shapes:
                if not isinstance(shape, dict):
                    continue
                line = shape_to_yolo(shape, width, height)
                if line is not None:
                    lines.append(line)
            labelme_items.append((json_path, image_path, lines, width, height))

    random.Random(args.seed).shuffle(labelme_items)
    for json_path, image_path, lines, width, height in labelme_items:
        split = split_for_path(json_path, args.seed, args.train_ratio, args.val_ratio)
        out_name = stable_name("person", image_path)
        out_img, out_label = write_item(args.output, split, image_path, lines, out_name, args.copy_images)
        stats[f"labelme_{split}_images"] += 1
        stats[f"labelme_{split}_phone_boxes"] += len(lines)
        if width >= 1000 and height >= 700:
            stats["labelme_large_frame_like_images"] += 1
        manifest.append(
            {
                "source_type": "person_labelme",
                "source_image": str(image_path),
                "source_label": str(json_path),
                "split": split,
                    "output_image": str(out_img),
                    "output_label": str(out_label),
                    "phone_boxes": len(lines),
                    "dropped_non_phone_boxes": 0,
                }
            )

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        f"path: {args.output}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: phone\n",
        encoding="utf-8",
    )

    with (args.output / "_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_type", "source_image", "source_label", "split", "output_image", "output_label", "phone_boxes", "dropped_non_phone_boxes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    for split in ("train", "val", "test"):
        stats[f"total_{split}_images"] = sum(1 for _ in (args.output / split / "images").iterdir())
        stats[f"total_{split}_labels"] = sum(1 for _ in (args.output / split / "labels").iterdir())
    stats["total_images"] = stats["total_train_images"] + stats["total_val_images"] + stats["total_test_images"]
    stats["total_phone_boxes"] = (
        stats["network_train_phone_boxes"]
        + stats["network_val_phone_boxes"]
        + stats["network_test_phone_boxes"]
        + stats["labelme_train_phone_boxes"]
        + stats["labelme_val_phone_boxes"]
        + stats["labelme_test_phone_boxes"]
    )

    summary = {
        "network_yolo": str(args.network_yolo),
        "network_phone_class": args.network_phone_class,
        "person_labelme": [str(p) for p in args.person_labelme],
        "output": str(args.output),
        "data_yaml": str(data_yaml),
        "stats": dict(stats),
    }
    (args.output / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
