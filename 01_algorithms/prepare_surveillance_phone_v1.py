#!/usr/bin/env python3
"""Prepare the independent surveillance-camera phone dataset for YOLO detection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_label(path: Path) -> tuple[list[str], int, int]:
    output: list[str] = []
    concatenated_lines = 0
    dropped_boxes = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) % 5:
            raise ValueError(f"{path}: label line has {len(tokens)} fields")
        if len(tokens) > 5:
            concatenated_lines += 1

        for offset in range(0, len(tokens), 5):
            class_id = int(float(tokens[offset]))
            x, y, width, height = map(float, tokens[offset + 1 : offset + 5])
            if class_id not in (0, 1):
                raise ValueError(f"{path}: unsupported class {class_id}")
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                dropped_boxes += 1
                continue
            output.append(f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}")

    return output, concatenated_lines, dropped_boxes


def replace_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.symlink(source.resolve(), destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    summary: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "classes": {"0": "face", "1": "phone"},
        "splits": {},
    }

    for split in SPLITS:
        source_images = source / split / "images"
        source_labels = source / split / "labels"
        output_images = output / "images" / split
        output_labels = output / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        images = sorted(
            path for path in source_images.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        stats = {
            "images": len(images),
            "boxes": 0,
            "empty_labels": 0,
            "concatenated_lines_fixed": 0,
            "invalid_boxes_dropped": 0,
        }

        for image in images:
            source_label = source_labels / f"{image.stem}.txt"
            if not source_label.exists():
                raise FileNotFoundError(f"Missing label for {image}")
            labels, fixed, dropped = parse_label(source_label)
            replace_symlink(image, output_images / image.name)
            (output_labels / f"{image.stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )
            stats["boxes"] += len(labels)
            stats["empty_labels"] += int(not labels)
            stats["concatenated_lines_fixed"] += fixed
            stats["invalid_boxes_dropped"] += dropped

        summary["splits"][split] = stats

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: face",
                "  1: phone",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
