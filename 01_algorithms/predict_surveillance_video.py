#!/usr/bin/env python3
"""Run the independent surveillance-camera YOLO model on one video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--class-map",
        default="",
        help="Override class names, e.g. '0:phone,1:face' when a dataset class mapping was reversed.",
    )
    return parser.parse_args()


def draw_label(image, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_text = max(th + baseline + 4, y1)
    cv2.rectangle(image, (x1, y_text - th - baseline - 6), (x1 + tw + 8, y_text + 4), color, -1)
    cv2.putText(image, text, (x1 + 4, y_text - baseline - 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {args.output}")

    model = YOLO(str(args.model))
    names = dict(model.names)
    if args.class_map:
        for item in args.class_map.split(","):
            class_id, name = item.split(":", 1)
            names[int(class_id.strip())] = name.strip()
    colors = {
        0: (0, 0, 255),     # phone for corrected SURV001 mapping
        1: (40, 180, 40),   # face for corrected SURV001 mapping
    }
    counts = {"frames": 0, "frames_with_face": 0, "frames_with_phone": 0, "face_boxes": 0, "phone_boxes": 0}

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        frame_face = 0
        frame_phone = 0
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                color = colors.get(cls_id, (255, 180, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
                draw_label(frame, label, x1, y1, color)
                if names.get(cls_id) == "face":
                    frame_face += 1
                elif names.get(cls_id) == "phone":
                    frame_phone += 1

        counts["frames"] += 1
        counts["face_boxes"] += frame_face
        counts["phone_boxes"] += frame_phone
        counts["frames_with_face"] += int(frame_face > 0)
        counts["frames_with_phone"] += int(frame_phone > 0)
        writer.write(frame)
        if index % 100 == 0 or index == total:
            print(f"[{index}/{total}] face_frames={counts['frames_with_face']} phone_frames={counts['frames_with_phone']}", flush=True)

    cap.release()
    writer.release()
    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "model": str(args.model),
        "video": str(args.video),
        "output": str(args.output),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "fps": fps,
        "width": width,
        "height": height,
        **counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] video={args.output}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
