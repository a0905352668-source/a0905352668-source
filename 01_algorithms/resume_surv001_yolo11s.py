#!/usr/bin/env python3
"""Resume the persistent SURV001 surveillance-camera training run."""

from ultralytics import YOLO


LAST = (
    "/media/boshi/Data/TrainData/FacePhoneTrainData/runs/surveillance/"
    "SURV001_yolo11s_phone_face_reorganized_640_b32_20260612/weights/last.pt"
)


def main() -> None:
    YOLO(LAST).train(resume=True)


if __name__ == "__main__":
    main()
