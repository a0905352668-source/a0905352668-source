#!/usr/bin/env python3
"""Train surveillance-camera phone detector V1 independently from PCF models."""

from ultralytics import YOLO


DATA = (
    "/media/boshi/Data/TrainData/FacePhoneTrainData/Dataset_txt/"
    "SURV001_reorganized_phone_face_clean/data.yaml"
)
PROJECT = "/media/boshi/Data/TrainData/FacePhoneTrainData/runs/surveillance"
NAME = "SURV001_yolo11s_phone_face_reorganized_640_b32_20260612"
PRETRAINED = "/home/boshi/AI_programming/FacePhoneDet/yolo11s.pt"


def main() -> None:
    model = YOLO(PRETRAINED)
    model.train(
        data=DATA,
        project=PROJECT,
        name=NAME,
        epochs=100,
        patience=20,
        batch=32,
        imgsz=640,
        device=0,
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=3.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.7,
        close_mosaic=10,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        seed=20260612,
        deterministic=True,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
