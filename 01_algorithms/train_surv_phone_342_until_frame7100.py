#!/usr/bin/env python3
"""Train surveillance phone-only model with manually checked frames up to frame007100."""

from ultralytics import YOLO


DATA = (
    "/home/boshi/AI_programming/FacePhoneDet/datasets/"
    "surveillance_phone_only_Train_Data_342_until_041538_frame007100_70_30_20260624_yolo/data.yaml"
)
PRETRAINED = (
    "/home/boshi/AI_programming/FacePhoneDet/runs/PhoneDet_finetune/"
    "SURV_PHONE_ONLY_yolo11s_Train_Data_252_20260624/weights/best.pt"
)
PROJECT = "/home/boshi/AI_programming/FacePhoneDet/runs/PhoneDet_finetune"
NAME = "SURV_PHONE_ONLY_yolo11s_Train_Data_342_until_041538_frame7100_70_30_20260624"


def main() -> None:
    model = YOLO(PRETRAINED)
    model.train(
        data=DATA,
        project=PROJECT,
        name=NAME,
        epochs=120,
        patience=30,
        batch=4,
        imgsz=1280,
        device=0,
        workers=2,
        optimizer="AdamW",
        lr0=0.00012,
        lrf=0.04,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        hsv_h=0.003,
        hsv_s=0.12,
        hsv_v=0.12,
        degrees=0.0,
        translate=0.03,
        scale=0.12,
        fliplr=0.0,
        mosaic=0.1,
        close_mosaic=15,
        mixup=0.0,
        copy_paste=0.0,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        seed=20260624,
        deterministic=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
