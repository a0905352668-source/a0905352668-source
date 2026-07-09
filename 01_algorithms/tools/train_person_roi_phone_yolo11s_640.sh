#!/usr/bin/env bash
set -o pipefail

export YOLO_AUTOINSTALL=False
export PIP_DISABLE_PIP_VERSION_CHECK=1

MODEL=/media/boshi/Data/JianKong/07_models/phone_models/TRUE_PERSON_ROI_PHONE_yolo11s_network_roi_plus_own_roi_20260701_best_epoch36.pt
DATA=/media/boshi/Data/JianKong/05_yolo_datasets/person_roi_phone_network_plus_person_20260703_yolo/data.yaml
PROJECT=/media/boshi/Data/JianKong/06_training_runs
RUN_NAME=person_roi_phone_yolo11s_640_network_plus_person_20260703
YOLO=/home/boshi/miniconda3/envs/yolo/bin/yolo

run_train() {
  local batch="$1"
  local name="${RUN_NAME}_b${batch}"
  echo "===== TRAIN START $(date +%F_%T) batch=${batch} name=${name} ====="
  "$YOLO" detect train \
    model="$MODEL" \
    data="$DATA" \
    imgsz=640 \
    epochs=80 \
    batch="$batch" \
    device=0 \
    workers=8 \
    project="$PROJECT" \
    name="$name" \
    patience=20 \
    cache=False \
    close_mosaic=10 \
    plots=True
}

run_train 32 || {
  echo "===== batch=32 failed at $(date +%F_%T), retry batch=16 ====="
  run_train 16
}
code=$?
echo "===== TRAIN END $(date +%F_%T) exit=${code} ====="
exit "$code"
