#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=person_roi_phone_yolo11s_640_network_plus_person_20260703
PIDFILE=/media/boshi/Data/JianKong/06_training_runs/${RUN_NAME}.pid
WATCH_LOG=/media/boshi/Data/JianKong/06_training_runs/${RUN_NAME}_watch.log
MODELS_DIR=/media/boshi/Data/JianKong/07_models/phone_models

echo "watch_start $(date +%F_%T)" > "$WATCH_LOG"
if [ ! -f "$PIDFILE" ]; then
  echo "missing_pidfile $PIDFILE" >> "$WATCH_LOG"
  exit 1
fi

PID=$(cat "$PIDFILE")
echo "watch_pid $PID" >> "$WATCH_LOG"
while ps -p "$PID" >/dev/null 2>&1; do
  sleep 60
done

RUN_DIR=/media/boshi/Data/00_active_projects/JianKong/06_training_runs/${RUN_NAME}_b32
if [ ! -f "$RUN_DIR/weights/best.pt" ]; then
  RUN_DIR=/media/boshi/Data/00_active_projects/JianKong/06_training_runs/${RUN_NAME}_b16
fi

echo "train_stopped $(date +%F_%T)" >> "$WATCH_LOG"
echo "run_dir $RUN_DIR" >> "$WATCH_LOG"
mkdir -p "$MODELS_DIR"

if [ -f "$RUN_DIR/weights/best.pt" ]; then
  cp -f "$RUN_DIR/weights/best.pt" "$MODELS_DIR/PERSON_ROI_PHONE_yolo11s_640_network_plus_person_20260703_best.pt"
  cp -f "$RUN_DIR/weights/last.pt" "$MODELS_DIR/PERSON_ROI_PHONE_yolo11s_640_network_plus_person_20260703_last.pt"
  echo "copied_best $MODELS_DIR/PERSON_ROI_PHONE_yolo11s_640_network_plus_person_20260703_best.pt" >> "$WATCH_LOG"
  echo "copied_last $MODELS_DIR/PERSON_ROI_PHONE_yolo11s_640_network_plus_person_20260703_last.pt" >> "$WATCH_LOG"
else
  echo "missing_weights" >> "$WATCH_LOG"
fi

if [ -f "$RUN_DIR/results.csv" ]; then
  tail -5 "$RUN_DIR/results.csv" >> "$WATCH_LOG"
fi

echo "watch_end $(date +%F_%T)" >> "$WATCH_LOG"
