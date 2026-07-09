#!/usr/bin/env bash
set -euo pipefail

export YOLO_AUTOINSTALL=False
export PIP_DISABLE_PIP_VERSION_CHECK=1

PY=/home/boshi/miniconda3/envs/yolo/bin/python
SCRIPT=/media/boshi/Data/JianKong/01_algorithms/tools/prelabel_person_roi_phone_from_videos.py
LOG=/media/boshi/Data/JianKong/06_training_runs/prelabel_20260701_7videos_person_roi_10fps_trt_2fps_pid3s.log
PIDFILE=/media/boshi/Data/JianKong/06_training_runs/prelabel_20260701_7videos_person_roi_10fps_trt_2fps_pid3s.pid
OUT=/media/boshi/Data/JianKong/04_labelme_datasets/Person标注/20260703_20260701_7videos_person_roi_phone_prelabel_10fps_trt_2fps_pid3s

echo $$ > "$PIDFILE"
{
  echo "QUEUE_START $(date '+%F %T')"
  echo "script $SCRIPT"
  echo "output $OUT"

  while pgrep -af 'yolo detect train|detect train resume=True' | grep -v grep >/dev/null; do
    echo "WAIT_TRAINING $(date '+%F %T')"
    pgrep -af 'yolo detect train|detect train resume=True' | head -5 || true
    sleep 60
  done

  for _ in $(seq 1 120); do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo 0)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo 0)
    echo "WAIT_GPU $(date '+%F %T') mem=${mem}MiB util=${util}%"
    if [ "${mem:-99999}" -lt 1800 ]; then
      break
    fi
    sleep 30
  done

  echo "PRELABEL_START $(date '+%F %T')"
  mkdir -p "$OUT"
  ionice -c2 -n7 nice -n 10 "$PY" "$SCRIPT" \
    --video-root /media/boshi/Data/JianKong/03_raw_videos_and_frames/2026-07-01 \
    --output-dir "$OUT" \
    --pose-model /media/boshi/Data/JianKong/07_models/pose_models/yolo11s-pose.engine \
    --phone-model /media/boshi/Data/JianKong/07_models/phone_models/TRUE_PERSON_ROI_PHONE_yolo11s_network_roi_plus_own_roi_20260701_best_epoch36.engine \
    --calib-dir /media/boshi/Data/JianKong/02_configs/surveillance \
    --infer-fps 10 \
    --save-fps 2 \
    --per-id-seconds 3 \
    --pose-imgsz 960 \
    --phone-imgsz 640 \
    --roi-batch-size 8 \
    --device 0
  code=$?
  echo "PRELABEL_END $(date '+%F %T') exit=${code}"
  exit "$code"
} >> "$LOG" 2>&1
