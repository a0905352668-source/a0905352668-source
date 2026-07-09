#!/usr/bin/env bash
set -euo pipefail

cd /media/boshi/Data/JianKong/01_algorithms

PORT="${PORT:-8767}"
MAX_BATCHES="${MAX_BATCHES:-0}"
INFER_FPS="${INFER_FPS:-8}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-10}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/boshi/Data/JianKong/03_raw_videos_and_frames/2026-07-03}"
PICK_FROM_END="${PICK_FROM_END:-1}"
LOG="${LOG:-/tmp/jiankong_realtime_sim_${PORT}.log}"
CLIP_BUILD_MODE="${CLIP_BUILD_MODE:-background}"
NO_CPP_VIDEO="${NO_CPP_VIDEO:-1}"

CPP_VIDEO_ARGS=()
if [[ "$NO_CPP_VIDEO" == "1" ]]; then
  CPP_VIDEO_ARGS+=(--no-cpp-video)
else
  CPP_VIDEO_ARGS+=(--with-cpp-video)
fi

oldpids=$(ss -ltnp 2>/dev/null | awk -v port=":${PORT}" '$0 ~ port {while(match($0,/pid=[0-9]+/)){print substr($0,RSTART+4,RLENGTH-4); $0=substr($0,RSTART+RLENGTH)}}' | sort -u)
for pid in $oldpids; do
  kill "$pid" 2>/dev/null || true
done

: > "$LOG"
nohup env PYTHONPATH=/media/boshi/Data/JianKong/01_algorithms \
  /home/boshi/miniconda3/bin/python -m realtime_sim.run_realtime_sim \
  --mode realtime \
  --max-batches "$MAX_BATCHES" \
  --infer-fps "$INFER_FPS" \
  --segment-seconds "$SEGMENT_SECONDS" \
  --source-root "$SOURCE_ROOT" \
  --pick-from-end "$PICK_FROM_END" \
  --port "$PORT" \
  --keep-serving \
  --cpp-binary /media/boshi/Data/JianKong/01_algorithms/tools/cpp_full_pipeline_bench_gpu_novideo \
  "${CPP_VIDEO_ARGS[@]}" \
  --clip-build-mode "$CLIP_BUILD_MODE" \
  --clip-python /home/boshi/miniconda3/bin/python \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "url=http://192.168.50.2:${PORT}/"
echo "log=$LOG"
