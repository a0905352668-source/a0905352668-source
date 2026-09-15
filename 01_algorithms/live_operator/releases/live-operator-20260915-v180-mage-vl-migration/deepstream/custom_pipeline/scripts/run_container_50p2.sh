#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/boshi/Data/JianKong}"
STAGING_ROOT="${STAGING_ROOT:-${PROJECT_ROOT}/00_staging/deepstream_7x8_20260714}"
SOURCE_DIR="${SOURCE_DIR:-${STAGING_ROOT}/source/deepstream/custom_pipeline}"
BUILD_DIR="${BUILD_DIR:-${STAGING_ROOT}/build/custom_pipeline}"
CURRENT_INFERENCE_REGISTRY="${CURRENT_INFERENCE_REGISTRY:-${PROJECT_ROOT}/01_algorithms/CURRENT_INFERENCE_VERSION.json}"
LIVE_RUNTIME_MANIFEST="${LIVE_RUNTIME_MANIFEST:-${PROJECT_ROOT}/02_configs/runtime/live_runtime_manifest.json}"
LIVE_OPERATOR_PYTHONPATH="${LIVE_OPERATOR_PYTHONPATH:-${STAGING_ROOT}/source}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CALIB_DIR="${CALIB_DIR:-}"
POSE_ENGINE="${POSE_ENGINE:-}"
PHONE_ENGINE="${PHONE_ENGINE:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/06_training_runs/deepstream_7x8_${RUN_STAMP}}"
FIXED_TEMPLATE_DIR="${FIXED_TEMPLATE_DIR:-$(dirname "$(dirname "${OUTPUT_DIR}")")/fixed_object_templates}"
IMAGE="${IMAGE:-jiankong/deepstream:7.0-samples-dev}"
GPU_LOCK="${GPU_LOCK:-/tmp/jiankong_gpu0.lock}"
if [[ -z "${JIAN_KONG_OWNER_TOKEN:-}" ]]; then
  JIAN_KONG_OWNER_TOKEN="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_hex(16))')"
fi
if [[ ! "${JIAN_KONG_OWNER_TOKEN}" =~ ^[0-9a-f]{32}$ ]]; then
  echo "JIAN_KONG_OWNER_TOKEN_INVALID=${JIAN_KONG_OWNER_TOKEN}" >&2
  exit 2
fi

requested_runtime=(
  "${DEEPSTREAM_BINARY:-}"
  "${DEEPSTREAM_BINARY_SHA256:-}"
  "${POSE_ENGINE:-}"
  "${PHONE_ENGINE:-}"
  "${CALIB_DIR:-}"
  "${INFER_FPS:-}"
)
if ! manifest_identity="$(PYTHONPATH="${LIVE_OPERATOR_PYTHONPATH}" "${PYTHON_BIN}" \
    -m live_operator.inference resolve-shell \
    "${CURRENT_INFERENCE_REGISTRY}" "${LIVE_RUNTIME_MANIFEST}")"; then
  echo "MANIFEST_DEEPSTREAM_INVALID=${LIVE_RUNTIME_MANIFEST}" >&2
  exit 2
fi
mapfile -t manifest_lines <<<"${manifest_identity}"
if [[ "${#manifest_lines[@]}" -ne 6 ]]; then
  echo "MANIFEST_DEEPSTREAM_INVALID=${LIVE_RUNTIME_MANIFEST}" >&2
  exit 2
fi
for index in 0 1 2 3 4 5; do
  if [[ -n "${requested_runtime[${index}]}" && \
        "${requested_runtime[${index}]}" != "${manifest_lines[${index}]}" ]]; then
    echo "EXPLICIT_RUNTIME_IDENTITY_MISMATCH_INDEX=${index}" >&2
    exit 2
  fi
done
DEEPSTREAM_BINARY="${manifest_lines[0]}"
DEEPSTREAM_BINARY_SHA256="${manifest_lines[1]}"
POSE_ENGINE="${manifest_lines[2]}"
PHONE_ENGINE="${manifest_lines[3]}"
CALIB_DIR="${manifest_lines[4]}"
INFER_FPS="${manifest_lines[5]}"

for directory in "${SOURCE_DIR}" "${CALIB_DIR}"; do
  [[ -d "${directory}" ]] || { echo "MISSING_DIRECTORY=${directory}" >&2; exit 2; }
done
for file in "${POSE_ENGINE}" "${PHONE_ENGINE}" "${DEEPSTREAM_BINARY}"; do
  [[ -s "${file}" ]] || { echo "MISSING_OR_EMPTY_FILE=${file}" >&2; exit 2; }
done
printf '%s  %s\n' "${DEEPSTREAM_BINARY_SHA256}" "${DEEPSTREAM_BINARY}" | sha256sum -c - >/dev/null
mkdir -p "${OUTPUT_DIR}" "${FIXED_TEMPLATE_DIR}"

camera_manifest_args=()
if [[ -n "${CAMERA_MANIFEST_FILE:-}" ]]; then
  [[ -s "${CAMERA_MANIFEST_FILE}" ]] || {
    echo "MISSING_OR_EMPTY_CAMERA_MANIFEST=${CAMERA_MANIFEST_FILE}" >&2
    exit 2
  }
  camera_manifest_args=(
    -e CAMERA_MANIFEST_FILE=/configs/camera_manifest.json
    -v "${CAMERA_MANIFEST_FILE}:/configs/camera_manifest.json:ro"
  )
fi

unset USE_NEW_NVSTREAMMUX || true
exec {gpu_lock_fd}>"${GPU_LOCK}"
if ! flock -n "${gpu_lock_fd}"; then
  echo "GPU_LOCK_BUSY=${GPU_LOCK}" >&2
  exit 75
fi

docker run --rm \
  --name "jk-ds70-7x8-${RUN_STAMP}" \
  --label "jiankong.owner=${JIAN_KONG_OWNER_TOKEN}" \
  --runtime="${NVIDIA_RUNTIME:-nvidia}" \
  --network host \
  --ipc host \
  --shm-size=2g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e POSE_PLAN=/models/pose960_static_b7.plan \
  -e PHONE_PLAN=/models/phone_b16_img512_fp16_trt86.engine \
  -e CALIB_DIR=/configs/calibration \
  -e OUTPUT_DIR=/output \
  -e BUILD_DIR=/workspace/build \
  -e RTSP_BASE="${RTSP_BASE:-rtsp://192.168.50.3:8554}" \
  -e LOCAL_RELAY_BASE="${LOCAL_RELAY_BASE:-}" \
  -e INFER_FPS="${INFER_FPS:-8}" \
  -e DURATION_SEC="${DURATION_SEC:-300}" \
  -e ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION \
  -e STATIC_OBSERVATION_SECONDS \
  -e STATIC_PENDING_SECONDS \
  -e STATIC_LONG_CONFIRM_SECONDS \
  -e STATIC_MIN_DETECTION_RATIO \
  -e STATIC_MAX_GAP_SECONDS \
  -e STATIC_POSITION_RADIUS_RATIO \
  -e STATIC_HOTSPOT_ENABLED \
  -e FIXED_TEMPLATE_DIR=/fixed_templates \
  "${camera_manifest_args[@]}" \
  -v "${SOURCE_DIR}:/workspace/source:ro" \
  -v "${DEEPSTREAM_BINARY}:/workspace/build/jiankong_custom_pipeline:ro" \
  -v "${POSE_ENGINE}:/models/pose960_static_b7.plan:ro" \
  -v "${PHONE_ENGINE}:/models/phone_b16_img512_fp16_trt86.engine:ro" \
  -v "${CALIB_DIR}:/configs/calibration:ro" \
  -v "${OUTPUT_DIR}:/output:rw" \
  -v "${FIXED_TEMPLATE_DIR}:/fixed_templates:ro" \
  -w /workspace/source \
  "${IMAGE}" \
  env -u USE_NEW_NVSTREAMMUX \
  bash /workspace/source/scripts/run_7x8.sh
