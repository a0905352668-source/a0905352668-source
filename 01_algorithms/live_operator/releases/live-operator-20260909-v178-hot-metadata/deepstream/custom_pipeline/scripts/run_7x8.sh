#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-ds70}"
: "${POSE_PLAN:?set POSE_PLAN}"
: "${PHONE_PLAN:?set PHONE_PLAN}"
: "${CALIB_DIR:?set CALIB_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

RTSP_BASE="${LOCAL_RELAY_BASE:-${RTSP_BASE:-rtsp://192.168.50.3:8554}}"
DURATION_SEC="${DURATION_SEC:-300}"
INFER_FPS="${INFER_FPS:-8}"
PIPELINE_BIN="${BUILD_DIR}/jiankong_custom_pipeline"
ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION="${ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION:-0}"
STATIC_OBSERVATION_SECONDS="${STATIC_OBSERVATION_SECONDS:-3.0}"
STATIC_PENDING_SECONDS="${STATIC_PENDING_SECONDS:-0.75}"
STATIC_LONG_CONFIRM_SECONDS="${STATIC_LONG_CONFIRM_SECONDS:-6.0}"
STATIC_MIN_DETECTION_RATIO="${STATIC_MIN_DETECTION_RATIO:-0.60}"
STATIC_MAX_GAP_SECONDS="${STATIC_MAX_GAP_SECONDS:-0.75}"
STATIC_POSITION_RADIUS_RATIO="${STATIC_POSITION_RADIUS_RATIO:-0.03}"
STATIC_HOTSPOT_ENABLED="${STATIC_HOTSPOT_ENABLED:-1}"
FIXED_TEMPLATE_DIR="${FIXED_TEMPLATE_DIR:-}"
CAMERA_MANIFEST_FILE="${CAMERA_MANIFEST_FILE:-}"

require_nonempty_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "MISSING_OR_EMPTY_${label}=${path}" >&2
    exit 2
  fi
}

require_nonempty_file "${PIPELINE_BIN}" "PIPELINE_BINARY"
require_nonempty_file "${POSE_PLAN}" "POSE_ENGINE"
require_nonempty_file "${PHONE_PLAN}" "PHONE_ENGINE"
[[ -d "${CALIB_DIR}" ]] || { echo "MISSING_CALIB_DIR=${CALIB_DIR}" >&2; exit 2; }

camera_args=()
if [[ -n "${CAMERA_MANIFEST_FILE}" ]]; then
  require_nonempty_file "${CAMERA_MANIFEST_FILE}" "CAMERA_MANIFEST"
  camera_args+=(--camera-manifest "${CAMERA_MANIFEST_FILE}" --relay-base "${RTSP_BASE}")
else
  calibration_files=(
    camera_01_screen_calibration_v21.json
    camera_02_screen_calibration_v21.json
    camera_mechanical_01_screen_calibration_v21.json
    camera_mechanical_02_screen_calibration_v21.json
    camera_software_01_screen_calibration_v21.json
    camera_software_02_screen_calibration_v21.json
    camera_corridor_screen_calibration_v21.json
  )
  for calibration_file in "${calibration_files[@]}"; do
    require_nonempty_file "${CALIB_DIR}/${calibration_file}" "CALIBRATION"
  done

  # Legacy v1 route policy: camera01 has no protected screen.
  python3 - "${CALIB_DIR}/camera_01_screen_calibration_v21.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
screens = payload.get("screens")
if not isinstance(screens, list) or screens:
    raise SystemExit(
        f"CAMERA01_CALIBRATION_MUST_HAVE_EXPLICIT_EMPTY_SCREENS={path}"
    )
PY
  camera_args=(
    --rtsp "dianqi1=${RTSP_BASE}/camera01"
    --rtsp "dianqi2=${RTSP_BASE}/camera02"
    --rtsp "jixie1=${RTSP_BASE}/camera03"
    --rtsp "jixie2=${RTSP_BASE}/camera04"
    --rtsp "ruanjian1=${RTSP_BASE}/camera05"
    --rtsp "ruanjian2=${RTSP_BASE}/camera06"
    --rtsp "zoulang=${RTSP_BASE}/camera07"
  )
fi

mkdir -p "${OUTPUT_DIR}"
[[ -w "${OUTPUT_DIR}" ]] || { echo "OUTPUT_NOT_WRITABLE=${OUTPUT_DIR}" >&2; exit 2; }

spatial_static_args=(
  --static-observation-seconds "${STATIC_OBSERVATION_SECONDS}"
  --static-pending-seconds "${STATIC_PENDING_SECONDS}"
  --static-long-confirm-seconds "${STATIC_LONG_CONFIRM_SECONDS}"
  --static-min-detection-ratio "${STATIC_MIN_DETECTION_RATIO}"
  --static-max-gap-seconds "${STATIC_MAX_GAP_SECONDS}"
  --static-position-radius-ratio "${STATIC_POSITION_RADIUS_RATIO}"
)
case "${ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION}" in
  1) spatial_static_args+=(--enable-spatial-static-phone-suppression) ;;
  0) spatial_static_args+=(--disable-spatial-static-phone-suppression) ;;
  *) echo "INVALID_ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION=${ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION}" >&2; exit 2 ;;
esac
case "${STATIC_HOTSPOT_ENABLED}" in
  1) spatial_static_args+=(--static-hotspot-enabled) ;;
  0) spatial_static_args+=(--static-hotspot-disabled) ;;
  *) echo "INVALID_STATIC_HOTSPOT_ENABLED=${STATIC_HOTSPOT_ENABLED}" >&2; exit 2 ;;
esac
fixed_template_args=()
if [[ -n "${FIXED_TEMPLATE_DIR}" ]]; then
  [[ -d "${FIXED_TEMPLATE_DIR}" ]] || {
    echo "MISSING_FIXED_TEMPLATE_DIR=${FIXED_TEMPLATE_DIR}" >&2
    exit 2
  }
  fixed_template_args+=(--fixed-template-dir "${FIXED_TEMPLATE_DIR}")
fi

exec "${PIPELINE_BIN}" \
  "${camera_args[@]}" \
  --duration-sec "${DURATION_SEC}" \
  --infer-fps "${INFER_FPS}" \
  --source-width 2560 \
  --source-height 1440 \
  --pose-plan "${POSE_PLAN}" \
  --phone-plan "${PHONE_PLAN}" \
  --calib-dir "${CALIB_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpu-preprocess \
  --no-video \
  --disable-static-phone-suppression \
  "${fixed_template_args[@]}" \
  "${spatial_static_args[@]}"
