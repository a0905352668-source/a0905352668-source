#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-ds70}"
: "${POSE_PLAN:?set POSE_PLAN}"
: "${PHONE_PLAN:?set PHONE_PLAN}"
: "${CALIB_DIR:?set CALIB_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${RTSP_FILE:?set RTSP_FILE}"
: "${CALIBRATION_FILE:?set CALIBRATION_FILE}"

PIPELINE_BIN="${BUILD_DIR}/jiankong_custom_pipeline"
INFER_FPS="${INFER_FPS:-8}"
DURATION_SEC="${DURATION_SEC:-86400}"
SOURCE_WIDTH="${SOURCE_WIDTH:-2560}"
SOURCE_HEIGHT="${SOURCE_HEIGHT:-1440}"

for required in "${PIPELINE_BIN}" "${POSE_PLAN}" "${PHONE_PLAN}" "${RTSP_FILE}" "${CALIBRATION_FILE}"; do
  [[ -s "${required}" ]] || { echo "MISSING_OR_EMPTY_FILE=${required}" >&2; exit 2; }
done
[[ -d "${CALIB_DIR}" ]] || { echo "MISSING_CALIBRATION_DIRECTORY=${CALIB_DIR}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"
[[ -w "${OUTPUT_DIR}" ]] || { echo "OUTPUT_NOT_WRITABLE=${OUTPUT_DIR}" >&2; exit 2; }

calibration_args=()
while IFS='=' read -r camera calibration; do
  [[ -n "${camera}" && -n "${calibration}" ]] || {
    echo "INVALID_CALIBRATION_ENTRY" >&2
    exit 2
  }
  [[ -s "${CALIB_DIR}/${calibration}" ]] || {
    echo "MISSING_CALIBRATION=${CALIB_DIR}/${calibration}" >&2
    exit 2
  }
  calibration_args+=(--calibration "${camera}=${calibration}")
done < "${CALIBRATION_FILE}"
[[ "${#calibration_args[@]}" -gt 0 ]] || { echo "NO_CALIBRATIONS" >&2; exit 2; }

exec "${PIPELINE_BIN}" \
  --rtsp-file "${RTSP_FILE}" \
  "${calibration_args[@]}" \
  --duration-sec "${DURATION_SEC}" \
  --infer-fps "${INFER_FPS}" \
  --source-width "${SOURCE_WIDTH}" \
  --source-height "${SOURCE_HEIGHT}" \
  --pose-plan "${POSE_PLAN}" \
  --phone-plan "${PHONE_PLAN}" \
  --calib-dir "${CALIB_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpu-preprocess \
  --no-video \
  --enable-spatial-static-phone-suppression \
  --static-hotspot-enabled
