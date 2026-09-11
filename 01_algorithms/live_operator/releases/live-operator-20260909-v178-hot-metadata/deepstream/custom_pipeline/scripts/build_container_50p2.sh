#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/boshi/Data/JianKong}"
STAGING_ROOT="${STAGING_ROOT:-${PROJECT_ROOT}/00_staging/deepstream_7x8_20260714}"
SOURCE_DIR="${SOURCE_DIR:-${STAGING_ROOT}/source/deepstream/custom_pipeline}"
REPO_ROOT="${REPO_ROOT:-}"
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(cd "${SOURCE_DIR}/../.." && pwd)"
fi
BUILD_DIR="${BUILD_DIR:-${STAGING_ROOT}/build/custom_pipeline}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGING_ROOT}/output/build-validation}"
CALIB_DIR="${CALIB_DIR:-${PROJECT_ROOT}/02_configs/surveillance}"
POSE_ENGINE="${POSE_ENGINE:-${PROJECT_ROOT}/06_training_runs/raw_trt_plans_20260706_114432/pose960_static_b7.plan}"
PHONE_ENGINE="${PHONE_ENGINE:-${PROJECT_ROOT}/06_training_runs/cpp_group01_group02_latest_model_static_20260710_132056/phone_b16_img512_fp16_trt86.engine}"
IMAGE="${IMAGE:-jiankong/deepstream:7.0-samples-dev}"
DOCKERFILE="${DOCKERFILE:-${SOURCE_DIR}/docker/Dockerfile.samples-dev}"

require_path() {
  local path="$1"
  local kind="$2"
  if [[ "${kind}" == "dir" ]]; then
    [[ -d "${path}" ]] || { echo "MISSING_DIRECTORY=${path}" >&2; exit 2; }
  else
    [[ -s "${path}" ]] || { echo "MISSING_OR_EMPTY_FILE=${path}" >&2; exit 2; }
  fi
}

require_path "${SOURCE_DIR}" dir
require_path "${DOCKERFILE}" file
require_path "${REPO_ROOT}/tools/extract_static_phone_regression_fixture.py" file
require_path "${REPO_ROOT}/tools/validate_static_phone_regression.py" file
require_path "${CALIB_DIR}" dir
require_path "${POSE_ENGINE}" file
require_path "${PHONE_ENGINE}" file
mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

unset USE_NEW_NVSTREAMMUX || true
DOCKERFILE_ARG="${DOCKERFILE}"
if [[ "${DOCKERFILE}" == "${SOURCE_DIR}/"* ]]; then
  DOCKERFILE_ARG="${DOCKERFILE#${SOURCE_DIR}/}"
fi
(
  cd "${SOURCE_DIR}"
  docker build --pull=false -t "${IMAGE}" -f "${DOCKERFILE_ARG}" .
)

docker run --rm \
  --name jk-ds70-custom-build \
  --runtime="${NVIDIA_RUNTIME:-nvidia}" \
  --network host \
  --ipc host \
  --shm-size=2g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e JIANKONG_REPO_ROOT=/workspace/repo \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${SOURCE_DIR}:/workspace/source:ro" \
  -v "${BUILD_DIR}:/workspace/build:rw" \
  -v "${POSE_ENGINE}:/models/pose960_static_b7.plan:ro" \
  -v "${PHONE_ENGINE}:/models/phone_b16_img512_fp16_trt86.engine:ro" \
  -v "${CALIB_DIR}:/configs/calibration:ro" \
  -v "${OUTPUT_DIR}:/output:rw" \
  -w /workspace/source \
  "${IMAGE}" \
  env -u USE_NEW_NVSTREAMMUX \
  BUILD_DIR=/workspace/build \
  bash /workspace/source/scripts/build.sh
