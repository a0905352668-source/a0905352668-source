#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-ds70}"
DEEPSTREAM_ROOT="${DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream-7.0}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DJIANKONG_BUILD_DEEPSTREAM_PIPELINE=ON \
  -DDEEPSTREAM_ROOT="${DEEPSTREAM_ROOT}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
