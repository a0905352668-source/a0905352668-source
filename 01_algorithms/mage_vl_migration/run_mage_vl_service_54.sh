#!/usr/bin/env bash
# Dedicated Mage-VL review service for 192.168.104.54.  Runtime material is
# provisioned separately on that host; this tracked wrapper deliberately does
# not create or repair private files.
set -euo pipefail

PROJECT_ROOT="/home/zty/YL/JianKong"
SERVICE_ROOT="${PROJECT_ROOT}/01_algorithms/mage_vl_service"
RELEASES_DIR="${SERVICE_ROOT}/releases"
CURRENT_LINK="${SERVICE_ROOT}/current"
MODEL_DIR="${PROJECT_ROOT}/07_models/vlm_models/versions/v20260809_01_mage_vl_awq_int4"
VENV_PYTHON="${PROJECT_ROOT}/08_envs/mage-vl-20260915/bin/python"
RUNTIME_DIR="${SERVICE_ROOT}/runtime"
TLS_DIR="${RUNTIME_DIR}/tls"
TLS_CERT_FILE="${RUNTIME_DIR}/tls/mage-vl-54.crt"
TLS_KEY_FILE="${RUNTIME_DIR}/tls/mage-vl-54.key"
SHARED_SECRET_FILE="${RUNTIME_DIR}/mage-vl-shared.secret"
CACHE_DIR="${RUNTIME_DIR}/cache"
GPU_LOCK_FILE="${RUNTIME_DIR}/gpu0.lock"

fail() {
  echo "jiankong-mage-vl-54: $*" >&2
  exit 78
}

# Check each path component with lstat semantics before any operation that may
# follow it.  current is intentionally the one allowed symlink; its resolved
# target is checked separately below.
assert_no_symlink_path() {
  local path="$1"
  local component="/"
  local segment
  local -a segments

  [[ "${path}" == /* ]] || fail "path is not absolute"
  IFS=/ read -r -a segments <<< "${path#/}"
  for segment in "${segments[@]}"; do
    [[ -n "${segment}" ]] || continue
    component="${component%/}/${segment}"
    [[ -e "${component}" || -L "${component}" ]] || fail "required path is absent"
    [[ -L "${component}" ]] && fail "symlinked path is not allowed"
  done
}

assert_owned_mode() {
  local path="$1"
  local expected_mode="$2"
  local actual_uid
  local actual_mode

  assert_no_symlink_path "${path}"
  actual_uid="$(stat -c '%u' -- "${path}")"
  actual_mode="$(stat -c '%a' -- "${path}")"
  [[ "${actual_uid}" == "${SERVICE_UID}" ]] || fail "runtime material has wrong owner"
  [[ "${actual_mode}" == "${expected_mode}" ]] || fail "runtime material has unsafe mode"
}

EXPECTED_SERVICE_UID="$(id -u zty)"
SERVICE_UID="$(id -u)"
[[ "${SERVICE_UID}" == "${EXPECTED_SERVICE_UID}" ]] || fail "must run as zty"

assert_no_symlink_path "${SERVICE_ROOT}"
assert_no_symlink_path "${RELEASES_DIR}"
assert_no_symlink_path "${RUNTIME_DIR}"
assert_no_symlink_path "${TLS_DIR}"
assert_no_symlink_path "${CACHE_DIR}"
assert_owned_mode "${RUNTIME_DIR}" 700
assert_owned_mode "${TLS_DIR}" 700
assert_owned_mode "${CACHE_DIR}" 700
assert_owned_mode "${TLS_KEY_FILE}" 600
assert_owned_mode "${SHARED_SECRET_FILE}" 600
assert_owned_mode "${GPU_LOCK_FILE}" 600
assert_owned_mode "${TLS_CERT_FILE}" 644

[[ -L "${CURRENT_LINK}" ]] || fail "current release selector must be a symlink"
CURRENT_RELEASE="$(realpath -e -- "${CURRENT_LINK}")"
RESOLVED_RELEASES_DIR="$(realpath -e -- "${RELEASES_DIR}")"
[[ -d "${CURRENT_RELEASE}" ]] || fail "current release is not a directory"
[[ "${CURRENT_RELEASE}" == "${RESOLVED_RELEASES_DIR}"/* ]] || fail "current release escapes releases"
assert_no_symlink_path "${CURRENT_RELEASE}"

[[ -d "${MODEL_DIR}" ]] || fail "model directory is absent"
[[ -x "${VENV_PYTHON}" ]] || fail "dedicated virtualenv Python is absent"

# Do not let a systemd manager environment or an interactive shell change the
# imported application.  The selected release is the sole Python import root.
unset PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE LD_PRELOAD
export PATH=/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${CURRENT_RELEASE}"

exec "${VENV_PYTHON}" -P -m live_operator.mage_vl_service \
  --host "192.168.104.54" \
  --port "8879" \
  --model "${MODEL_DIR}" \
  --model-version "mage-vl-awq-v20260809-1f7f5266fa4e-phone-use-prompt-v2" \
  --shared-secret-file "${SHARED_SECRET_FILE}" \
  --cache-dir "${CACHE_DIR}" \
  --gpu-weight-memory "3800MiB" \
  --cpu-memory "24GiB" \
  --max-request-bytes "67108864" \
  --tls-cert-file "${TLS_CERT_FILE}" \
  --tls-key-file "${TLS_KEY_FILE}" \
  --gpu-lock-file "${GPU_LOCK_FILE}"
