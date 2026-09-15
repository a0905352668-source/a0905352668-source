#!/usr/bin/env bash
# Dedicated Mage-VL review service for 192.168.104.54. Runtime material is
# provisioned separately on that host; this tracked wrapper never creates it.
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
  return 78
}

# Every component is inspected before the caller performs an operation that
# could follow it. current is the one deliberate symlink and is handled by the
# release validator rather than this helper.
assert_no_symlink_path() {
  local path="$1"
  local component="/"
  local segment
  local -a segments

  [[ "${path}" == /* ]] || { fail "path is not absolute"; return; }
  IFS=/ read -r -a segments <<< "${path#/}"
  for segment in "${segments[@]}"; do
    [[ -n "${segment}" ]] || continue
    component="${component%/}/${segment}"
    [[ -L "${component}" ]] && { fail "symlinked path is not allowed"; return; }
    [[ -e "${component}" ]] || { fail "required path is absent"; return; }
  done
}

assert_owned_mode_type() {
  local path="$1"
  local expected_type="$2"
  local expected_mode="$3"
  local expected_uid="$4"
  local actual_type actual_uid actual_mode

  assert_no_symlink_path "${path}" || return
  actual_type="$(stat -c '%F' -- "${path}")" || { fail "cannot inspect runtime material"; return; }
  actual_uid="$(stat -c '%u' -- "${path}")" || { fail "cannot inspect runtime material"; return; }
  actual_mode="$(stat -c '%a' -- "${path}")" || { fail "cannot inspect runtime material"; return; }
  [[ "${actual_type}" == "${expected_type}" ]] || { fail "runtime material has wrong type"; return; }
  [[ "${actual_uid}" == "${expected_uid}" ]] || { fail "runtime material has wrong owner"; return; }
  [[ "${actual_mode}" == "${expected_mode}" ]] || { fail "runtime material has unsafe mode"; return; }
}

assert_owned_mode() {
  local path="$1"
  local expected_mode="$2"
  local expected_uid="$3"
  local expected_type="$4"

  assert_owned_mode_type "${path}" "${expected_type}" "${expected_mode}" "${expected_uid}"
}

assert_trusted_directory() {
  local path="$1"
  local expected_uid="$2"
  local actual_type actual_uid actual_mode

  assert_no_symlink_path "${path}" || return
  actual_type="$(stat -c '%F' -- "${path}")" || { fail "cannot inspect release directory"; return; }
  actual_uid="$(stat -c '%u' -- "${path}")" || { fail "cannot inspect release directory"; return; }
  actual_mode="$(stat -c '%a' -- "${path}")" || { fail "cannot inspect release directory"; return; }
  [[ "${actual_type}" == "directory" ]] || { fail "release path is not a directory"; return; }
  [[ "${actual_uid}" == "${expected_uid}" ]] || { fail "release path has wrong owner"; return; }
  (( (8#${actual_mode} & 8#22) == 0 )) || { fail "release path is group/world writable"; return; }
}

# Check each directory from a trusted root to a selected descendant. Checking
# only the endpoint would leave an attacker who can write an intermediate
# parent able to replace the endpoint after preflight.
assert_trusted_descendant_directories() {
  local root_dir="$1"
  local target_dir="$2"
  local expected_uid="$3"
  local relative_path component segment
  local -a segments

  [[ "${target_dir}" == "${root_dir}" || "${target_dir}" == "${root_dir}"/* ]] || {
    fail "trusted descendant escapes root"
    return
  }
  assert_trusted_directory "${root_dir}" "${expected_uid}" || return
  [[ "${target_dir}" == "${root_dir}" ]] && return
  relative_path="${target_dir#"${root_dir}"/}"
  component="${root_dir}"
  IFS=/ read -r -a segments <<< "${relative_path}"
  for segment in "${segments[@]}"; do
    [[ -n "${segment}" ]] || continue
    component="${component}/${segment}"
    assert_trusted_directory "${component}" "${expected_uid}" || return
  done
}

assert_trusted_regular_file() {
  local path="$1"
  local expected_uid="$2"
  local actual_type actual_uid actual_mode

  assert_no_symlink_path "${path}" || return
  actual_type="$(stat -c '%F' -- "${path}")" || { fail "cannot inspect release file"; return; }
  actual_uid="$(stat -c '%u' -- "${path}")" || { fail "cannot inspect release file"; return; }
  actual_mode="$(stat -c '%a' -- "${path}")" || { fail "cannot inspect release file"; return; }
  [[ "${actual_type}" == "regular file" ]] || { fail "release file has wrong type"; return; }
  [[ "${actual_uid}" == "${expected_uid}" ]] || { fail "release file has wrong owner"; return; }
  (( (8#${actual_mode} & 8#22) == 0 )) || { fail "release file is group/world writable"; return; }
}

# The complete imported package tree is checked, not only __init__.py. This
# prevents another local user from swapping a transitive module after current
# has been selected.
assert_trusted_package_tree() {
  local package_dir="$1"
  local expected_uid="$2"
  local entry

  assert_trusted_directory "${package_dir}" "${expected_uid}" || return
  while IFS= read -r -d '' entry; do
    [[ -L "${entry}" ]] && { fail "symlinked imported path is not allowed"; return; }
    if [[ -d "${entry}" ]]; then
      assert_trusted_directory "${entry}" "${expected_uid}" || return
    elif [[ -f "${entry}" ]]; then
      assert_trusted_regular_file "${entry}" "${expected_uid}" || return
    else
      fail "imported path has unsupported type"
      return
    fi
  done < <(find -P "${package_dir}" -print0)
}

validate_runtime_layout() {
  local runtime_dir="$1"
  local expected_uid="$2"
  local tls_dir="${runtime_dir}/tls"
  local cache_dir="${runtime_dir}/cache"

  assert_owned_mode "${runtime_dir}" 700 "${expected_uid}" directory || return
  assert_owned_mode "${tls_dir}" 700 "${expected_uid}" directory || return
  assert_owned_mode "${cache_dir}" 700 "${expected_uid}" directory || return
  assert_owned_mode "${tls_dir}/mage-vl-54.key" 600 "${expected_uid}" "regular file" || return
  assert_owned_mode "${runtime_dir}/mage-vl-shared.secret" 600 "${expected_uid}" "regular file" || return
  assert_owned_mode "${runtime_dir}/gpu0.lock" 600 "${expected_uid}" "regular file" || return
  assert_owned_mode "${tls_dir}/mage-vl-54.crt" 644 "${expected_uid}" "regular file"
}

validate_release_layout() {
  local current_link="$1"
  local releases_dir="$2"
  local expected_uid="$3"
  local current_parent="${current_link%/*}"
  local current_release resolved_releases_dir

  assert_trusted_directory "${current_parent}" "${expected_uid}" || return
  assert_trusted_directory "${releases_dir}" "${expected_uid}" || return
  [[ -L "${current_link}" ]] || { fail "current release selector must be a symlink"; return; }
  current_release="$(realpath -e -- "${current_link}")" || { fail "current release cannot be resolved"; return; }
  resolved_releases_dir="$(realpath -e -- "${releases_dir}")" || { fail "releases directory cannot be resolved"; return; }
  [[ "${current_release}" == "${resolved_releases_dir}"/* ]] || { fail "current release escapes releases"; return; }
  assert_trusted_directory "${current_release}" "${expected_uid}" || return
  assert_trusted_package_tree "${current_release}/live_operator" "${expected_uid}" || return
  printf '%s\n' "${current_release}"
}

main() {
  local expected_service_uid service_uid

  [[ "$#" == 0 ]] || { fail "arguments are not accepted"; return; }
  export PATH=/usr/bin:/bin
  expected_service_uid="$(id -u zty)" || { fail "zty user is absent"; return; }
  service_uid="$(id -u)"
  [[ "${service_uid}" == "${expected_service_uid}" ]] || { fail "must run as zty"; return; }

  # Project/SERVICE_ROOT/release components form the trusted code authority.
  assert_trusted_descendant_directories "${PROJECT_ROOT}" "${SERVICE_ROOT}" "${service_uid}"
  assert_trusted_descendant_directories "${SERVICE_ROOT}" "${RELEASES_DIR}" "${service_uid}"
  validate_runtime_layout "${RUNTIME_DIR}" "${service_uid}"
  CURRENT_RELEASE="$(validate_release_layout "${CURRENT_LINK}" "${RELEASES_DIR}" "${service_uid}")"

  [[ -d "${MODEL_DIR}" ]] || { fail "model directory is absent"; return; }
  [[ -x "${VENV_PYTHON}" ]] || { fail "dedicated virtualenv Python is absent"; return; }

  # Do not let a manager environment or an interactive shell change imports.
  unset PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE LD_PRELOAD
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
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
