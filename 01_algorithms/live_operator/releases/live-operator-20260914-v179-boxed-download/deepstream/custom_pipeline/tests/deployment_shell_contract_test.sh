#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INNER_RUN="${ROOT}/scripts/run_7x8.sh"
HOST_BUILD="${ROOT}/scripts/build_container_50p2.sh"
HOST_RUN="${ROOT}/scripts/run_container_50p2.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

grep -Fq '/media/boshi/Data/JianKong' "${HOST_BUILD}" || fail "canonical root absent from build wrapper"
grep -Fq '/media/boshi/Data/JianKong' "${HOST_RUN}" || fail "canonical root absent from run wrapper"
if grep -Fq '/media/boshi/Data/00_active_projects/JianKong' "${HOST_BUILD}" "${HOST_RUN}"; then
  fail "legacy project root remains in host wrappers"
fi

for wrapper in "${HOST_BUILD}" "${HOST_RUN}"; do
  for mount in \
    '/workspace/source' \
    '/workspace/build' \
    '/models/pose960_static_b7.plan' \
    '/models/phone_b16_img512_fp16_trt86.engine' \
    '/configs/calibration' \
    '/output'; do
    grep -Fq "${mount}" "${wrapper}" || fail "missing mount ${mount} in ${wrapper}"
  done
  grep -Fq -- '--runtime="${NVIDIA_RUNTIME:-nvidia}"' "${wrapper}" || fail "NVIDIA runtime absent from ${wrapper}"
  grep -Fq -- '--network host' "${wrapper}" || fail "--network host absent from ${wrapper}"
  grep -Fq 'env -u USE_NEW_NVSTREAMMUX' "${wrapper}" || fail "nvstreammux env is not unset in ${wrapper}"
done

grep -Fq '/tmp/jiankong_gpu0.lock' "${HOST_RUN}" || fail "host run wrapper does not own GPU lock"
grep -Fq 'GPU_LOCK_BUSY=' "${HOST_RUN}" || fail "host run wrapper has no busy-lock diagnostic"
grep -Fq -- '--label "jiankong.owner=${JIAN_KONG_OWNER_TOKEN}"' "${HOST_RUN}" || fail "owner label absent from docker run"
grep -Fq 'JIAN_KONG_OWNER_TOKEN_INVALID=' "${HOST_RUN}" || fail "owner token validation absent from run wrapper"
owner_error="$(mktemp)"
if JIAN_KONG_OWNER_TOKEN='unsafe-token' bash "${HOST_RUN}" >/dev/null 2>"${owner_error}"; then
  rm -f "${owner_error}"
  fail "unsafe owner token was accepted"
fi
grep -Fq 'JIAN_KONG_OWNER_TOKEN_INVALID=unsafe-token' "${owner_error}" || fail "unsafe owner token failure is not explicit"
rm -f "${owner_error}"
if grep -Fq '/tmp/jiankong_gpu0.lock' "${INNER_RUN}"; then
  fail "container-internal runner must not reacquire the host GPU lock"
fi
if grep -Eq '(^|[[:space:]])flock([[:space:]]|$)' "${INNER_RUN}"; then
  fail "container-internal runner contains flock"
fi

echo "PASS deployment_shell_contract_test"
