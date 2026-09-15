"""Static contract for the dedicated Mage-VL service on 192.168.104.54.

The deployment files are deliberately self-contained: runtime secrets are
installed on the host and must never be represented by tracked configuration.
"""

from __future__ import annotations

from pathlib import Path


MIGRATION_DIR = Path(__file__).resolve().parents[1]
RUNNER = MIGRATION_DIR / "run_mage_vl_service_54.sh"
UNIT = MIGRATION_DIR / "jiankong-mage-vl-54.service"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_runner_pins_the_dedicated_host_layout_and_service_arguments() -> None:
    runner = _text(RUNNER)

    for value in (
        'PROJECT_ROOT="/home/zty/YL/JianKong"',
        'SERVICE_ROOT="${PROJECT_ROOT}/01_algorithms/mage_vl_service"',
        'RELEASES_DIR="${SERVICE_ROOT}/releases"',
        'CURRENT_LINK="${SERVICE_ROOT}/current"',
        'MODEL_DIR="${PROJECT_ROOT}/07_models/vlm_models/versions/v20260809_01_mage_vl_awq_int4"',
        'VENV_PYTHON="${PROJECT_ROOT}/08_envs/mage-vl-20260915/bin/python"',
        'RUNTIME_DIR="${SERVICE_ROOT}/runtime"',
        'TLS_CERT_FILE="${RUNTIME_DIR}/tls/mage-vl-54.crt"',
        'TLS_KEY_FILE="${RUNTIME_DIR}/tls/mage-vl-54.key"',
        'SHARED_SECRET_FILE="${RUNTIME_DIR}/mage-vl-shared.secret"',
        'CACHE_DIR="${RUNTIME_DIR}/cache"',
        'GPU_LOCK_FILE="${RUNTIME_DIR}/gpu0.lock"',
        'export CUDA_VISIBLE_DEVICES=0',
        'export PYTHONPATH="${CURRENT_RELEASE}"',
        'exec "${VENV_PYTHON}" -P -m live_operator.mage_vl_service',
        '--host "192.168.104.54"',
        '--port "8879"',
        '--model "${MODEL_DIR}"',
        '--model-version "mage-vl-awq-v20260809-1f7f5266fa4e-phone-use-prompt-v2"',
        '--gpu-weight-memory "3800MiB"',
        '--cpu-memory "24GiB"',
        '--max-request-bytes "67108864"',
        '--tls-cert-file "${TLS_CERT_FILE}"',
        '--tls-key-file "${TLS_KEY_FILE}"',
        '--shared-secret-file "${SHARED_SECRET_FILE}"',
        '--cache-dir "${CACHE_DIR}"',
        '--gpu-lock-file "${GPU_LOCK_FILE}"',
    ):
        assert value in runner


def test_runner_fails_closed_for_symlinked_or_unsafe_runtime_material() -> None:
    runner = _text(RUNNER)

    assert 'assert_no_symlink_path()' in runner
    assert '[[ -L "${component}" ]]' in runner
    assert 'assert_owned_mode()' in runner
    assert "stat -c '%u'" in runner
    assert "stat -c '%a'" in runner
    for path, mode in (
        ('"${RUNTIME_DIR}"', '700'),
        ('"${TLS_DIR}"', '700'),
        ('"${CACHE_DIR}"', '700'),
        ('"${TLS_KEY_FILE}"', '600'),
        ('"${SHARED_SECRET_FILE}"', '600'),
        ('"${GPU_LOCK_FILE}"', '600'),
        ('"${TLS_CERT_FILE}"', '644'),
    ):
        assert f"assert_owned_mode {path} {mode}" in runner


def test_runner_accepts_only_a_current_release_beneath_the_fixed_releases_root() -> None:
    runner = _text(RUNNER)

    assert 'CURRENT_RELEASE="$(realpath -e -- "${CURRENT_LINK}")"' in runner
    assert 'RESOLVED_RELEASES_DIR="$(realpath -e -- "${RELEASES_DIR}")"' in runner
    assert '"${CURRENT_RELEASE}" == "${RESOLVED_RELEASES_DIR}"/*' in runner
    assert 'assert_no_symlink_path "${CURRENT_RELEASE}"' in runner


def test_systemd_unit_is_a_single_purpose_non_public_service() -> None:
    unit = _text(UNIT)

    for value in (
        'User=zty',
        'Group=zty',
        'ExecStart=/home/zty/YL/JianKong/01_algorithms/mage_vl_migration/run_mage_vl_service_54.sh',
        'Restart=on-failure',
        'RestartSec=5s',
    ):
        assert value in unit
    assert '0.0.0.0' not in unit
    assert '8879' not in unit  # binding is a runner-only responsibility
    assert 'Environment=' not in unit
    assert 'EnvironmentFile=' not in unit
    assert 'LoadCredential=' not in unit
    assert 'secret=' not in unit.lower()
    assert 'password=' not in unit.lower()
    assert 'private_key=' not in unit.lower()
