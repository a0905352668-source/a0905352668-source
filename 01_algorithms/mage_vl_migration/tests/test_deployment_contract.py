"""Static contract for the dedicated Mage-VL service on 192.168.104.54.

The deployment files are deliberately self-contained: runtime secrets are
installed on the host and must never be represented by tracked configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


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
    assert 'assert_owned_mode_type()' in runner
    assert 'assert_trusted_directory()' in runner
    assert 'assert_trusted_descendant_directories()' in runner
    assert 'assert_trusted_package_tree()' in runner
    assert "stat -c '%u'" in runner
    assert "stat -c '%a'" in runner
    assert "stat -c '%f'" in runner
    assert "stat -c '%F'" not in runner
    assert 'S_IFMT=8#170000' in runner
    assert 'S_IFREG=8#100000' in runner
    assert 'S_IFDIR=8#40000' in runner
    assert 'assert_trusted_descendant_directories "${PROJECT_ROOT}" "${SERVICE_ROOT}" "${service_uid}"' in runner
    for path, mode, type_bits in (
        ('"${runtime_dir}"', '700', '"${S_IFDIR}"'),
        ('"${tls_dir}"', '700', '"${S_IFDIR}"'),
        ('"${cache_dir}"', '700', '"${S_IFDIR}"'),
        ('"${tls_dir}/mage-vl-54.key"', '600', '"${S_IFREG}"'),
        ('"${runtime_dir}/mage-vl-shared.secret"', '600', '"${S_IFREG}"'),
        ('"${runtime_dir}/gpu0.lock"', '600', '"${S_IFREG}"'),
        ('"${tls_dir}/mage-vl-54.crt"', '644', '"${S_IFREG}"'),
    ):
        assert f"assert_owned_mode {path} {mode}" in runner
        assert type_bits in runner


def test_runner_accepts_only_a_current_release_beneath_the_fixed_releases_root() -> None:
    runner = _text(RUNNER)

    assert 'current_release="$(realpath -e -- "${current_link}")"' in runner
    assert 'resolved_releases_dir="$(realpath -e -- "${releases_dir}")"' in runner
    assert '"${current_release}" == "${resolved_releases_dir}"/*' in runner
    assert 'assert_trusted_directory "${current_release}" "${expected_uid}"' in runner


def test_systemd_unit_is_a_single_purpose_non_public_service() -> None:
    unit = _text(UNIT)

    for value in (
        'User=zty',
        'Group=zty',
        'ExecStart=/home/zty/YL/JianKong/01_algorithms/mage_vl_migration/run_mage_vl_service_54.sh',
        'Restart=on-failure',
        'RestartSec=5s',
        'RestartPreventExitStatus=78',
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


linux_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason="runner validation uses the GNU/Linux deployment metadata contract",
)


def _shell_validate(function: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; "$2" "${@:3}"',
            "deployment-contract",
            str(RUNNER),
            function,
            *args,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _chmod(path: Path, mode: int) -> None:
    path.chmod(mode)


def _safe_release_fixture(tmp_path: Path) -> tuple[Path, Path]:
    releases = tmp_path / "service" / "releases"
    release = releases / "approved-release"
    package = release / "live_operator" / "nested"
    package.mkdir(parents=True)
    init_file = release / "live_operator" / "__init__.py"
    service_file = package / "mage_vl_service.py"
    init_file.write_text("", encoding="utf-8")
    service_file.write_text("", encoding="utf-8")
    _chmod(init_file, 0o644)
    _chmod(service_file, 0o755)
    for directory in (tmp_path / "service", releases, release, release / "live_operator", package):
        _chmod(directory, 0o755)
    current = tmp_path / "service" / "current"
    current.symlink_to(releases / "approved-release")
    return current, releases


def _safe_runtime_fixture(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    tls = runtime / "tls"
    cache = runtime / "cache"
    tls.mkdir(parents=True)
    cache.mkdir()
    (tls / "mage-vl-54.crt").write_text("certificate", encoding="utf-8")
    (tls / "mage-vl-54.key").write_text("key", encoding="utf-8")
    (runtime / "mage-vl-shared.secret").write_text("secret", encoding="utf-8")
    (runtime / "gpu0.lock").write_text("", encoding="utf-8")
    for directory in (runtime, tls, cache):
        _chmod(directory, 0o700)
    _chmod(tls / "mage-vl-54.crt", 0o644)
    for private_file in (
        tls / "mage-vl-54.key",
        runtime / "mage-vl-shared.secret",
        runtime / "gpu0.lock",
    ):
        _chmod(private_file, 0o600)
    return runtime


@linux_only
def test_linux_validation_accepts_safe_release_and_runtime_fixtures(tmp_path: Path) -> None:
    current, releases = _safe_release_fixture(tmp_path)
    runtime = _safe_runtime_fixture(tmp_path)
    uid = str(os.getuid())

    assert _shell_validate("validate_release_layout", str(current), str(releases), uid).returncode == 0
    assert _shell_validate("validate_runtime_layout", str(runtime), uid).returncode == 0


@linux_only
def test_linux_validation_accepts_an_empty_regular_gpu_lock(tmp_path: Path) -> None:
    runtime = _safe_runtime_fixture(tmp_path)
    lock = runtime / "gpu0.lock"

    assert lock.read_bytes() == b""
    assert _shell_validate("validate_runtime_layout", str(runtime), str(os.getuid())).returncode == 0


@linux_only
def test_linux_validation_rejects_symlinked_runtime_material(tmp_path: Path) -> None:
    runtime = _safe_runtime_fixture(tmp_path)
    (runtime / "mage-vl-shared.secret").unlink()
    (runtime / "mage-vl-shared.secret").symlink_to(runtime / "gpu0.lock")

    result = _shell_validate("validate_runtime_layout", str(runtime), str(os.getuid()))

    assert result.returncode == 78


@linux_only
@pytest.mark.parametrize(
    ("path", "setup"),
    [
        ("mage-vl-shared.secret", lambda path: _chmod(path, 0o640)),
        ("gpu0.lock", lambda path: (path.unlink(), path.mkdir())),
    ],
)
def test_linux_validation_rejects_wrong_runtime_mode_or_type(
    tmp_path: Path, path: str, setup: object
) -> None:
    runtime = _safe_runtime_fixture(tmp_path)
    target = runtime / path
    setup(target)  # type: ignore[operator]

    result = _shell_validate("validate_runtime_layout", str(runtime), str(os.getuid()))

    assert result.returncode == 78


@linux_only
def test_linux_validation_rejects_current_release_outside_releases(tmp_path: Path) -> None:
    current, releases = _safe_release_fixture(tmp_path)
    outside = tmp_path / "outside-release" / "live_operator"
    outside.mkdir(parents=True)
    (outside / "__init__.py").write_text("", encoding="utf-8")
    _chmod(outside.parent, 0o755)
    _chmod(outside, 0o755)
    current.unlink()
    current.symlink_to(outside.parent)

    result = _shell_validate("validate_release_layout", str(current), str(releases), str(os.getuid()))

    assert result.returncode == 78


@linux_only
def test_linux_validation_rejects_unavailable_expected_owner(tmp_path: Path) -> None:
    current, releases = _safe_release_fixture(tmp_path)

    result = _shell_validate(
        "validate_release_layout", str(current), str(releases), str(os.getuid() + 1)
    )

    assert result.returncode == 78
