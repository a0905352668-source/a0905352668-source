from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from live_operator.inference import (
    DeepStreamProcess,
    resolve_current_inference_version,
    restore_live_runtime_manifest,
    write_live_runtime_manifest,
)


CALIBRATIONS = (
    "camera_01_screen_calibration_v21.json",
    "camera_02_screen_calibration_v21.json",
    "camera_mechanical_01_screen_calibration_v21.json",
    "camera_mechanical_02_screen_calibration_v21.json",
    "camera_software_01_screen_calibration_v21.json",
    "camera_software_02_screen_calibration_v21.json",
    "camera_corridor_screen_calibration_v21.json",
)
MANIFEST_NAME = "live_runtime_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(tmp_path: Path, *, status: str = "active_user_approved") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {
        "binary": (tmp_path / "pipeline", b"binary"),
        "source": (tmp_path / "pipeline.cu", b"source"),
        "pose": (tmp_path / "pose.plan", b"pose"),
        "phone": (tmp_path / "phone.engine", b"phone"),
        "deepstream": (tmp_path / "jiankong_custom_pipeline", b"deepstream"),
        "deepstream_source": (
            tmp_path / "jiankong_custom_pipeline.cu",
            b"deepstream-source",
        ),
    }
    for path, content in files.values():
        path.write_bytes(content)
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    for index, name in enumerate(CALIBRATIONS):
        screens = [] if index == 0 else [[[0, 0], [1, 0], [1, 1]]]
        (calibration_dir / name).write_text(
            json.dumps({"screens": screens}), encoding="utf-8"
        )
    registry_payload = {
        "schema_version": 1,
        "active_version": "approved-v1",
        "status": status,
        "binary": str(files["binary"][0]),
        "binary_sha256": _sha256(files["binary"][0]),
        "source": str(files["source"][0]),
        "source_sha256": _sha256(files["source"][0]),
        "runtime": {
            "infer_fps": 8,
            "pose_plan": str(files["pose"][0]),
            "phone_engine": str(files["phone"][0]),
            "calibration_dir": str(calibration_dir),
        },
    }
    registry = tmp_path / "CURRENT_INFERENCE_VERSION.json"
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "active_version": "approved-v1",
        "deepstream": {
            "binary": str(files["deepstream"][0]),
            "binary_sha256": _sha256(files["deepstream"][0]),
            "source": str(files["deepstream_source"][0]),
            "source_sha256": _sha256(files["deepstream_source"][0]),
        },
        "artifact_sha256": {
            "pose_plan": _sha256(files["pose"][0]),
            "phone_engine": _sha256(files["phone"][0]),
            "calibrations": {
                name: _sha256(calibration_dir / name) for name in CALIBRATIONS
            },
        },
    }
    _manifest_path(registry).write_text(json.dumps(manifest), encoding="utf-8")
    _manifest_path(registry).chmod(0o600)
    return registry


def _manifest_path(registry: Path) -> Path:
    return registry.parent / MANIFEST_NAME


def _manifest(registry: Path) -> dict:
    return json.loads(_manifest_path(registry).read_text(encoding="utf-8"))


def _save_manifest(registry: Path, payload: dict) -> None:
    _manifest_path(registry).write_text(json.dumps(payload), encoding="utf-8")


def _resolve(registry: Path):
    return resolve_current_inference_version(registry, _manifest_path(registry))


def test_generates_atomic_private_sidecar_without_modifying_registry(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry_before = registry.read_bytes()
    old_manifest = _manifest_path(registry).read_bytes()
    deepstream = tmp_path / "new-deepstream"
    source = tmp_path / "new-deepstream.cu"
    deepstream.write_bytes(b"new binary")
    source.write_bytes(b"new source")

    result = write_live_runtime_manifest(
        deepstream,
        source,
        registry_path=registry,
        manifest_path=_manifest_path(registry),
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert registry.read_bytes() == registry_before
    assert payload["active_version"] == "approved-v1"
    assert payload["deepstream"] == {
        "binary": str(deepstream),
        "binary_sha256": _sha256(deepstream),
        "source": str(source),
        "source_sha256": _sha256(source),
    }
    if os.name == "posix":
        assert stat.S_IMODE(result.stat().st_mode) == 0o600
    assert not result.with_name(f".{result.name}.tmp").exists()
    assert result.with_suffix(result.suffix + ".bak").read_bytes() == old_manifest


def test_sidecar_backup_is_rollback_copy_on_replacement(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    first = _manifest_path(registry).read_bytes()
    deepstream = tmp_path / "replacement"
    source = tmp_path / "replacement.cu"
    deepstream.write_bytes(b"replacement")
    source.write_bytes(b"replacement source")
    write_live_runtime_manifest(
        deepstream, source, registry_path=registry, manifest_path=_manifest_path(registry)
    )
    assert _manifest_path(registry).with_suffix(".json.bak").read_bytes() == first


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-file contract")
def test_writer_ignores_predictable_symlink_temp_and_keeps_target_private(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    manifest = _manifest_path(registry)
    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    predictable = manifest.with_name(f".{manifest.name}.tmp")
    predictable.symlink_to(victim)
    permissive = manifest.with_name(f".{manifest.name}.bak.tmp")
    permissive.write_text("public sentinel", encoding="utf-8")
    permissive.chmod(0o644)
    deepstream = tmp_path / "new-bin"
    source = tmp_path / "new-source"
    deepstream.write_bytes(b"new")
    source.write_bytes(b"source")
    write_live_runtime_manifest(
        deepstream, source, registry_path=registry, manifest_path=manifest
    )
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert predictable.is_symlink()
    assert permissive.read_text(encoding="utf-8") == "public sentinel"
    assert stat.S_IMODE(permissive.stat().st_mode) == 0o644
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-file contract")
def test_resolver_rejects_symlink_and_non_private_sidecar(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    manifest = _manifest_path(registry)
    manifest.chmod(0o644)
    with pytest.raises(ValueError, match="permissions must be 0600"):
        _resolve(registry)
    manifest.chmod(0o600)
    real = manifest.with_name("real.json")
    manifest.replace(real)
    manifest.symlink_to(real)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _resolve(registry)


def test_restore_validates_backup_then_atomically_restores(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    original = _manifest_path(registry).read_bytes()
    replacement = tmp_path / "replacement"
    replacement_source = tmp_path / "replacement.cu"
    replacement.write_bytes(b"replacement")
    replacement_source.write_bytes(b"replacement source")
    write_live_runtime_manifest(
        replacement,
        replacement_source,
        registry_path=registry,
        manifest_path=_manifest_path(registry),
    )
    restored = restore_live_runtime_manifest(
        registry_path=registry, manifest_path=_manifest_path(registry)
    )
    assert restored.read_bytes() == original
    assert _resolve(registry).deepstream_binary.name == "jiankong_custom_pipeline"


def test_restore_rejects_invalid_backup_without_changing_current(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    manifest = _manifest_path(registry)
    current = manifest.read_bytes()
    backup = manifest.with_suffix(".json.bak")
    backup.write_text("{}", encoding="utf-8")
    if os.name == "posix":
        backup.chmod(0o600)
    with pytest.raises(ValueError):
        restore_live_runtime_manifest(registry_path=registry, manifest_path=manifest)
    assert manifest.read_bytes() == current


def test_resolves_approved_registry_with_matching_sidecar(tmp_path: Path) -> None:
    version = _resolve(_registry(tmp_path))
    assert version.active_version == "approved-v1"
    assert version.deepstream_binary.name == "jiankong_custom_pipeline"
    assert version.deepstream_source.name == "jiankong_custom_pipeline.cu"
    assert version.infer_fps == 8
    assert tuple(path.name for path in version.calibration_files) == CALIBRATIONS


def test_rejects_unapproved_or_mismatched_active_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="active_user_approved"):
        _resolve(_registry(tmp_path / "candidate", status="candidate"))
    registry = _registry(tmp_path / "mismatch")
    sidecar = _manifest(registry)
    sidecar["active_version"] = "different-version"
    _save_manifest(registry, sidecar)
    with pytest.raises(ValueError, match="active_version mismatch"):
        _resolve(registry)


def test_rejects_missing_sidecar_contract(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    sidecar = _manifest(registry)
    del sidecar["deepstream"]["binary"]
    _save_manifest(registry, sidecar)
    with pytest.raises(ValueError, match="deepstream.binary"):
        _resolve(registry)
    registry = _registry(tmp_path / "hashes")
    sidecar = _manifest(registry)
    del sidecar["artifact_sha256"]
    _save_manifest(registry, sidecar)
    with pytest.raises(ValueError, match="artifact_sha256"):
        _resolve(registry)


@pytest.mark.parametrize("asset", ("pose_plan", "phone_engine"))
def test_rejects_tampered_engine(tmp_path: Path, asset: str) -> None:
    registry = _registry(tmp_path)
    main = json.loads(registry.read_text(encoding="utf-8"))
    Path(main["runtime"][asset]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match=f"{asset} SHA256 mismatch"):
        _resolve(registry)


def test_rejects_tampered_deepstream_binary_source_and_calibration(tmp_path: Path) -> None:
    for key in ("binary", "source"):
        registry = _registry(tmp_path / key)
        Path(_manifest(registry)["deepstream"][key]).write_bytes(b"tampered")
        with pytest.raises(ValueError, match=f"deepstream_{key} SHA256 mismatch"):
            _resolve(registry)
    registry = _registry(tmp_path / "calibration")
    main = json.loads(registry.read_text(encoding="utf-8"))
    calibration = Path(main["runtime"]["calibration_dir"]) / CALIBRATIONS[3]
    calibration.write_text(json.dumps({"screens": []}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{CALIBRATIONS[3]} SHA256 mismatch"):
        _resolve(registry)


def test_parses_all_calibrations_and_enforces_screen_policy(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    main = json.loads(registry.read_text(encoding="utf-8"))
    sidecar = _manifest(registry)
    calibration_dir = Path(main["runtime"]["calibration_dir"])
    camera01 = calibration_dir / CALIBRATIONS[0]
    camera01.write_text(json.dumps({"screens": [[1]]}), encoding="utf-8")
    sidecar["artifact_sha256"]["calibrations"][CALIBRATIONS[0]] = _sha256(camera01)
    _save_manifest(registry, sidecar)
    with pytest.raises(ValueError, match="camera01.*empty screens"):
        _resolve(registry)

    registry = _registry(tmp_path / "other")
    main = json.loads(registry.read_text(encoding="utf-8"))
    sidecar = _manifest(registry)
    camera02 = Path(main["runtime"]["calibration_dir"]) / CALIBRATIONS[1]
    camera02.write_text(json.dumps({"screens": []}), encoding="utf-8")
    sidecar["artifact_sha256"]["calibrations"][CALIBRATIONS[1]] = _sha256(camera02)
    _save_manifest(registry, sidecar)
    with pytest.raises(ValueError, match=f"{CALIBRATIONS[1]}.*non-empty screens"):
        _resolve(registry)


class _FakeProcess:
    returncode = None
    terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_process_uses_exact_artifacts_and_scrubs_parent_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = _resolve(_registry(tmp_path))
    captured = {}
    fake = _FakeProcess()
    monkeypatch.setenv("CAMERA_PASSWORD", "DistinctFakePassword")
    monkeypatch.setenv("RTSP_URL", "rtsp://u:DistinctFakePassword@camera/101")
    monkeypatch.setenv("SECRET", "DistinctFakePassword")

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return fake

    process = DeepStreamProcess(
        version,
        tmp_path / "run",
        launcher="/opt/jiankong/run_container_50p2.sh",
        popen_factory=popen,
    ).start()
    assert captured["env"]["DEEPSTREAM_BINARY"] == str(version.deepstream_binary)
    assert captured["env"]["DEEPSTREAM_BINARY_SHA256"] == version.deepstream_binary_sha256
    assert "DistinctFakePassword" not in repr(captured)
    assert process.status()["state"] == "running"
    process.stop()
    assert process.status() == {"state": "exited", "returncode": 0}


def test_scripts_preserve_order_lock_and_authenticated_sidecar_fallback() -> None:
    root = Path(__file__).resolve().parents[2]
    inner = (root / "deepstream/custom_pipeline/scripts/run_7x8.sh").read_text(encoding="utf-8")
    host = (root / "deepstream/custom_pipeline/scripts/run_container_50p2.sh").read_text(encoding="utf-8")
    tokens = [
        f'--rtsp "{view}=${{RTSP_BASE}}/camera{index:02d}"'
        for index, view in enumerate(
            ("dianqi1", "dianqi2", "jixie1", "jixie2", "ruanjian1", "ruanjian2", "zoulang"),
            start=1,
        )
    ]
    assert [inner.index(token) for token in tokens] == sorted(inner.index(token) for token in tokens)
    assert "--no-video" in inner
    assert "/tmp/jiankong_gpu0.lock" in host and "flock -n" in host
    assert "LIVE_RUNTIME_MANIFEST" in host
    assert "live_runtime_manifest.json" in host
    assert "-m live_operator.inference resolve-shell" in host
    assert "MANIFEST_DEEPSTREAM_INVALID" in host
    assert '"${DEEPSTREAM_BINARY}:/workspace/build/jiankong_custom_pipeline:ro"' in host
    assert host.index("sha256sum -c -") < host.index('exec {gpu_lock_fd}>"${GPU_LOCK}"')
    assert '${BUILD_DIR}/jiankong_custom_pipeline' not in host


def test_wrapper_fallback_executes_full_resolver_before_gpu_lock(tmp_path: Path) -> None:
    def shell_path(path: Path) -> str:
        text = path.as_posix()
        return f"/{text[0].lower()}{text[2:]}" if text[1:3] == ":/" else text

    bash = shutil.which("bash")
    if bash is None:
        candidate = Path(r"C:\Program Files\Git\bin\bash.exe")
        if not candidate.exists():
            pytest.skip("bash unavailable")
        bash = str(candidate)
    registry = _registry(tmp_path)
    main = json.loads(registry.read_text(encoding="utf-8"))
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    lock_marker = tmp_path / "flock-called"
    flock = fake_bin / "flock"
    flock.write_text(
        f"#!/usr/bin/env bash\nprintf called > '{shell_path(lock_marker)}'\nexit 1\n",
        encoding="utf-8",
    )
    flock.chmod(0o755)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"
    host = Path(__file__).resolve().parents[2] / "deepstream/custom_pipeline/scripts/run_container_50p2.sh"
    env = dict(os.environ)
    env.update({
        "PATH": f"{shell_path(fake_bin)}:/usr/bin:/bin",
        "PYTHON_BIN": Path(sys.executable).as_posix(),
        "LIVE_OPERATOR_PYTHONPATH": Path(__file__).resolve().parents[2].as_posix(),
        "CURRENT_INFERENCE_REGISTRY": registry.as_posix(),
        "LIVE_RUNTIME_MANIFEST": _manifest_path(registry).as_posix(),
        "SOURCE_DIR": shell_path(source_dir),
        "OUTPUT_DIR": shell_path(output_dir),
        "GPU_LOCK": shell_path(tmp_path / "gpu.lock"),
    })
    completed = subprocess.run([bash, host.as_posix()], env=env, capture_output=True, text=True)
    assert completed.returncode == 75
    assert lock_marker.exists()

    lock_marker.unlink()
    Path(main["runtime"]["pose_plan"]).write_bytes(b"tampered")
    completed = subprocess.run([bash, host.as_posix()], env=env, capture_output=True, text=True)
    assert completed.returncode == 2
    assert not lock_marker.exists()

    cases = ("phone", "deepstream", "calibration", "active_version")
    for case in cases:
        case_registry = _registry(tmp_path / case)
        case_main = json.loads(case_registry.read_text(encoding="utf-8"))
        case_sidecar = _manifest(case_registry)
        env["CURRENT_INFERENCE_REGISTRY"] = case_registry.as_posix()
        env["LIVE_RUNTIME_MANIFEST"] = _manifest_path(case_registry).as_posix()
        if case == "phone":
            Path(case_main["runtime"]["phone_engine"]).write_bytes(b"tampered")
        elif case == "deepstream":
            Path(case_sidecar["deepstream"]["binary"]).write_bytes(b"tampered")
        elif case == "calibration":
            calibration = Path(case_main["runtime"]["calibration_dir"]) / CALIBRATIONS[4]
            calibration.write_text("{}", encoding="utf-8")
        else:
            case_sidecar["active_version"] = "mismatch"
            _save_manifest(case_registry, case_sidecar)
        completed = subprocess.run(
            [bash, host.as_posix()], env=env, capture_output=True, text=True
        )
        assert completed.returncode == 2, case
        assert not lock_marker.exists(), case

    registry = _registry(tmp_path / "explicit-bypass")
    arbitrary = tmp_path / "arbitrary-pipeline"
    arbitrary.write_bytes(b"self-consistent but not approved")
    env.update(
        {
            "CURRENT_INFERENCE_REGISTRY": registry.as_posix(),
            "LIVE_RUNTIME_MANIFEST": _manifest_path(registry).as_posix(),
            "DEEPSTREAM_BINARY": shell_path(arbitrary),
            "DEEPSTREAM_BINARY_SHA256": _sha256(arbitrary),
            "POSE_ENGINE": shell_path(arbitrary),
            "PHONE_ENGINE": shell_path(arbitrary),
            "CALIB_DIR": shell_path(tmp_path),
            "INFER_FPS": "8",
        }
    )
    completed = subprocess.run(
        [bash, host.as_posix()], env=env, capture_output=True, text=True
    )
    assert completed.returncode == 2
    assert "EXPLICIT_RUNTIME_IDENTITY_MISMATCH" in completed.stderr
    assert not lock_marker.exists()
