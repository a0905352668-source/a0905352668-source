"""Resolve and launch the user-approved live inference version."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from live_operator.config import DEFAULT_CONFIG_PATH, LiveConfig


DEFAULT_REGISTRY_PATH = Path(
    "/media/boshi/Data/JianKong/01_algorithms/CURRENT_INFERENCE_VERSION.json"
)
DEFAULT_MANIFEST_PATH = Path(
    "/media/boshi/Data/JianKong/02_configs/runtime/live_runtime_manifest.json"
)
DEFAULT_LAUNCHER = Path(
    "/media/boshi/Data/JianKong/00_staging/deepstream_7x8_20260714/"
    "source/deepstream/custom_pipeline/scripts/run_container_50p2.sh"
)
CALIBRATION_NAMES = (
    "camera_01_screen_calibration_v21.json",
    "camera_02_screen_calibration_v21.json",
    "camera_mechanical_01_screen_calibration_v21.json",
    "camera_mechanical_02_screen_calibration_v21.json",
    "camera_software_01_screen_calibration_v21.json",
    "camera_software_02_screen_calibration_v21.json",
    "camera_corridor_screen_calibration_v21.json",
)


@dataclass(frozen=True)
class InferenceVersion:
    active_version: str
    binary: Path
    source: Path
    deepstream_binary: Path
    deepstream_binary_sha256: str
    deepstream_source: Path
    deepstream_source_sha256: str
    pose_plan: Path
    phone_engine: Path
    calibration_dir: Path
    calibration_files: tuple[Path, ...]
    infer_fps: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _read_private_bytes(path: Path, label: str) -> bytes:
    if os.name != "posix":
        try:
            return path.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid {label}: {path}") from error
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise ValueError(f"{label} permissions must be 0600: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError(f"{label} permissions must be 0600: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while opening: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_private_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(_read_private_bytes(path, label).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _required_text(payload: Mapping[str, Any], key: str, prefix: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid field: {prefix}{key}")
    return value


def _required_mapping(
    payload: Mapping[str, Any], key: str, prefix: str = ""
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid field: {prefix}{key}")
    return value


def _required_file(path_text: str, label: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if _sha256(path) != expected.lower():
        raise ValueError(f"{label} SHA256 mismatch: {path}")


def _approved_registry(path: Path) -> dict[str, Any]:
    payload = _load_object(path, "inference registry")
    if payload.get("status") != "active_user_approved":
        raise ValueError("registry status must be active_user_approved")
    _required_text(payload, "active_version")
    return payload


def _runtime_paths(registry: Mapping[str, Any]) -> tuple[int, Path, Path, Path]:
    runtime = _required_mapping(registry, "runtime")
    infer_fps = runtime.get("infer_fps")
    if infer_fps not in {8, 10}:
        raise ValueError("runtime.infer_fps must be 8 or 10")
    pose_plan = _required_file(
        _required_text(runtime, "pose_plan", "runtime."), "pose_plan"
    )
    phone_engine = _required_file(
        _required_text(runtime, "phone_engine", "runtime."), "phone_engine"
    )
    calibration_dir = Path(
        _required_text(runtime, "calibration_dir", "runtime.")
    )
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"missing calibration_dir: {calibration_dir}")
    return infer_fps, pose_plan, phone_engine, calibration_dir


def _legacy_calibrations(calibration_dir: Path) -> tuple[Path, ...]:
    calibrations = tuple(calibration_dir / name for name in CALIBRATION_NAMES)
    for calibration in calibrations:
        if not calibration.is_file():
            raise FileNotFoundError(f"missing calibration: {calibration}")
    return calibrations


def _configured_calibrations(
    calibration_dir: Path, config_path: str | os.PathLike[str] | None
) -> tuple[tuple[Path, ...], LiveConfig | None]:
    """Resolve exactly the calibrations used by enabled cameras.

    Older runtime manifests did not contain a camera inventory.  They retain
    their original seven-file validation path until a management-page publish
    writes a v2 manifest.
    """

    if config_path is None or not Path(config_path).is_file():
        return _legacy_calibrations(calibration_dir), None
    config = LiveConfig.load(config_path)
    calibrations: list[Path] = []
    root = calibration_dir.resolve()
    for camera in config.enabled_cameras:
        candidate = (root / camera.calibration).resolve()
        if candidate.parent != root or not candidate.is_file():
            raise FileNotFoundError(
                f"missing calibration for {camera.relay}: {candidate}"
            )
        _validate_calibration(candidate, expects_screens=camera.has_screen)
        calibrations.append(candidate)
    if not calibrations:
        raise ValueError("live configuration has no enabled camera calibrations")
    return tuple(calibrations), config


def _validate_calibration(calibration: Path, *, expects_screens: bool | None = None) -> None:
    payload = _load_object(calibration, f"calibration JSON {calibration.name}")
    screens = payload.get("screens")
    if not isinstance(screens, list):
        raise ValueError(f"{calibration.name} must contain an explicit screens array")
    if expects_screens is None:
        # Compatibility for v1 manifests. The original contract intentionally
        # validates only the seven approved files and their empty/non-empty
        # screen policy; polygon validation is performed for v2 inventories.
        if calibration.name == CALIBRATION_NAMES[0]:
            if screens:
                raise ValueError("camera01 calibration must contain explicit empty screens")
        elif not screens:
            raise ValueError(f"{calibration.name} must contain non-empty screens")
        return
    if expects_screens is True and not screens:
        raise ValueError(f"{calibration.name} must contain at least one screen")
    if expects_screens is False and screens:
        raise ValueError(f"{calibration.name} is configured without screens")
    for index, screen in enumerate(screens, start=1):
        if not isinstance(screen, Mapping):
            raise ValueError(f"invalid screen {index} in {calibration.name}")
        polygon = screen.get("screen_poly", screen.get("points"))
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"invalid screen polygon {index} in {calibration.name}")


def _atomic_private_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("manifest temporary must be a regular file")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            old_bytes = _read_private_bytes(path, "live runtime manifest")
            backup_descriptor, backup_text = tempfile.mkstemp(
                prefix=f".{backup.name}.", suffix=".tmp", dir=path.parent
            )
            backup_temporary = Path(backup_text)
            try:
                if os.name == "posix":
                    os.fchmod(backup_descriptor, 0o600)
                if not stat.S_ISREG(os.fstat(backup_descriptor).st_mode):
                    raise ValueError("manifest backup temporary must be regular")
                with os.fdopen(backup_descriptor, "wb") as handle:
                    backup_descriptor = -1
                    handle.write(old_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(backup_temporary, backup)
                os.chmod(backup, 0o600)
            finally:
                if backup_descriptor >= 0:
                    os.close(backup_descriptor)
                backup_temporary.unlink(missing_ok=True)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        return path
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_live_runtime_manifest(
    deepstream_binary: str | os.PathLike[str],
    deepstream_source: str | os.PathLike[str],
    *,
    registry_path: str | os.PathLike[str] = DEFAULT_REGISTRY_PATH,
    manifest_path: str | os.PathLike[str] = DEFAULT_MANIFEST_PATH,
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
) -> Path:
    """Atomically write a private runtime sidecar without changing approval."""

    registry = _approved_registry(Path(registry_path))
    _infer_fps, pose, phone, calibration_dir = _runtime_paths(registry)
    calibrations, config = _configured_calibrations(calibration_dir, config_path)
    for calibration in calibrations:
        _validate_calibration(calibration)
    binary = _required_file(str(deepstream_binary), "deepstream_binary")
    source = _required_file(str(deepstream_source), "deepstream_source")
    payload = {
        "schema_version": 2,
        "active_version": _required_text(registry, "active_version"),
        "deepstream": {
            "binary": str(binary),
            "binary_sha256": _sha256(binary),
            "source": str(source),
            "source_sha256": _sha256(source),
        },
        "artifact_sha256": {
            "pose_plan": _sha256(pose),
            "phone_engine": _sha256(phone),
            "calibrations": {
                calibration.name: _sha256(calibration)
                for calibration in calibrations
            },
        },
    }
    if config is not None:
        config_file = Path(config_path)
        payload["camera_config"] = {
            "path": str(config_file),
            "sha256": _sha256(config_file),
            "enabled_relays": [camera.relay for camera in config.enabled_cameras],
            "calibrations": [camera.calibration for camera in config.enabled_cameras],
        }
    return _atomic_private_json(Path(manifest_path), payload)


def resolve_current_inference_version(
    registry_path: str | os.PathLike[str] = DEFAULT_REGISTRY_PATH,
    manifest_path: str | os.PathLike[str] = DEFAULT_MANIFEST_PATH,
) -> InferenceVersion:
    """Validate approved registry plus sidecar without acquiring the GPU."""

    registry = _approved_registry(Path(registry_path))
    manifest = _load_private_object(Path(manifest_path), "live runtime manifest")
    active_version = _required_text(registry, "active_version")
    if _required_text(manifest, "active_version") != active_version:
        raise ValueError("registry and live runtime manifest active_version mismatch")

    binary = _required_file(_required_text(registry, "binary"), "binary")
    source = _required_file(_required_text(registry, "source"), "source")
    _verify_hash(binary, _required_text(registry, "binary_sha256"), "binary")
    _verify_hash(source, _required_text(registry, "source_sha256"), "source")
    infer_fps, pose, phone, calibration_dir = _runtime_paths(registry)
    camera_config = manifest.get("camera_config")
    if camera_config is None:
        calibrations = _legacy_calibrations(calibration_dir)
        configured = None
    else:
        if not isinstance(camera_config, Mapping):
            raise ValueError("invalid field: camera_config")
        config_path = Path(_required_text(camera_config, "path", "camera_config."))
        _verify_hash(
            config_path,
            _required_text(camera_config, "sha256", "camera_config."),
            "camera_config",
        )
        calibrations, configured = _configured_calibrations(calibration_dir, config_path)
        expected_relays = camera_config.get("enabled_relays")
        expected_calibrations = camera_config.get("calibrations")
        if (
            not isinstance(expected_relays, list)
            or not isinstance(expected_calibrations, list)
            or configured is None
            or expected_relays != [camera.relay for camera in configured.enabled_cameras]
            or expected_calibrations != [camera.calibration for camera in configured.enabled_cameras]
        ):
            raise ValueError("camera configuration changed after runtime manifest publish")

    deepstream = _required_mapping(manifest, "deepstream")
    deepstream_binary = _required_file(
        _required_text(deepstream, "binary", "deepstream."), "deepstream_binary"
    )
    deepstream_binary_sha256 = _required_text(
        deepstream, "binary_sha256", "deepstream."
    ).lower()
    _verify_hash(
        deepstream_binary, deepstream_binary_sha256, "deepstream_binary"
    )
    deepstream_source = _required_file(
        _required_text(deepstream, "source", "deepstream."), "deepstream_source"
    )
    deepstream_source_sha256 = _required_text(
        deepstream, "source_sha256", "deepstream."
    ).lower()
    _verify_hash(
        deepstream_source, deepstream_source_sha256, "deepstream_source"
    )

    artifact_hashes = _required_mapping(manifest, "artifact_sha256")
    _verify_hash(
        pose,
        _required_text(artifact_hashes, "pose_plan", "artifact_sha256."),
        "pose_plan",
    )
    _verify_hash(
        phone,
        _required_text(artifact_hashes, "phone_engine", "artifact_sha256."),
        "phone_engine",
    )
    calibration_hashes = _required_mapping(
        artifact_hashes, "calibrations", "artifact_sha256."
    )
    if set(calibration_hashes) != {calibration.name for calibration in calibrations}:
        raise ValueError("artifact_sha256.calibrations does not match enabled cameras")
    for calibration in calibrations:
        _verify_hash(
            calibration,
            _required_text(
                calibration_hashes,
                calibration.name,
                "artifact_sha256.calibrations.",
            ),
            calibration.name,
        )
        _validate_calibration(calibration)

    return InferenceVersion(
        active_version,
        binary,
        source,
        deepstream_binary,
        deepstream_binary_sha256,
        deepstream_source,
        deepstream_source_sha256,
        pose,
        phone,
        calibration_dir,
        calibrations,
        infer_fps,
    )


def restore_live_runtime_manifest(
    *,
    registry_path: str | os.PathLike[str] = DEFAULT_REGISTRY_PATH,
    manifest_path: str | os.PathLike[str] = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Validate and atomically restore the private `.bak` sidecar."""

    destination = Path(manifest_path)
    backup = destination.with_suffix(destination.suffix + ".bak")
    resolve_current_inference_version(registry_path, backup)
    backup_bytes = _read_private_bytes(backup, "live runtime manifest backup")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".restore", dir=destination.parent
    )
    temporary = Path(temporary_text)
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(backup_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        return destination
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _print_resolved_shell_values(registry_path: str, manifest_path: str) -> None:
    version = resolve_current_inference_version(registry_path, manifest_path)
    values = (
        version.deepstream_binary,
        version.deepstream_binary_sha256,
        version.pose_plan,
        version.phone_engine,
        version.calibration_dir,
        str(version.infer_fps),
    )
    encoded_values = []
    for value in values:
        text = value.as_posix() if isinstance(value, Path) else str(value)
        if os.name == "nt" and len(text) >= 3 and text[1:3] == ":/":
            text = f"/{text[0].lower()}{text[2:]}"
        if "\n" in text or "\r" in text or not text:
            raise ValueError("resolved shell value contains an unsafe newline")
        encoded_values.append(text.encode("utf-8"))
    sys.stdout.buffer.write(b"\n".join(encoded_values) + b"\n")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "resolve-shell":
        _print_resolved_shell_values(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: python -m live_operator.inference resolve-shell REGISTRY MANIFEST")


class DeepStreamProcess:
    """Own the host launcher; the host script remains responsible for the GPU lock."""

    def __init__(
        self,
        version: InferenceVersion,
        run_dir: str | os.PathLike[str],
        *,
        relay_base: str = "rtsp://127.0.0.1:8554",
        launcher: str | os.PathLike[str] = DEFAULT_LAUNCHER,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        if relay_base != "rtsp://127.0.0.1:8554":
            raise ValueError("live operator relay_base must be local MediaMTX")
        self.version = version
        self.run_dir = Path(run_dir)
        self.relay_base = relay_base
        self.launcher = os.fspath(launcher)
        self._popen_factory = popen_factory
        self._process: Any | None = None

    def start(self) -> "DeepStreamProcess":
        if self._process is not None and self._process.poll() is None:
            return self
        output_dir = self.run_dir / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)
        allowed_parent_keys = (
            "PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL",
            "DOCKER_HOST", "NVIDIA_RUNTIME",
        )
        environment = {
            key: os.environ[key] for key in allowed_parent_keys if key in os.environ
        }
        environment.update(
            {
                "LOCAL_RELAY_BASE": self.relay_base,
                "RTSP_BASE": self.relay_base,
                "POSE_ENGINE": str(self.version.pose_plan),
                "PHONE_ENGINE": str(self.version.phone_engine),
                "CALIB_DIR": str(self.version.calibration_dir),
                "OUTPUT_DIR": str(output_dir),
                "INFER_FPS": str(self.version.infer_fps),
                "DEEPSTREAM_BINARY": str(self.version.deepstream_binary),
                "DEEPSTREAM_BINARY_SHA256": self.version.deepstream_binary_sha256,
            }
        )
        self._process = self._popen_factory([self.launcher], env=environment)
        return self

    def status(self) -> dict[str, int | str | None]:
        if self._process is None:
            return {"state": "not_started", "returncode": None}
        returncode = self._process.poll()
        return {
            "state": "running" if returncode is None else "exited",
            "returncode": returncode,
        }

    def stop(self, timeout: float = 10.0) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=timeout)
