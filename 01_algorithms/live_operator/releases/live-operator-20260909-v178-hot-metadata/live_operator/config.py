"""Private, versioned live-camera configuration.

The first production deployment stored one shared credential and seven fixed
camera names.  That representation is intentionally still accepted so an
already-running site can be upgraded without retyping credentials.  New files
use the v2 camera list: every camera has its own connection parameters and a
calibration file, so adding or retiring a camera does not require source edits.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


DEFAULT_CONFIG_PATH = Path(
    "/media/boshi/Data/JianKong/02_configs/runtime/live_operator.json"
)
RTSP_PATH = "/Streaming/Channels/101"
CONFIG_SCHEMA_VERSION = 2

# Kept only for automatic migration of the original seven-route configuration.
_LEGACY_ROUTES = (
    ("camera01", "dianqi1", "camera_01_screen_calibration_v21.json", False),
    ("camera02", "dianqi2", "camera_02_screen_calibration_v21.json", True),
    ("camera03", "jixie1", "camera_mechanical_01_screen_calibration_v21.json", True),
    ("camera04", "jixie2", "camera_mechanical_02_screen_calibration_v21.json", True),
    ("camera05", "ruanjian1", "camera_software_01_screen_calibration_v21.json", True),
    ("camera06", "ruanjian2", "camera_software_02_screen_calibration_v21.json", True),
    ("camera07", "zoulang", "camera_corridor_screen_calibration_v21.json", True),
)

# The operator is a single long-lived process. Retaining every distinct
# credential for that process lifetime covers delayed/asynchronous log messages
# and lets redact_text reliably hide individual camera credentials.
_SECRETS: set[str] = set()
_SECRETS_LOCK = threading.RLock()
_IS_POSIX = os.name == "posix"
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
_RELAY = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
_CALIBRATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.json")


def _register_credentials(username: str, password: str) -> None:
    encoded_username = quote(username, safe="")
    encoded_password = quote(password, safe="")
    with _SECRETS_LOCK:
        _SECRETS.update(
            secret
            for secret in (
                password,
                encoded_password,
                f"{username}:{password}",
                f"{encoded_username}:{encoded_password}",
            )
            if secret
        )


def redact_text(text: str) -> str:
    """Replace registered passwords and RTSP userinfo with a fixed marker."""

    redacted = str(text)
    with _SECRETS_LOCK:
        secrets = sorted(_SECRETS, key=len, reverse=True)
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"missing or invalid field: {field_name}")
    return value


def _validate_host(value: str, field_name: str) -> None:
    if value != value.strip() or len(value) > 253:
        raise ValueError(f"invalid field: {field_name}")
    if any(ord(character) < 33 or ord(character) == 127 for character in value):
        raise ValueError(f"invalid field: {field_name}")
    labels = value.split(".")
    if not labels or any(_DNS_LABEL.fullmatch(label) is None for label in labels):
        raise ValueError(f"invalid field: {field_name}")
    if all(label.isdigit() for label in labels):
        try:
            ipaddress.IPv4Address(value)
        except ipaddress.AddressValueError as error:
            raise ValueError(f"invalid field: {field_name}") from error


def _validate_posix_config_file(file_status: os.stat_result) -> None:
    if not stat.S_ISREG(file_status.st_mode):
        raise ValueError("configuration must be a regular file")
    if stat.S_IMODE(file_status.st_mode) != 0o600:
        raise ValueError("configuration permissions must be 0600")


def _validate_label(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 80:
        raise ValueError(f"invalid field: {field_name}")
    if any(ord(character) < 32 or character in "/\\" for character in value):
        raise ValueError(f"invalid field: {field_name}")


@dataclass(frozen=True)
class CameraConfig:
    """One camera connection and its versioned screen calibration."""

    relay: str
    view: str
    ip: str
    calibration: str
    has_screen: bool = True
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    port: int = 554
    path: str = RTSP_PATH
    enabled: bool = True

    def __post_init__(self) -> None:
        if _RELAY.fullmatch(self.relay) is None:
            raise ValueError("invalid field: cameras.relay")
        _validate_label(self.view, f"cameras.{self.relay}.view")
        _validate_host(self.ip, f"cameras.{self.relay}.host")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError(f"invalid field: cameras.{self.relay}.port")
        if (
            not isinstance(self.path, str)
            or not self.path.startswith("/")
            or len(self.path) > 512
            or any(ord(character) < 33 or character in "#?" for character in self.path)
        ):
            raise ValueError(f"invalid field: cameras.{self.relay}.path")
        if not isinstance(self.calibration, str) or _CALIBRATION.fullmatch(self.calibration) is None:
            raise ValueError(f"invalid field: cameras.{self.relay}.calibration")
        if not isinstance(self.enabled, bool) or not isinstance(self.has_screen, bool):
            raise ValueError(f"invalid field: cameras.{self.relay}.enabled")
        _required_text(self.username, f"cameras.{self.relay}.username")
        _required_text(self.password, f"cameras.{self.relay}.password")
        _register_credentials(self.username, self.password)

    def rtsp_url(self) -> str:
        username = quote(self.username, safe="")
        password = quote(self.password, safe="")
        authority = self.ip if self.port == 554 else f"{self.ip}:{self.port}"
        return f"rtsp://{username}:{password}@{authority}{self.path}"

    def public_dict(self) -> dict[str, Any]:
        return {
            "relay": self.relay,
            "view": self.view,
            "host": self.ip,
            "port": self.port,
            "path": self.path,
            "calibration": self.calibration,
            "enabled": self.enabled,
            "has_screen": self.has_screen,
            "username": self.username,
            "password_configured": bool(self.password),
        }


@dataclass(frozen=True)
class LiveConfig:
    """Runtime camera inventory.

    ``username`` and ``password`` are retained for backwards-compatible callers
    and legacy file migration. New v2 files keep credentials on each camera and
    therefore leave those two attributes empty.
    """

    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    cameras: tuple[CameraConfig, ...] = ()
    source_width: int = 2560
    source_height: int = 1440

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("live configuration requires at least one camera")
        # The deployed DeepStream reader and all production calibration files
        # use original 2560x1440 coordinates.  Camera inventory is dynamic,
        # but accepting another resolution here would silently mis-map screen
        # polygons and person/phone geometry.
        if self.source_width != 2560:
            raise ValueError("invalid field: runtime.source_width")
        if self.source_height != 1440:
            raise ValueError("invalid field: runtime.source_height")
        relays = [camera.relay for camera in self.cameras]
        views = [camera.view for camera in self.cameras]
        calibrations = [camera.calibration for camera in self.cameras]
        if len(set(relays)) != len(relays):
            raise ValueError("duplicate camera relay")
        if len(set(views)) != len(views):
            raise ValueError("duplicate camera view")
        if len(set(calibrations)) != len(calibrations):
            raise ValueError("each camera requires a distinct calibration file")
        has_username = bool(self.username)
        has_password = bool(self.password)
        if has_username != has_password:
            raise ValueError("missing or invalid field: username/password")
        if has_username:
            _required_text(self.username, "username")
            _required_text(self.password, "password")
            for camera in self.cameras:
                if camera.username != self.username or camera.password != self.password:
                    raise ValueError("camera credentials do not match live configuration")
            _register_credentials(self.username, self.password)

    @property
    def enabled_cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(camera for camera in self.cameras if camera.enabled)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LiveConfig":
        config_path = Path(path)
        if _IS_POSIX:
            path_status = config_path.lstat()
            _validate_posix_config_file(path_status)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(config_path, flags)
            try:
                _validate_posix_config_file(os.fstat(descriptor))
                with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                    descriptor = -1
                    payload = json.load(handle)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        else:
            with config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("configuration must be a JSON object")
        if payload.get("schema_version") == CONFIG_SCHEMA_VERSION:
            return cls._from_v2(payload)
        return cls._from_legacy(payload)

    @classmethod
    def _from_v2(cls, payload: dict[str, Any]) -> "LiveConfig":
        runtime = payload.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("invalid field: runtime")
        raw_cameras = payload.get("cameras")
        if not isinstance(raw_cameras, list):
            raise ValueError("missing or invalid field: cameras")
        cameras: list[CameraConfig] = []
        for index, item in enumerate(raw_cameras, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"invalid field: cameras[{index}]")
            relay = _required_text(item.get("relay"), f"cameras[{index}].relay")
            cameras.append(
                CameraConfig(
                    relay=relay,
                    view=_required_text(item.get("view"), f"cameras.{relay}.view"),
                    ip=_required_text(item.get("host", item.get("ip")), f"cameras.{relay}.host"),
                    calibration=_required_text(item.get("calibration"), f"cameras.{relay}.calibration"),
                    has_screen=bool(item.get("has_screen", True)),
                    username=_required_text(item.get("username"), f"cameras.{relay}.username"),
                    password=_required_text(item.get("password"), f"cameras.{relay}.password"),
                    port=item.get("port", 554),
                    path=item.get("path", RTSP_PATH),
                    enabled=item.get("enabled", True),
                )
            )
        return cls(
            cameras=tuple(cameras),
            source_width=runtime.get("source_width", 2560),
            source_height=runtime.get("source_height", 1440),
        )

    @classmethod
    def _from_legacy(cls, payload: dict[str, Any]) -> "LiveConfig":
        username = _required_text(payload.get("username"), "username")
        password = _required_text(payload.get("password"), "password")
        camera_ips = payload.get("cameras")
        if not isinstance(camera_ips, dict):
            raise ValueError("missing or invalid field: cameras")
        cameras: list[CameraConfig] = []
        for relay, view, calibration, has_screen in _LEGACY_ROUTES:
            ip = _required_text(camera_ips.get(view), f"cameras.{view}")
            _validate_host(ip, f"cameras.{view}")
            cameras.append(
                CameraConfig(
                    relay=relay,
                    view=view,
                    ip=ip,
                    calibration=calibration,
                    has_screen=has_screen,
                    username=username,
                    password=password,
                )
            )
        expected_views = {route[1] for route in _LEGACY_ROUTES}
        unknown_views = set(camera_ips) - expected_views
        if unknown_views:
            names = ", ".join(sorted(str(view) for view in unknown_views))
            raise ValueError(f"unknown camera views: {names}")
        return cls(username=username, password=password, cameras=tuple(cameras))

    def public_cameras(self) -> list[dict[str, Any]]:
        return [camera.public_dict() for camera in self.cameras]

    def save(self, path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> None:
        """Atomically save v2 configuration with owner-only permissions."""

        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "runtime": {
                "source_width": self.source_width,
                "source_height": self.source_height,
            },
            "cameras": [
                {
                    "relay": camera.relay,
                    "view": camera.view,
                    "host": camera.ip,
                    "port": camera.port,
                    "path": camera.path,
                    "username": camera.username,
                    "password": camera.password,
                    "calibration": camera.calibration,
                    "enabled": camera.enabled,
                    "has_screen": camera.has_screen,
                }
                for camera in self.cameras
            ],
        }
        existing_status = None
        if _IS_POSIX:
            try:
                existing_status = config_path.lstat()
            except FileNotFoundError:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            else:
                if not stat.S_ISREG(existing_status.st_mode):
                    raise ValueError("configuration target must be a regular file")
                flags = os.O_WRONLY
            flags |= getattr(os, "O_NOFOLLOW", 0)
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

        descriptor = os.open(config_path, flags, 0o600)
        try:
            if _IS_POSIX:
                opened_status = os.fstat(descriptor)
                if not stat.S_ISREG(opened_status.st_mode):
                    raise ValueError("configuration target must be a regular file")
                if existing_status is not None and (
                    opened_status.st_dev,
                    opened_status.st_ino,
                ) != (existing_status.st_dev, existing_status.st_ino):
                    raise ValueError("configuration target changed while opening")
                os.fchmod(descriptor, 0o600)
                os.ftruncate(descriptor, 0)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(config_path, 0o600)
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def make_legacy_config(
    username: str,
    password: str,
    camera_ips: dict[str, str],
) -> LiveConfig:
    """Construct the legacy seven-camera inventory for the terminal bootstrap."""

    cameras: list[CameraConfig] = []
    for relay, view, calibration, has_screen in _LEGACY_ROUTES:
        cameras.append(
            CameraConfig(
                relay=relay,
                view=view,
                ip=_required_text(camera_ips.get(view), f"cameras.{view}"),
                calibration=calibration,
                has_screen=has_screen,
                username=username,
                password=password,
            )
        )
    return LiveConfig(username=username, password=password, cameras=tuple(cameras))


def iter_enabled(cameras: Iterable[CameraConfig]) -> tuple[CameraConfig, ...]:
    return tuple(camera for camera in cameras if camera.enabled)
