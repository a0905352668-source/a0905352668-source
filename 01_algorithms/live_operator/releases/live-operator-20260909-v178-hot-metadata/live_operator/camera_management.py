"""Versioned camera inventory and screen-calibration editing helpers.

This module deliberately has no HTTP or process-control dependency.  It is the
single validation boundary used by the dashboard: callers receive a password-
free inventory, submit a full edited draft, and get either an atomically saved
configuration or a validation error.  Applying a saved draft is a separate,
managed lifecycle operation.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from live_operator.config import CameraConfig, LiveConfig


_SCREEN_ID = re.compile(r"screen_[0-9]{2,3}")
_DEFAULT_MAX_ACTIVE_CAMERAS = 15
_STANDARD_NVR_CHANNEL = re.compile(
    r"^(?:d|ch(?:annel)?|通道)?\s*([1-9][0-9]{0,2})$", re.IGNORECASE
)


class CameraManagementError(ValueError):
    pass


def _standard_nvr_path(channel: Any) -> str:
    """Convert the operator-facing NVR channel into the usual Hikvision path.

    Camera cards deliberately accept either ``D4`` or ``4``.  Keeping this
    translation in the server as well as the browser means a hand-crafted API
    request cannot accidentally persist a blank/raw path.
    """

    text = str(channel or "").strip()
    match = _STANDARD_NVR_CHANNEL.fullmatch(text)
    if match is None:
        raise CameraManagementError("NVR 通道号无效，请填写如 D4、38")
    return f"/Streaming/Channels/{match.group(1)}01"


def _default_calibration_filename(relay: str) -> str:
    safe = re.sub(r"[^a-z0-9_.-]+", "_", relay.lower()).strip("_-")
    return f"{safe or 'new_camera'}_screen_calibration.json"


def _next_generated_relay(used_relays: set[str]) -> str:
    """Allocate an opaque stable relay name for a newly added camera."""

    for number in range(1, 1000):
        candidate = f"camera{number:02d}"
        if candidate not in used_relays:
            return candidate
    raise CameraManagementError("摄像头数量过多，无法分配内部编号")


def _atomic_json(path: Path, payload: Mapping[str, Any], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CameraManagementError(f"无法读取标定文件：{path.name}") from error
    if not isinstance(payload, dict):
        raise CameraManagementError(f"标定文件格式错误：{path.name}")
    return payload


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraManagementError(f"{field} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise CameraManagementError(f"{field} 必须是有限数值")
    return number


def _points(value: Any, width: int, height: int, field: str) -> list[list[float]]:
    if not isinstance(value, list) or not 3 <= len(value) <= 12:
        raise CameraManagementError(f"{field} 需要 3 至 12 个顶点")
    points: list[list[float]] = []
    for index, point in enumerate(value, start=1):
        if not isinstance(point, list) or len(point) != 2:
            raise CameraManagementError(f"{field} 第 {index} 个顶点无效")
        x = _number(point[0], f"{field} x")
        y = _number(point[1], f"{field} y")
        if x < 0 or x > width or y < 0 or y > height:
            raise CameraManagementError(f"{field} 顶点超出画面范围")
        points.append([round(x, 2), round(y, 2)])
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    if abs(area) <= 4.0:
        raise CameraManagementError(f"{field} 面积过小")
    return points


def _rectangle(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y1, 2)],
            [round(x2, 2), round(y2, 2)], [round(x1, 2), round(y2, 2)]]


def _derived_zones(points: list[list[float]], width: int, height: int) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Build conservative operator-editable zones from an annotated screen.

    The screen itself remains a perspective polygon.  Its nearby and front
    areas are intentionally axis-aligned working zones, matching the current
    production calibration representation.  They can be regenerated whenever
    the operator moves the screen outline, avoiding stale zones.
    """

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    screen_width = max(4.0, right - left)
    screen_height = max(4.0, bottom - top)
    near = _rectangle(
        max(0.0, left - max(30.0, screen_width * 1.2)),
        max(0.0, top - max(30.0, screen_height * 0.9)),
        min(float(width), right + max(30.0, screen_width * 1.2)),
        min(float(height), bottom + max(30.0, screen_height * 0.9)),
    )
    front_left = max(0.0, left - max(18.0, screen_width * 0.65))
    front_right = min(float(width), right + max(18.0, screen_width * 0.65))
    front_top = max(0.0, top - max(18.0, screen_height * 0.35))
    front_bottom = min(float(height), bottom + max(24.0, screen_height * 0.7))
    front = _rectangle(front_left, front_top, front_right, front_bottom)
    split_left = front_left + (front_right - front_left) * 0.55
    split_right = front_left + (front_right - front_left) * 0.45
    return near, [
        {"name": "front", "polygon": front, "weight": 1.0},
        {
            "name": "left_side",
            "polygon": _rectangle(front_left, front_top, split_left, front_bottom),
            "weight": 0.8,
        },
        {
            "name": "right_side",
            "polygon": _rectangle(split_right, front_top, front_right, front_bottom),
            "weight": 0.8,
        },
    ]


def _edited_calibration(
    camera: CameraConfig,
    value: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    raw_screens = value.get("screens")
    if not isinstance(raw_screens, list):
        raise CameraManagementError(f"{camera.view} 缺少屏幕标注")
    if camera.enabled and camera.has_screen and not raw_screens:
        raise CameraManagementError(f"{camera.view} 已启用屏幕监测，但尚未标定屏幕位置")
    screens: list[dict[str, Any]] = []
    for index, raw_screen in enumerate(raw_screens, start=1):
        if not isinstance(raw_screen, Mapping):
            raise CameraManagementError(f"{camera.view} 第 {index} 个屏幕格式错误")
        screen_id = raw_screen.get("screen_id")
        if not isinstance(screen_id, str) or _SCREEN_ID.fullmatch(screen_id) is None:
            screen_id = f"screen_{index:02d}"
        points = _points(raw_screen.get("screen_poly", raw_screen.get("points")), width, height,
                         f"{camera.view}/{screen_id}")
        near_zone, danger_zones = _derived_zones(points, width, height)
        screens.append(
            {
                "screen_id": screen_id,
                "screen_poly": points,
                "near_zone": near_zone,
                "danger_zones": danger_zones,
                "ignore_zones": [],
                "params": {
                    "enable_dynamic_person_zone": True,
                    "person_expand_x": 0.35,
                    "person_expand_y": 0.2,
                    "near_zone_detect_interval": 10,
                    "phone_valid_conf_person_roi": 0.35,
                    "phone_valid_conf_near_zone": 0.5,
                    "phone_strong_conf": 0.55,
                    "ignore_center_inside": True,
                    "angle_thresh": 90.0,
                    "angle_relaxed_thresh": 110.0,
                    "hand_radius_ratio": 0.18,
                    "hand_radius_min": 40.0,
                    "corridor_width_ratio": 0.35,
                    "corridor_width_min": 80.0,
                    "alert_counter_threshold": 10.0,
                    "person_state_window": 30,
                    "person_state_min_hits": 16,
                    "person_state_risk_threshold": 0.65,
                    "person_state_alarm_risk": 0.9,
                    "occluded_enable": False,
                    "near_screen_override": False,
                },
            }
        )
    return {
        "version": "2.2",
        "camera_id": camera.relay,
        "frame_size": [width, height],
        "description": "Managed in JianKong camera calibration page.",
        "screens": screens,
    }


class CameraManagementStore:
    def __init__(
        self,
        config_path: str | os.PathLike[str],
        calibration_dir: str | os.PathLike[str],
        *,
        max_active_cameras: int = _DEFAULT_MAX_ACTIVE_CAMERAS,
    ) -> None:
        self.config_path = Path(config_path)
        self.calibration_dir = Path(calibration_dir)
        self.max_active_cameras = max_active_cameras
        self.history_root = self.config_path.parent / "camera_management_history"
        # Drafts are intentionally kept away from the files consumed by the
        # running DeepStream process.  A bad browser submission must never
        # change the configuration that a watchdog would use to recover the
        # current production run.
        self.draft_config_path = self.config_path.with_name(
            f"{self.config_path.stem}.pending.json"
        )
        self.draft_calibration_dir = self.config_path.parent / "camera_calibration_drafts"

    def _draft_config(self) -> LiveConfig | None:
        if not self.draft_config_path.is_file():
            return None
        try:
            return LiveConfig.load(self.draft_config_path)
        except (OSError, ValueError) as error:
            raise CameraManagementError("待发布摄像头配置无效，请先重新保存草稿") from error

    def _calibration_path(self, filename: str, *, draft: bool) -> Path:
        root = self.draft_calibration_dir if draft else self.calibration_dir
        return root / filename

    def _read_calibration_for_state(
        self, filename: str, *, prefers_draft: bool
    ) -> dict[str, Any] | None:
        if prefers_draft:
            draft_path = self._calibration_path(filename, draft=True)
            if draft_path.is_file():
                return _read_json(draft_path)
        active_path = self._calibration_path(filename, draft=False)
        if active_path.is_file():
            return _read_json(active_path)
        return None

    def public_state(self) -> dict[str, Any]:
        active_config = LiveConfig.load(self.config_path)
        draft_config = self._draft_config()
        config = draft_config or active_config
        prefers_draft = draft_config is not None
        calibrations: dict[str, Any] = {}
        for camera in config.cameras:
            payload = self._read_calibration_for_state(
                camera.calibration, prefers_draft=prefers_draft
            )
            if payload is not None:
                screens = payload.get("screens", [])
                calibrations[camera.relay] = {
                    "calibration": camera.calibration,
                    "frame_size": payload.get("frame_size", [config.source_width, config.source_height]),
                    "screens": [
                        {
                            "screen_id": item.get("screen_id"),
                            "screen_poly": item.get("screen_poly", item.get("points", [])),
                        }
                        for item in screens
                        if isinstance(item, Mapping)
                    ],
                }
        return {
            "schema_version": 1,
            "max_active_cameras": self.max_active_cameras,
            "active_cameras": len(active_config.enabled_cameras),
            "draft_active_cameras": len(config.enabled_cameras),
            "pending_activation": prefers_draft,
            "runtime": {
                "source_width": config.source_width,
                "source_height": config.source_height,
            },
            "cameras": config.public_cameras(),
            "calibrations": calibrations,
        }

    def preview_camera(self, relay: str) -> tuple[CameraConfig, bool]:
        """Return the draft/active source and whether its local relay is usable.

        A newly entered camera has no MediaMTX relay until it is published.  It
        still needs one read-only frame for screen calibration, so its saved
        draft connection is an allowed preview source.  Existing unchanged
        cameras keep using the local relay and never open an extra source
        connection merely because the management page is open.
        """

        active = LiveConfig.load(self.config_path)
        draft = self._draft_config()
        selected_config = draft or active
        selected = next((camera for camera in selected_config.cameras if camera.relay == relay), None)
        if selected is None:
            raise CameraManagementError("摄像头不存在于当前配置草稿")
        active_camera = next((camera for camera in active.cameras if camera.relay == relay), None)
        uses_active_relay = (
            active_camera is not None
            and active_camera.enabled
            and selected.enabled
            and (
                selected.ip,
                selected.port,
                selected.path,
                selected.username,
                selected.password,
            ) == (
                active_camera.ip,
                active_camera.port,
                active_camera.path,
                active_camera.username,
                active_camera.password,
            )
        )
        return selected, uses_active_relay

    def save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise CameraManagementError("配置内容必须是对象")
        active = LiveConfig.load(self.config_path)
        current = self._draft_config() or active
        raw_runtime = payload.get("runtime", {})
        raw_cameras = payload.get("cameras")
        if not isinstance(raw_runtime, Mapping) or not isinstance(raw_cameras, list):
            raise CameraManagementError("缺少摄像头配置")
        existing = {camera.relay: camera for camera in current.cameras}
        active_existing = {camera.relay: camera for camera in active.cameras}
        used_relays = {
            str(item.get("relay")).strip()
            for item in raw_cameras
            if isinstance(item, Mapping) and str(item.get("relay") or "").strip()
        }
        used_relays.update(existing)
        used_relays.update(active_existing)
        cameras: list[CameraConfig] = []
        for index, item in enumerate(raw_cameras, start=1):
            if not isinstance(item, Mapping):
                raise CameraManagementError(f"第 {index} 路摄像头格式错误")
            relay_text = str(item.get("relay") or "").strip()
            if not relay_text:
                relay_text = _next_generated_relay(used_relays)
                used_relays.add(relay_text)
            old = existing.get(relay_text)
            active_old = active_existing.get(relay_text)
            password = item.get("password")
            if password in (None, ""):
                password = old.password if old is not None else (active_old.password if active_old else "")
            username = item.get("username")
            if username in (None, ""):
                username = old.username if old is not None else (active_old.username if active_old else "")
            channel = item.get("channel")
            raw_path = item.get("path")
            if channel is not None and str(channel).strip():
                path = _standard_nvr_path(channel)
            else:
                path = str(raw_path or "")
            calibration = str(item.get("calibration") or "")
            if not calibration:
                calibration = _default_calibration_filename(relay_text)
            cameras.append(
                CameraConfig(
                    relay=relay_text,
                    view=str(item.get("view") or ""),
                    ip=str(item.get("host", item.get("ip", "")) or ""),
                    calibration=calibration,
                    has_screen=bool(item.get("has_screen", True)),
                    username=str(username or ""),
                    password=str(password or ""),
                    port=item.get("port", 554),
                    path=path,
                    enabled=item.get("enabled", True),
                )
            )
        config = LiveConfig(
            cameras=tuple(cameras),
            source_width=raw_runtime.get("source_width", current.source_width),
            source_height=raw_runtime.get("source_height", current.source_height),
        )
        if len(config.enabled_cameras) > self.max_active_cameras:
            raise CameraManagementError(
                f"当前模型最多同时启用 {self.max_active_cameras} 路摄像头"
            )
        raw_calibrations = payload.get("calibrations", {})
        if not isinstance(raw_calibrations, Mapping):
            raise CameraManagementError("标定内容格式错误")
        updated: dict[str, dict[str, Any]] = {}
        for camera in config.cameras:
            proposed = raw_calibrations.get(camera.relay)
            if proposed is None:
                continue
            if not isinstance(proposed, Mapping):
                raise CameraManagementError(f"{camera.view} 标定内容格式错误")
            # A newly saved camera has no live relay/background yet.  Permit
            # its draft to exist without an empty fake calibration; publish()
            # remains the strict boundary and will reject it until the
            # operator has actually marked at least one screen.
            if camera.has_screen and proposed.get("screens") == []:
                continue
            updated[camera.calibration] = _edited_calibration(
                camera, proposed, width=config.source_width, height=config.source_height
            )
        self._backup_current()
        for filename, calibration in updated.items():
            _atomic_json(
                self._calibration_path(filename, draft=True), calibration, 0o600
            )
        config.save(self.draft_config_path)
        return self.public_state()

    def preflight_pending(self) -> dict[str, Any]:
        """Verify every planned active source before touching production files.

        This deliberately performs no MediaMTX, DeepStream, GPU, or lifecycle
        action.  It catches an incorrect camera address/credential/path before
        the privileged publisher pauses the watchdog or restarts production.
        """

        draft = self._draft_config()
        if draft is None:
            raise CameraManagementError("没有待发布的摄像头配置")
        try:
            # Import lazily to keep the dashboard's ordinary draft path free
            # from lifecycle side effects and to avoid an import cycle.
            from live_operator.cli import LifecycleError, RuntimeHooks

            RuntimeHooks().probe_sources(draft)
        except LifecycleError as error:
            raise CameraManagementError("摄像头连接预检失败，请检查 IP、端口、路径和账号") from error
        return {
            "active_cameras": len(draft.enabled_cameras),
            "relays": [camera.relay for camera in draft.enabled_cameras],
        }

    def publish(self, after_apply: Callable[[], Any]) -> dict[str, Any]:
        """Atomically promote the draft and run the manifest publisher.

        ``after_apply`` is deliberately injected: this data-layer module does
        not own process control.  The privileged launcher first updates the
        inference manifest while the watchdog is paused, then performs the
        regular stop/start sequence.  If manifest publication fails, every
        active configuration file is restored before the error is returned.
        """

        if not callable(after_apply):
            raise TypeError("after_apply must be callable")
        draft = self._draft_config()
        if draft is None:
            raise CameraManagementError("没有待发布的摄像头配置")

        active = LiveConfig.load(self.config_path)
        filenames = {
            camera.calibration for camera in active.cameras
        } | {camera.calibration for camera in draft.cameras}
        active_snapshot: dict[Path, tuple[bytes, int] | None] = {}
        for filename in filenames:
            path = self._calibration_path(filename, draft=False)
            if path.is_file():
                active_snapshot[path] = (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
            else:
                active_snapshot[path] = None
        config_snapshot = self.config_path.read_bytes()

        promoted: dict[str, dict[str, Any]] = {}
        for camera in draft.cameras:
            payload = self._read_calibration_for_state(
                camera.calibration, prefers_draft=True
            )
            if payload is None:
                # A camera deliberately left disabled is not part of this
                # publish operation.  It may be saved ahead of calibration
                # without preventing the already-ready cameras from going
                # online.
                if camera.enabled and camera.has_screen:
                    raise CameraManagementError(f"{camera.view} 缺少待发布屏幕标定")
                payload = _edited_calibration(
                    camera, {"screens": []}, width=draft.source_width, height=draft.source_height
                )
            _edited_calibration(
                camera,
                {"screens": payload.get("screens", [])},
                width=draft.source_width,
                height=draft.source_height,
            )
            promoted[camera.calibration] = payload

        self._backup_current()
        try:
            for filename, payload in promoted.items():
                _atomic_json(self._calibration_path(filename, draft=False), payload, 0o644)
            draft.save(self.config_path)
            after_apply()
        except Exception as error:
            _atomic_bytes(self.config_path, config_snapshot, 0o600)
            for path, snapshot in active_snapshot.items():
                if snapshot is None:
                    path.unlink(missing_ok=True)
                else:
                    bytes_value, mode = snapshot
                    _atomic_bytes(path, bytes_value, mode)
            raise CameraManagementError("待发布配置未通过发布校验，已恢复线上配置") from error

        self._discard_draft_files(draft)
        return self.public_state()

    def discard_draft(self) -> dict[str, Any]:
        """Discard only the pending version; active production files remain intact."""

        draft = self._draft_config()
        if draft is None:
            return self.public_state()
        self._backup_current()
        self._discard_draft_files(draft)
        return self.public_state()

    def _discard_draft_files(self, draft: LiveConfig) -> None:
        self.draft_config_path.unlink(missing_ok=True)
        for camera in draft.cameras:
            self._calibration_path(camera.calibration, draft=True).unlink(missing_ok=True)
        try:
            self.draft_calibration_dir.rmdir()
        except OSError:
            pass

    def _backup_current(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self.history_root / stamp
        destination.mkdir(parents=True, exist_ok=False)
        # History is a recovery record of the active version, never a copy of
        # the browser draft.  A draft can contain invalid connection details.
        shutil.copy2(self.config_path, destination / self.config_path.name)
        calibration_backup = destination / "calibrations"
        calibration_backup.mkdir()
        config = LiveConfig.load(self.config_path)
        for camera in config.cameras:
            source = self._calibration_path(
                camera.calibration, draft=self.draft_config_path.is_file()
            )
            if not source.is_file():
                source = self._calibration_path(camera.calibration, draft=False)
            if source.is_file():
                shutil.copy2(source, calibration_backup / source.name)


def publish_pending_camera_configuration(
    config_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Promote a pending inventory and seal it into the inference manifest.

    This function intentionally does not restart any process.  The root-owned
    ``jiankong-camera-publish`` wrapper pauses the watchdog, calls this as the
    runtime user, then reuses the existing safe stop/start entry point.
    """

    from live_operator.inference import (
        resolve_current_inference_version,
        write_live_runtime_manifest,
    )

    version = resolve_current_inference_version()
    store = CameraManagementStore(config_path, version.calibration_dir)
    return store.publish(
        lambda: write_live_runtime_manifest(
            version.deepstream_binary,
            version.deepstream_source,
            config_path=config_path,
        )
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="JianKong camera draft publisher")
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--publish", action="store_true")
    action.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        from live_operator.inference import resolve_current_inference_version

        version = resolve_current_inference_version()
        state = CameraManagementStore(args.config, version.calibration_dir).preflight_pending()
    else:
        state = publish_pending_camera_configuration(args.config)
    print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
