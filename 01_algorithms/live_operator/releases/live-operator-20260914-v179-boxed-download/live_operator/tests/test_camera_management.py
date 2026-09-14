from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from live_operator.camera_auth import CameraAuthError, CameraManagementAuthenticator
from live_operator.camera_management import (
    CameraManagementError,
    CameraManagementStore,
    publish_pending_camera_configuration,
)
from live_operator.config import CameraConfig, LiveConfig
from live_operator.dashboard import DashboardApp


def _write_active_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "runtime" / "live_operator.json"
    calibration_dir = tmp_path / "calibrations"
    calibration_dir.mkdir()
    LiveConfig(
        cameras=(
            CameraConfig(
                relay="camera01",
                view="office_a",
                ip="192.0.2.10",
                calibration="office_a.json",
                username="operator",
                password="not-a-real-password",
            ),
        )
    ).save(config_path)
    (calibration_dir / "office_a.json").write_text(
        json.dumps(
            {
                "version": "2.2",
                "frame_size": [2560, 1440],
                "screens": [
                    {
                        "screen_id": "screen_01",
                        "screen_poly": [[800, 400], [1200, 400], [1200, 700], [800, 700]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path, calibration_dir


def _draft_payload() -> dict[str, object]:
    return {
        "runtime": {"source_width": 2560, "source_height": 1440},
        "cameras": [
            {
                "relay": "camera01",
                "view": "office_a_renamed",
                "host": "192.0.2.10",
                "port": 554,
                "path": "/Streaming/Channels/101",
                "username": "operator",
                "calibration": "office_a.json",
                "enabled": True,
                "has_screen": True,
            }
        ],
        "calibrations": {
            "camera01": {
                "screens": [
                    {
                        "screen_id": "screen_01",
                        "screen_poly": [[810, 410], [1210, 410], [1210, 710], [810, 710]],
                    }
                ]
            }
        },
    }


def test_save_creates_private_draft_without_changing_active_files(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    store = CameraManagementStore(config_path, calibration_dir)
    active_config = config_path.read_bytes()
    active_calibration = (calibration_dir / "office_a.json").read_bytes()

    saved = store.save(_draft_payload())

    assert saved["pending_activation"] is True
    assert saved["active_cameras"] == 1
    assert saved["draft_active_cameras"] == 1
    assert config_path.read_bytes() == active_config
    assert (calibration_dir / "office_a.json").read_bytes() == active_calibration
    assert store.draft_config_path.is_file()
    assert stat.S_IMODE(store.draft_config_path.stat().st_mode) == 0o600
    draft_calibration = store.draft_calibration_dir / "office_a.json"
    assert draft_calibration.is_file()
    assert stat.S_IMODE(draft_calibration.stat().st_mode) == 0o600
    payload = json.loads(draft_calibration.read_text(encoding="utf-8"))
    assert payload["screens"][0]["screen_poly"][0] == [810.0, 410.0]
    assert payload["screens"][0]["near_zone"]
    assert payload["screens"][0]["danger_zones"]


def test_screen_camera_draft_can_wait_for_first_calibration(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    store = CameraManagementStore(config_path, calibration_dir)
    saved = store.save(
        {
            "runtime": {"source_width": 2560, "source_height": 1440},
            "cameras": [
                {
                    "relay": "fs_d4",
                    "view": "FS-D4",
                    "host": "192.168.210.12",
                    "channel": "D4",
                    "username": "operator",
                    "password": "not-a-real-password",
                    "enabled": False,
                    "has_screen": True,
                }
            ],
            "calibrations": {"fs_d4": {"screens": []}},
        }
    )

    assert saved["pending_activation"] is True
    published = store.publish(lambda: None)

    assert published["pending_activation"] is False
    assert LiveConfig.load(config_path).cameras[0].enabled is False


def test_disabled_camera_preview_uses_direct_source_not_missing_relay(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    store = CameraManagementStore(config_path, calibration_dir)
    store.save(
        {
            "runtime": {"source_width": 2560, "source_height": 1440},
            "cameras": [
                {
                    "relay": "camera01",
                    "view": "office_a",
                    "host": "192.0.2.10",
                    "path": "/Streaming/Channels/101",
                    "username": "operator",
                    "enabled": False,
                    "has_screen": True,
                }
            ],
            "calibrations": {"camera01": {"screens": []}},
        }
    )

    _camera, uses_active_relay = store.preview_camera("camera01")

    assert uses_active_relay is False


def test_save_accepts_nvr_channel_and_generates_calibration_filename(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    payload = {
        "runtime": {"source_width": 2560, "source_height": 1440},
        "cameras": [
            {
                "relay": "fs_38",
                "view": "FS-38",
                "host": "192.168.210.12",
                "port": 554,
                "channel": "D38",
                "username": "operator",
                "password": "not-a-real-password",
                "enabled": False,
                "has_screen": False,
            }
        ],
        "calibrations": {},
    }

    CameraManagementStore(config_path, calibration_dir).save(payload)
    draft = LiveConfig.load(config_path.with_name("live_operator.pending.json"))

    assert draft.cameras[0].path == "/Streaming/Channels/3801"
    assert draft.cameras[0].calibration == "fs_38_screen_calibration.json"


def test_save_rejects_invalid_nvr_channel(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    payload = _draft_payload()
    payload["cameras"][0]["channel"] = "38-A"

    with pytest.raises(CameraManagementError, match="NVR 通道号无效"):
        CameraManagementStore(config_path, calibration_dir).save(payload)


def test_default_capacity_allows_ten_active_cameras_but_rejects_eleven(
    tmp_path: Path,
) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    store = CameraManagementStore(config_path, calibration_dir)

    def payload(count: int) -> dict[str, object]:
        return {
            "runtime": {"source_width": 2560, "source_height": 1440},
            "cameras": [
                {
                    "relay": f"camera{index:02d}",
                    "view": f"office_{index}",
                    "host": f"192.0.2.{index}",
                    "path": "/Streaming/Channels/101",
                    "username": "operator",
                    "password": "not-a-real-password",
                    "enabled": True,
                    "has_screen": False,
                }
                for index in range(1, count + 1)
            ],
            "calibrations": {},
        }

    saved = store.save(payload(10))
    assert saved["draft_active_cameras"] == 10

    with pytest.raises(CameraManagementError, match="最多同时启用 10 路"):
        store.save(payload(11))


def test_save_assigns_hidden_internal_relay_when_client_omits_it(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    payload = {
        "runtime": {"source_width": 2560, "source_height": 1440},
        "cameras": [
            {
                "view": "FS-38",
                "host": "192.168.210.12",
                "channel": "38",
                "username": "operator",
                "password": "not-a-real-password",
                "enabled": False,
                "has_screen": False,
            }
        ],
        "calibrations": {},
    }

    CameraManagementStore(config_path, calibration_dir).save(payload)
    draft = LiveConfig.load(config_path.with_name("live_operator.pending.json"))

    assert draft.cameras[0].relay == "camera02"
    assert draft.cameras[0].view == "FS-38"


def test_discard_removes_only_draft(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    store = CameraManagementStore(config_path, calibration_dir)
    store.save(_draft_payload())

    state = store.discard_draft()

    assert state["pending_activation"] is False
    assert not store.draft_config_path.exists()
    assert (calibration_dir / "office_a.json").is_file()


def test_publish_promotes_draft_only_after_manifest_callback_succeeds(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    store = CameraManagementStore(config_path, calibration_dir)
    store.save(_draft_payload())
    published = []

    state = store.publish(lambda: published.append("manifest-updated"))

    assert published == ["manifest-updated"]
    assert state["pending_activation"] is False
    assert LiveConfig.load(config_path).cameras[0].view == "office_a_renamed"
    assert json.loads((calibration_dir / "office_a.json").read_text(encoding="utf-8"))["screens"][0]["screen_poly"][0] == [810.0, 410.0]


def test_publish_rolls_back_active_files_when_manifest_callback_fails(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    store = CameraManagementStore(config_path, calibration_dir)
    store.save(_draft_payload())
    before_config = config_path.read_bytes()
    before_calibration = (calibration_dir / "office_a.json").read_bytes()

    with pytest.raises(CameraManagementError, match="已恢复"):
        store.publish(lambda: (_ for _ in ()).throw(RuntimeError("manifest failed")))

    assert config_path.read_bytes() == before_config
    assert (calibration_dir / "office_a.json").read_bytes() == before_calibration
    assert store.draft_config_path.is_file()


def test_pending_publish_writes_manifest_with_resolved_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    CameraManagementStore(config_path, calibration_dir).save(_draft_payload())
    binary = tmp_path / "deepstream-bin"
    source = tmp_path / "deepstream-source"
    binary.write_bytes(b"binary")
    source.write_bytes(b"source")
    version = SimpleNamespace(
        calibration_dir=calibration_dir,
        deepstream_binary=binary,
        deepstream_source=source,
    )
    observed: dict[str, object] = {}
    import live_operator.inference as inference_module

    monkeypatch.setattr(inference_module, "resolve_current_inference_version", lambda: version)

    def fake_write(binary_path: Path, source_path: Path, **kwargs: object) -> Path:
        observed.update(binary=binary_path, source=source_path, **kwargs)
        return tmp_path / "manifest.json"

    monkeypatch.setattr(inference_module, "write_live_runtime_manifest", fake_write)

    state = publish_pending_camera_configuration(config_path)

    assert state["pending_activation"] is False
    assert observed == {"binary": binary, "source": source, "config_path": config_path}


def test_dashboard_camera_draft_api_returns_no_password(tmp_path: Path) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    run_dir = tmp_path / "run"
    app = DashboardApp(run_dir, config_path=config_path, calibration_dir=calibration_dir)

    state_response = app.handle("GET", "/api/camera-management")
    assert state_response.status == 200
    state = json.loads(state_response.body)
    assert "password" not in state["cameras"][0]
    assert state["cameras"][0]["password_configured"] is True

    response = app.handle(
        "POST",
        "/api/camera-management/draft",
        body=json.dumps(_draft_payload()).encode("utf-8"),
    )
    assert response.status == 200
    assert json.loads(response.body)["pending_activation"] is True


def test_dashboard_camera_preview_uses_local_relay_and_jpg_tempfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import live_operator.dashboard as dashboard_module

    config_path, calibration_dir = _write_active_config(tmp_path)
    app = DashboardApp(tmp_path / "run", config_path=config_path, calibration_dir=calibration_dir)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"J" * 2048)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dashboard_module.subprocess, "run", fake_run)
    response = app.handle("GET", "/api/camera-management/camera01/preview")

    assert response.status == 200
    assert "rtsp://127.0.0.1:8554/camera01" in commands[0]
    assert commands[0][commands[0].index("-skip_frame") + 1] == "nokey"
    assert commands[0][-1].endswith(".jpg")
    assert b"".join(response.iter_body()) == b"J" * 2048


def test_dashboard_camera_preview_uses_saved_draft_source_for_new_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import live_operator.dashboard as dashboard_module

    config_path, calibration_dir = _write_active_config(tmp_path)
    store = CameraManagementStore(config_path, calibration_dir)
    store.save(
        {
            "runtime": {"source_width": 2560, "source_height": 1440},
            "cameras": [
                {
                    "relay": "fs_d4",
                    "view": "FS-D4",
                    "host": "192.168.210.12",
                    "channel": "D4",
                    "username": "operator",
                    "password": "not-a-real-password",
                    "enabled": False,
                    "has_screen": True,
                }
            ],
            "calibrations": {"fs_d4": {"screens": []}},
        }
    )
    app = DashboardApp(tmp_path / "run", config_path=config_path, calibration_dir=calibration_dir)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"J" * 2048)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dashboard_module.subprocess, "run", fake_run)
    response = app.handle("GET", "/api/camera-management/fs_d4/preview")

    assert response.status == 200
    assert "rtsp://operator:not-a-real-password@192.168.210.12/Streaming/Channels/401" in commands[0]


def test_camera_management_password_is_salted_and_not_plaintext(tmp_path: Path) -> None:
    auth = CameraManagementAuthenticator(tmp_path / "camera_management_admin.json")

    assert auth.authenticate_or_initialize("a-long-management-password") is True
    stored = auth.path.read_text(encoding="utf-8")
    assert "a-long-management-password" not in stored
    assert auth.authenticate_or_initialize("a-long-management-password") is False
    with pytest.raises(CameraAuthError, match="不正确"):
        auth.authenticate_or_initialize("wrong-management-password")


def test_dashboard_publish_starts_only_fixed_detached_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, calibration_dir = _write_active_config(tmp_path)
    app = DashboardApp(
        tmp_path / "run",
        config_path=config_path,
        calibration_dir=calibration_dir,
        camera_publish_command=("/usr/bin/sudo", "-n", "/usr/local/bin/jiankong-camera-publish"),
    )
    CameraManagementStore(config_path, calibration_dir).save(_draft_payload())
    observed: list[tuple[list[str], dict[str, object]]] = []

    class Process:
        def poll(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: object) -> Process:
        observed.append((command, kwargs))
        return Process()

    import live_operator.dashboard as dashboard_module

    monkeypatch.setattr(dashboard_module.subprocess, "Popen", fake_popen)
    response = app.handle(
        "POST",
        "/api/camera-management/publish",
        body=json.dumps({"password": "a-long-management-password"}).encode("utf-8"),
    )

    assert response.status == 202
    assert observed[0][0] == ["/usr/bin/sudo", "-n", "/usr/local/bin/jiankong-camera-publish"]
    assert observed[0][1]["start_new_session"] is True
    assert "a-long-management-password" not in b"".join(response.iter_body()).decode("utf-8")
