import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

import live_operator.config as config_module
from live_operator.config import CameraConfig, LiveConfig, redact_text


CAMERA_IPS = {
    "dianqi1": "192.0.2.11",
    "dianqi2": "192.0.2.12",
    "jixie1": "192.0.2.13",
    "jixie2": "192.0.2.14",
    "ruanjian1": "192.0.2.15",
    "ruanjian2": "192.0.2.16",
    "zoulang": "192.0.2.17",
}


def write_config(path: Path, **overrides: object) -> None:
    payload = {
        "username": "camera-user",
        "password": "not-a-real-p@ss:/?#[]",
        "cameras": CAMERA_IPS,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)


def test_load_binds_explicit_views_to_fixed_routes(tmp_path: Path) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)

    config = LiveConfig.load(path)

    assert [(camera.relay, camera.view, camera.ip, camera.calibration) for camera in config.cameras] == [
        ("camera01", "dianqi1", "192.0.2.11", "camera_01_screen_calibration_v21.json"),
        ("camera02", "dianqi2", "192.0.2.12", "camera_02_screen_calibration_v21.json"),
        ("camera03", "jixie1", "192.0.2.13", "camera_mechanical_01_screen_calibration_v21.json"),
        ("camera04", "jixie2", "192.0.2.14", "camera_mechanical_02_screen_calibration_v21.json"),
        ("camera05", "ruanjian1", "192.0.2.15", "camera_software_01_screen_calibration_v21.json"),
        ("camera06", "ruanjian2", "192.0.2.16", "camera_software_02_screen_calibration_v21.json"),
        ("camera07", "zoulang", "192.0.2.17", "camera_corridor_screen_calibration_v21.json"),
    ]


def test_rtsp_url_percent_encodes_credentials(tmp_path: Path) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    config = LiveConfig.load(path)

    assert config.cameras[0].rtsp_url() == (
        "rtsp://camera-user:not-a-real-p%40ss%3A%2F%3F%23%5B%5D"
        "@192.0.2.11/Streaming/Channels/101"
    )


def test_camera01_is_the_only_route_without_a_screen(tmp_path: Path) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    config = LiveConfig.load(path)

    assert config.cameras[0].relay == "camera01"
    assert config.cameras[0].has_screen is False
    assert all(camera.has_screen for camera in config.cameras[1:])


def test_redact_text_hides_raw_encoded_and_complete_userinfo(tmp_path: Path) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    config = LiveConfig.load(path)
    password = config.password
    encoded_password = quote(password, safe="")
    encoded_user = quote(config.username, safe="")
    userinfo = f"{encoded_user}:{encoded_password}"
    raw_userinfo = f"{config.username}:{password}"
    text = (
        f"raw={password} encoded={encoded_password} "
        f"rtsp://{raw_userinfo}@192.0.2.10/raw "
        f"rtsp://{userinfo}@192.0.2.11/encoded"
    )

    redacted = redact_text(text)

    assert password not in redacted
    assert encoded_password not in redacted
    assert userinfo not in redacted
    assert raw_userinfo not in redacted
    assert "rtsp://camera-user:[REDACTED]@" not in redacted
    assert redacted.count("rtsp://[REDACTED]@") == 2


def test_directly_constructed_camera_registers_complete_userinfo_for_redaction() -> None:
    camera = CameraConfig(
        relay="camera01",
        view="dianqi1",
        ip="192.0.2.11",
        calibration="camera_01_screen_calibration_v21.json",
        has_screen=False,
        username="direct-user",
        password="direct-p@ss:/",
    )

    redacted = redact_text(camera.rtsp_url())

    assert redacted == "rtsp://[REDACTED]@192.0.2.11/Streaming/Channels/101"


def test_save_uses_owner_only_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "runtime" / "live_operator.json"
    write_config(source)
    config = LiveConfig.load(source)
    open_modes: list[int] = []
    chmod_modes: list[int] = []
    real_open = os.open
    real_chmod = os.chmod

    def recording_open(path: os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        open_modes.append(mode)
        return real_open(path, flags, mode)

    def recording_chmod(path: os.PathLike[str], mode: int) -> None:
        chmod_modes.append(mode)
        real_chmod(path, mode)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "chmod", recording_chmod)

    config.save(destination)

    assert open_modes == [0o600]
    assert chmod_modes == [0o600]
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    reloaded = LiveConfig.load(destination)
    assert reloaded.cameras == config.cameras


def test_save_tightens_existing_file_before_writing_plaintext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "live_operator.json"
    write_config(source)
    config = LiveConfig.load(source)
    destination.write_text("old permissive content", encoding="utf-8")
    os.chmod(destination, 0o644)
    events: list[tuple[str, int | None]] = []
    real_dump = json.dump
    real_chmod = os.chmod

    def recording_fchmod(descriptor: int, mode: int) -> None:
        events.append(("fchmod", mode))

    def recording_dump(payload: object, handle: object, **kwargs: object) -> None:
        events.append(("dump", None))
        real_dump(payload, handle, **kwargs)

    def recording_chmod(path: os.PathLike[str], mode: int) -> None:
        events.append(("chmod", mode))
        real_chmod(path, mode)

    monkeypatch.setattr(config_module, "_IS_POSIX", True, raising=False)
    monkeypatch.setattr(config_module.os, "fchmod", recording_fchmod, raising=False)
    monkeypatch.setattr(config_module.json, "dump", recording_dump)
    monkeypatch.setattr(config_module.os, "chmod", recording_chmod)

    config.save(destination)

    assert events[0] == ("fchmod", 0o600)
    assert events.index(("fchmod", 0o600)) < events.index(("dump", None))
    assert events[-1] == ("chmod", 0o600)


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFDIR])
def test_posix_save_rejects_symlink_and_non_regular_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_type: int,
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "live_operator.json"
    write_config(source)
    config = LiveConfig.load(source)
    destination.write_text("must remain unchanged", encoding="utf-8")
    monkeypatch.setattr(config_module, "_IS_POSIX", True)
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda self: SimpleNamespace(st_mode=file_type | 0o600),
    )

    with pytest.raises(ValueError, match="regular file"):
        config.save(destination)

    assert destination.read_text(encoding="utf-8") == "must remain unchanged"


def test_posix_save_rejects_target_changed_between_lstat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "live_operator.json"
    write_config(source)
    config = LiveConfig.load(source)
    destination.write_text("old content", encoding="utf-8")
    monkeypatch.setattr(config_module, "_IS_POSIX", True)
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda self: SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600, st_dev=10, st_ino=20
        ),
    )
    monkeypatch.setattr(
        config_module.os,
        "fstat",
        lambda descriptor: SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600, st_dev=10, st_ino=21
        ),
    )

    with pytest.raises(ValueError, match="changed"):
        config.save(destination)


def test_posix_load_rejects_group_or_world_accessible_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    monkeypatch.setattr(config_module, "_IS_POSIX", True, raising=False)
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda self: SimpleNamespace(st_mode=stat.S_IFREG | 0o640),
    )

    with pytest.raises(ValueError, match="0600"):
        LiveConfig.load(path)


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFDIR])
def test_posix_load_rejects_symlink_and_non_regular_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_type: int,
) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    monkeypatch.setattr(config_module, "_IS_POSIX", True, raising=False)
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda self: SimpleNamespace(st_mode=file_type | 0o600),
    )

    with pytest.raises(ValueError, match="regular file"):
        LiveConfig.load(path)


def test_posix_load_accepts_regular_owner_only_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    monkeypatch.setattr(config_module, "_IS_POSIX", True, raising=False)
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda self: SimpleNamespace(st_mode=stat.S_IFREG | 0o600),
    )
    monkeypatch.setattr(
        config_module.os,
        "fstat",
        lambda descriptor: SimpleNamespace(st_mode=stat.S_IFREG | 0o600),
    )

    assert LiveConfig.load(path).cameras[0].ip == "192.0.2.11"


@pytest.mark.parametrize(
    ("missing_field", "payload"),
    [
        ("username", {"password": "dummy", "cameras": CAMERA_IPS}),
        ("password", {"username": "dummy", "cameras": CAMERA_IPS}),
        ("cameras", {"username": "dummy", "password": "dummy"}),
        (
            "cameras.dianqi1",
            {
                "username": "dummy",
                "password": "dummy",
                "cameras": {key: value for key, value in CAMERA_IPS.items() if key != "dianqi1"},
            },
        ),
    ],
)
def test_load_rejects_missing_fields(
    tmp_path: Path, missing_field: str, payload: dict[str, object]
) -> None:
    path = tmp_path / "live_operator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)

    with pytest.raises(ValueError, match=missing_field):
        LiveConfig.load(path)


@pytest.mark.parametrize(
    "injected_host",
    [
        " 192.0.2.11",
        "192.0.2.11/live",
        "user@192.0.2.11",
        "192.0.2.11:8554",
        "192.0.2.11?query=yes",
        "192.0.2.11#fragment",
        "192.0.2.11\nsecond-host",
        "999.999.999.999",
        "192.168.001.001",
    ],
)
def test_load_rejects_camera_host_injection(
    tmp_path: Path, injected_host: str
) -> None:
    path = tmp_path / "live_operator.json"
    camera_ips = dict(CAMERA_IPS)
    camera_ips["dianqi1"] = injected_host
    write_config(path, cameras=camera_ips)

    with pytest.raises(ValueError, match="cameras.dianqi1"):
        LiveConfig.load(path)


@pytest.mark.parametrize(("username", "password"), [("", "secret"), ("user", "")])
def test_direct_live_config_rejects_empty_credentials(
    tmp_path: Path, username: str, password: str
) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    loaded = LiveConfig.load(path)

    with pytest.raises(ValueError, match="username|password"):
        LiveConfig(username=username, password=password, cameras=loaded.cameras)


def test_direct_live_config_accepts_a_subset_and_any_camera_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    loaded = LiveConfig.load(path)

    subset = LiveConfig(
        username=loaded.username,
        password=loaded.password,
        cameras=loaded.cameras[:-1],
    )
    reordered = LiveConfig(
        username=loaded.username,
        password=loaded.password,
        cameras=tuple(reversed(loaded.cameras)),
    )

    assert len(subset.cameras) == 6
    assert reordered.cameras[0].relay == "camera07"


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("relay", "camera07", "duplicate camera relay"),
        ("view", "zoulang", "duplicate camera view"),
        ("calibration", "camera_corridor_screen_calibration_v21.json", "distinct calibration"),
    ],
)
def test_direct_live_config_rejects_duplicate_dynamic_identity(
    tmp_path: Path, field_name: str, bad_value: object, message: str
) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    loaded = LiveConfig.load(path)
    cameras = list(loaded.cameras)
    cameras[0] = replace(cameras[0], **{field_name: bad_value})

    with pytest.raises(ValueError, match=message):
        LiveConfig(
            username=loaded.username,
            password=loaded.password,
            cameras=tuple(cameras),
        )


def test_direct_live_config_rejects_camera_credentials_that_do_not_match(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live_operator.json"
    write_config(path)
    loaded = LiveConfig.load(path)
    cameras = list(loaded.cameras)
    cameras[0] = replace(cameras[0], password="different-fake-password")

    with pytest.raises(ValueError, match="credentials"):
        LiveConfig(
            username=loaded.username,
            password=loaded.password,
            cameras=tuple(cameras),
        )
