from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from live_operator.cli import RuntimeHooks
from live_operator.config import CameraConfig, LiveConfig
from live_operator.processes import ProcessIdentity


def _config() -> LiveConfig:
    return LiveConfig(
        cameras=(
            CameraConfig(
                relay="camera08",
                view="ffs",
                ip="192.0.2.80",
                calibration="ffs.json",
                username="camera-user",
                password="not-a-real-password",
                enabled=True,
            ),
            CameraConfig(
                relay="camera09",
                view="archive_room",
                ip="192.0.2.81",
                calibration="archive.json",
                username="camera-user",
                password="not-a-real-password",
                has_screen=False,
                enabled=False,
            ),
        )
    )


def test_deepstream_manifest_contains_only_enabled_dynamic_cameras(
    tmp_path: Path, monkeypatch
) -> None:
    import live_operator.cli as cli_module

    release = tmp_path / "release"
    launcher = release / "deepstream" / "custom_pipeline" / "scripts" / "run_container_50p2.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "__file__", str(release / "live_operator" / "cli.py"))
    monkeypatch.setattr(cli_module.os, "access", lambda path, _mode: Path(path) == launcher)
    config_path = tmp_path / "live_operator.json"
    _config().save(config_path)

    version = SimpleNamespace(
        deepstream_binary=tmp_path / "pipeline",
        deepstream_binary_sha256="a" * 64,
        pose_plan=tmp_path / "pose.plan",
        phone_engine=tmp_path / "phone.engine",
        calibration_dir=tmp_path / "calibrations",
    )
    observed = []

    class Process:
        pid = 123

    monkeypatch.setattr(cli_module, "resolve_current_inference_version", lambda: version)
    monkeypatch.setattr(
        cli_module.subprocess,
        "Popen",
        lambda command, **kwargs: observed.append((command, kwargs)) or Process(),
    )
    monkeypatch.setattr(
        cli_module,
        "capture_identity",
        lambda process, owner_token=None: ProcessIdentity(process.pid, "start", 123, owner_token),
    )

    hooks = RuntimeHooks()
    hooks.config_path = config_path
    hooks.start_component("deepstream", tmp_path / "run", _config())

    environment = observed[0][1]["env"]
    manifest = Path(environment["CAMERA_MANIFEST_FILE"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["cameras"] == [
        {
            "relay": "camera08",
            "view": "ffs",
            "calibration": "ffs.json",
            "has_screen": True,
        }
    ]

