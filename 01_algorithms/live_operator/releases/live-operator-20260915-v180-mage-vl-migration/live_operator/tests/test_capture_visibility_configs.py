from __future__ import annotations

import json
from pathlib import Path

from live_operator.capture_visibility import load_visibility_sidecar


CONFIG_ROOT = Path(__file__).resolve().parents[6] / "02_configs" / "surveillance" / "capture_visibility_v1"
ENABLED_CAMERAS = {
    "ffs_d4",
    "camera08",
    "camera09",
    "camera10",
    "camera11",
    "camera13",
    "camera14",
    "camera15",
}


def test_sidecars_cover_enabled_screen_cameras() -> None:
    paths = sorted(CONFIG_ROOT.glob("*.json"))

    assert {path.stem for path in paths} == ENABLED_CAMERAS
    assert {load_visibility_sidecar(path).camera for path in paths} == ENABLED_CAMERAS


def test_initial_sidecars_are_conservative_and_keep_calibrated_screen_pixels() -> None:
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = load_visibility_sidecar(path)

        assert raw["source_calibration_sha256"]
        assert raw["calibration_status"] == "screen-polygons-and-near-zones-only"
        assert config.frame_width == 2560
        assert config.frame_height == 1440
        assert config.screens
        assert config.occluders == {}
        for raw_screen in raw["screens"]:
            assert len(raw_screen["screen_polygon"]) >= 3
            assert raw_screen["capture_position_zones"]
            assert raw_screen["blocked_position_zones"] == []
            assert raw_screen["unknown_position_zones"] == [
                {
                    "zone_id": f'{raw_screen["screen_id"]}_unverified_rest',
                    "polygon": [[0, 0], [2560, 0], [2560, 1440], [0, 1440]],
                }
            ]
