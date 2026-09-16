from __future__ import annotations

import json
from pathlib import Path

import pytest

from live_operator.capture_visibility import (
    evaluate_visibility,
    load_visibility_sidecar,
)


def write_sidecar(tmp_path: Path, payload: dict | None = None) -> Path:
    body = payload or {
        "schema_version": 1,
        "camera": "camera08",
        "frame_width": 2560,
        "frame_height": 1440,
        "screens": [
            {
                "screen_id": "screen_01",
                "capture_position_zones": [
                    {
                        "zone_id": "seat_a",
                        "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]],
                    }
                ],
                "blocked_position_zones": [
                    {
                        "zone_id": "seat_behind_partition",
                        "occluder_id": "partition_a",
                        "polygon": [[100, 50], [180, 50], [180, 150], [100, 150]],
                    }
                ],
                "unknown_position_zones": [
                    {
                        "zone_id": "far_corner",
                        "polygon": [[300, 300], [400, 300], [400, 400]],
                    }
                ],
            }
        ],
        "occluders": [
            {
                "occluder_id": "partition_a",
                "polygon": [[90, 0], [95, 0], [95, 250], [90, 250]],
            }
        ],
    }
    path = tmp_path / "camera08.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_verified_blocked_zone_wins_over_overlapping_possible_zone(
    tmp_path: Path,
) -> None:
    config = load_visibility_sidecar(write_sidecar(tmp_path))

    relation = evaluate_visibility(
        config,
        screen_id="screen_01",
        anchor=(120.0, 90.0),
    )

    assert relation.state == "blocked"
    assert relation.zone_id == "seat_behind_partition"
    assert relation.occluder_id == "partition_a"


def test_possible_and_unlabelled_positions_remain_distinct(tmp_path: Path) -> None:
    config = load_visibility_sidecar(write_sidecar(tmp_path))

    possible = evaluate_visibility(config, screen_id="screen_01", anchor=(20.0, 20.0))
    unknown = evaluate_visibility(config, screen_id="screen_01", anchor=(900.0, 900.0))

    assert possible.state == "possible"
    assert possible.zone_id == "seat_a"
    assert unknown.state == "unknown"
    assert unknown.zone_id is None


def test_explicit_unknown_zone_is_reported_for_audit(tmp_path: Path) -> None:
    config = load_visibility_sidecar(write_sidecar(tmp_path))

    relation = evaluate_visibility(
        config,
        screen_id="screen_01",
        anchor=(350.0, 350.0),
    )

    assert relation.state == "unknown"
    assert relation.zone_id == "far_corner"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda body: body.update(schema_version=2), "schema_version"),
        (lambda body: body.update(frame_width=0), "frame_width"),
        (
            lambda body: body["screens"][0]["capture_position_zones"].append(
                body["screens"][0]["capture_position_zones"][0]
            ),
            "duplicate zone_id",
        ),
        (
            lambda body: body["screens"][0]["blocked_position_zones"][0].update(
                occluder_id="missing"
            ),
            "unknown occluder_id",
        ),
        (
            lambda body: body["screens"][0]["capture_position_zones"][0].update(
                polygon=[[0, 0], [10, 10], [0, 0]]
            ),
            "three distinct points",
        ),
        (
            lambda body: body["screens"][0]["capture_position_zones"][0].update(
                polygon=[[0, 0], [3000, 0], [10, 10]]
            ),
            "outside frame",
        ),
    ],
)
def test_invalid_sidecars_are_rejected(tmp_path: Path, mutation, message: str) -> None:
    source = json.loads(write_sidecar(tmp_path).read_text(encoding="utf-8"))
    mutation(source)

    with pytest.raises(ValueError, match=message):
        load_visibility_sidecar(write_sidecar(tmp_path, source))


def test_unknown_screen_id_is_rejected_at_evaluation(tmp_path: Path) -> None:
    config = load_visibility_sidecar(write_sidecar(tmp_path))

    with pytest.raises(ValueError, match="unknown screen_id"):
        evaluate_visibility(config, screen_id="screen_missing", anchor=(1.0, 1.0))

