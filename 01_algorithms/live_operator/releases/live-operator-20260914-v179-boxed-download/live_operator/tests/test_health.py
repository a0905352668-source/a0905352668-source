from live_operator.health import build_public_health


def test_public_status_preserves_safe_unicode_camera_view() -> None:
    payload = build_public_health(
        {
            "run_id": "live_20260806_175746",
            "state": "running",
            "aggregate_fps": 55.6,
            "updated_at": 100.0,
            "cameras": [
                {
                    "relay": "camera13",
                    "view": "总工办资料室-1",
                    "status": "online",
                    "fps": 8.0,
                },
                {
                    "relay": "camera14",
                    "view": "rtsp://must-not-be-public",
                    "status": "online",
                    "fps": 8.0,
                },
            ],
        },
        {},
        {},
        {},
        100.0,
    )

    assert payload["cameras"] == [
        {
            "relay": "camera13",
            "view": "总工办资料室-1",
            "status": "online",
            "fps": 8.0,
        },
        {
            "relay": "camera14",
            "view": "",
            "status": "online",
            "fps": 8.0,
        },
    ]


def test_degraded_camera_is_not_mislabeled_as_offline() -> None:
    payload = build_public_health(
        {
            "run_id": "live-run",
            "state": "running",
            "aggregate_fps": 8.0,
            "updated_at": 100.0,
            "cameras": [
                {
                    "relay": "camera09",
                    "view": "FFS-3",
                    "status": "degraded",
                    "fps": 7.95,
                    "source_errors": 1,
                }
            ],
        },
        {},
        {},
        {},
        100.0,
    )

    assert [alert["code"] for alert in payload["alerts"]] == [
        "camera_source_error"
    ]


def test_unknown_process_observation_is_not_a_user_facing_incident() -> None:
    observing = build_public_health(
        {"state": "running", "updated_at": 100.0, "cameras": []},
        {},
        {
            "state": "blocked",
            "reason_code": "unknown_process_observing",
            "updated_at": 100.0,
        },
        {},
        100.0,
    )
    recovered = build_public_health(
        {"state": "running", "updated_at": 120.0, "cameras": []},
        {},
        {
            "state": "healthy",
            "reason_code": "healthy",
            "updated_at": 120.0,
            "last_incident": {
                "active": False,
                "reason_code": "unknown_process_observing",
                "message": "An unowned process was awaiting confirmation.",
                "observed_at": 100.0,
                "resolved_at": 118.0,
            },
        },
        {},
        120.0,
    )

    assert observing["alerts"] == []
    assert recovered["alerts"] == []
    assert recovered["health"]["last_incident"]["reason_code"] == (
        "unknown_process_observing"
    )
