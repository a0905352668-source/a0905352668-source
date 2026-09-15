import json
import subprocess
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import pytest

from live_operator.dashboard import DashboardApp
from live_operator.config import CameraConfig


def response_bytes(response: object) -> bytes:
    body = response.body
    return body if isinstance(body, bytes) else b"".join(body)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_ready_events(run_dir: Path, events: list[dict[str, object]]) -> None:
    dashboard = run_dir / "dashboard"
    write_json(dashboard / "events.json", events)
    clips = dashboard / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for event in events:
        event_id = event["event_id"]
        (clips / f"{event_id}.mp4").write_bytes(b"clip")
        write_json(clips / f"{event_id}.json", {"event_id": event_id})


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    dashboard = tmp_path / "dashboard"
    write_json(
        dashboard / "status.json",
        {
            "run_id": "run-20260717-120000",
            "state": "running",
            "aggregate_fps": 56.0,
            "cameras": [
                {"relay": "camera01", "view": "dianqi1", "status": "online", "fps": 8.0},
                {"relay": "camera02", "view": "dianqi2", "status": "online", "fps": 8.0},
            ],
        },
    )
    write_json(dashboard / "events.json", [])
    return tmp_path


def test_home_and_static_assets_are_utf8_and_use_event_canvas_only(run_dir: Path) -> None:
    app = DashboardApp(run_dir)

    home = app.handle("GET", "/")
    script = app.handle("GET", "/static/app.js")
    styles = app.handle("GET", "/static/styles.css")

    assert home.status == 200
    assert home.headers["Content-Type"].startswith("text/html; charset=utf-8")
    html = home.body.decode("utf-8")
    assert "\ufffd" not in html
    assert '<meta charset="utf-8">' in html.lower()
    assert 'id="event-canvas"' in html
    assert 'id="event-fullscreen"' in html
    assert 'id="aggregate-fps"' not in html
    assert 'id="event-person"' not in html
    assert "camera-status" in html
    assert "latest-events" in html
    assert all(
        f'id="{element_id}"' in html
        for element_id in (
            "event-calendar",
            "calendar-prev",
            "calendar-next",
            "calendar-grid",
            "hour-filters",
            "event-pagination",
            "event-page-prev",
            "event-page-label",
            "event-page-next",
        )
    )
    assert 'id="review-actions"' in html
    assert 'id="review-error"' in html
    assert 'id="show-false-positives"' in html
    assert 'id="review-toast-undo"' in html
    assert html.count('data-review-result="') == 3
    assert all(
        f'data-review-result="{result}"' in html
        for result in ("pending", "confirmed", "false_positive")
    )
    assert "autoplay" not in html.lower()
    assert script.status == 200
    assert styles.status == 200
    javascript = script.body.decode("utf-8")
    assert "getContext(\"2d\")" in javascript
    assert "videoStage.requestFullscreen" in javascript
    assert "fullscreenchange" in javascript
    assert "camera.fps" not in javascript
    assert "人员 ID" not in javascript
    assert "正在收集报警后画面" not in javascript
    assert "片段整理中" not in javascript
    assert all(label in javascript for label in ("person", "phone", "screen"))
    assert javascript.count('label: "疑似拍屏"') == 2
    assert 'label: "人员"' not in javascript
    assert 'label: "候选"' not in javascript
    format_timestamp_source = javascript[
        javascript.index("function formatTimestamp") : javascript.index("function cameraLabel")
    ]
    assert 'timeZone: "Asia/Shanghai"' in format_timestamp_source
    assert 'button.setAttribute("aria-current", "date")' in javascript
    assert 'button.setAttribute("aria-pressed", String(isSelected))' in javascript
    stylesheet = styles.body.decode("utf-8")
    assert "video::-webkit-media-controls-fullscreen-button" in stylesheet
    assert ".calendar-grid" in stylesheet
    assert "grid-template-columns: repeat(7" in stylesheet
    assert ".review-controls" in stylesheet
    assert ".review-badge" in stylesheet
    assert ".event-pagination" in stylesheet
    assert all(token not in javascript for token in ("getUserMedia", "webrtc", "rtsp://"))


def test_status_api_returns_current_run_and_camera_fps(run_dir: Path) -> None:
    response = DashboardApp(run_dir).handle("GET", "/api/status")

    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-store"
    payload = json.loads(response.body)
    assert payload["run_id"] == "run-20260717-120000"
    assert payload["aggregate_fps"] == 56.0
    assert payload["cameras"][0] == {
        "relay": "camera01",
        "view": "dianqi1",
        "status": "online",
        "fps": 8.0,
    }


def test_status_api_reports_a_recent_recovered_watchdog_incident(run_dir: Path) -> None:
    lifecycle = run_dir / "lifecycle.json"
    watchdog = run_dir / "watchdog.json"
    write_json(lifecycle, {"target_fps_per_stream": 8})
    write_json(
        watchdog,
        {
            "state": "healthy",
            "reason_code": "healthy",
            "updated_at": 120.0,
            "last_incident": {
                "active": False,
                "reason_code": "unknown_process",
                "message": "An unowned runtime process was detected.",
                "observed_at": 100.0,
                "resolved_at": 118.0,
            },
        },
    )

    response = DashboardApp(
        run_dir,
        lifecycle_state_path=lifecycle,
        watchdog_health_path=watchdog,
        clock=lambda: 120.0,
    ).handle("GET", "/api/status")

    payload = json.loads(response.body)
    assert payload["health"]["last_incident"]["active"] is False
    assert payload["health"]["last_incident"]["reason_code"] == "unknown_process"
    assert payload["alerts"] == [
        {
            "code": "watchdog_recovered",
            "severity": "info",
            "message": "守护异常已恢复：unknown_process",
        }
    ]


def test_status_api_zeros_stale_cached_inference_measurements(run_dir: Path) -> None:
    write_json(
        run_dir / "dashboard" / "status.json",
        {
            "run_id": "run-20260717-120000",
            "state": "running",
            "aggregate_fps": 56.0,
            "updated_at": 100.0,
            "cameras": [
                {
                    "relay": "camera01",
                    "view": "dianqi1",
                    "status": "online",
                    "fps": 8.0,
                }
            ],
        },
    )

    response = DashboardApp(run_dir, clock=lambda: 106.0).handle(
        "GET", "/api/status"
    )

    payload = json.loads(response.body)
    assert payload["state"] == "stale"
    assert payload["aggregate_fps"] == 0.0
    assert payload["cameras"][0]["status"] == "unknown"
    assert payload["cameras"][0]["fps"] == 0.0
    assert [alert["code"] for alert in payload["alerts"]] == ["status_stale"]


def test_api_recursively_redacts_secrets_inside_public_string_values(
    run_dir: Path,
) -> None:
    username = "dashboard-user"
    password = "fake-dashboard-p@ss:/"
    CameraConfig(
        relay="camera01",
        view="dianqi1",
        ip="192.0.2.11",
        calibration="camera_01_screen_calibration_v21.json",
        has_screen=False,
        username=username,
        password=password,
    )
    encoded_user = quote(username, safe="")
    encoded_password = quote(password, safe="")
    write_json(
        run_dir / "dashboard" / "status.json",
        {
            "state": "failed",
            "error": f"raw={password}",
            "message": f"encoded={encoded_password}",
            "detail": {
                "raw_userinfo": f"{username}:{password}",
                "url": f"rtsp://{encoded_user}:{encoded_password}@192.0.2.11/live",
            },
        },
    )

    raw = response_bytes(DashboardApp(run_dir).handle("GET", "/api/status")).decode()

    assert password not in raw
    assert encoded_password not in raw
    assert f"{username}:{password}" not in raw
    assert f"{encoded_user}:{encoded_password}" not in raw
    assert "[REDACTED]" in raw


def test_events_api_observes_updates_and_adds_ready_clip_overlay_urls(
    run_dir: Path,
) -> None:
    app = DashboardApp(run_dir)
    dashboard = run_dir / "dashboard"
    legacy = dashboard / "clips" / "older-result.mp4"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"old-result")
    initial = json.loads(app.handle("GET", "/api/events").body)
    assert initial["events"] == []
    assert initial["pagination"] == {
        "page": 1,
        "page_size": 20,
        "total": 0,
        "total_pages": 0,
    }

    event = {
        "event_id": "camera-01_person-7_alarm-10000",
        "camera": "camera01",
        "person_id": "7",
        "status": "ready",
        "occurred_at": "2026-07-17T12:00:10.000000+08:00",
        "risk_peak": 0.91,
        "bbox_timeline": [{"time_sec": 10.0, "bbox": [1, 2, 3, 4]}],
    }
    write_json(dashboard / "events.json", [event])
    (dashboard / "clips" / f"{event['event_id']}.mp4").write_bytes(b"new-clip")
    write_json(
        dashboard / "clips" / f"{event['event_id']}.json",
        {"event_id": event["event_id"], "overlay": {"bbox_timeline": []}},
    )

    payload = json.loads(app.handle("GET", "/api/events").body)

    assert payload["events"] == [
        {
            "event_id": event["event_id"],
            "camera": event["camera"],
            "status": event["status"],
            "occurred_at": event["occurred_at"],
            "risk_peak": event["risk_peak"],
            "review_result": "pending",
            "clip_url": f"/clips/{event['event_id']}.mp4",
            "overlay_url": f"/clips/{event['event_id']}.json",
        }
    ]
    assert payload["selection"] == {
        "camera": "",
        "month": "2026-07",
        "date": "2026-07-17",
        "hour": "12",
    }
    assert payload["calendar"] == {"2026-07-17": 1}
    assert payload["hours"] == {"2026-07-17T12": 1}
    assert payload["summary"] == {
        "events": 1,
        "alarms": 0,
        "views": 1,
        "max_risk": 0.91,
    }
    assert "bbox_timeline" not in payload["events"][0]
    assert legacy.read_bytes() == b"old-result"


def test_event_cache_warmup_avoids_scan_on_first_api_request(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = {
        "event_id": "warm-event",
        "camera": "camera01",
        "status": "ready",
        "occurred_at": "2026-07-17T12:00:10+08:00",
        "risk_peak": 0.91,
    }
    write_ready_events(run_dir, [event])
    app = DashboardApp(run_dir)

    warmup = app.start_event_cache_warmup()
    warmup.join(timeout=2.0)
    assert not warmup.is_alive()

    def unexpected_refresh(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("first API request rescanned event history")

    monkeypatch.setattr(app, "_refresh_index_cache", unexpected_refresh)
    payload = json.loads(app.handle("GET", "/api/events").body)

    assert [item["event_id"] for item in payload["events"]] == ["warm-event"]


def test_events_api_can_retrieve_an_old_date_beyond_200_newer_events(
    run_dir: Path,
) -> None:
    dashboard = run_dir / "dashboard"
    events = [
        {
            "event_id": f"event-{index:03d}",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": f"2026-07-20T{12 + index // 60:02d}:{index % 60:02d}:00+08:00",
            "risk_peak": 0.9,
        }
        for index in range(205)
    ]
    old_event = {
        "event_id": "old-event",
        "camera": "camera01",
        "status": "ready",
        "occurred_at": "2026-07-01T08:30:00+08:00",
        "risk_peak": 0.7,
    }
    write_ready_events(run_dir, [old_event, *events])

    payload = json.loads(
        DashboardApp(run_dir).handle(
            "GET", "/api/events", query_string="date=2026-07-01&hour=08"
        ).body
    )

    assert [event["event_id"] for event in payload["events"]] == ["old-event"]
    assert payload["pagination"]["total"] == 1
    assert payload["calendar"] == {"2026-07-01": 1, "2026-07-20": 205}


def test_collecting_events_cannot_evict_older_ready_events(run_dir: Path) -> None:
    dashboard = run_dir / "dashboard"
    ready = {
        "event_id": "ready-event",
        "camera": "camera01",
        "status": "ready",
        "occurred_at": "2026-07-20T12:00:00+08:00",
        "risk_peak": 0.9,
    }
    collecting = [
        {
            "event_id": f"collecting-{index:03d}",
            "camera": "camera01",
            "status": "collecting",
            "occurred_at": "2026-07-20T12:01:00+08:00",
            "risk_peak": 0.9,
        }
        for index in range(205)
    ]
    write_json(dashboard / "events.json", [ready, *collecting])
    clips = dashboard / "clips"
    clips.mkdir(parents=True)
    (clips / "ready-event.mp4").write_bytes(b"clip")
    write_json(clips / "ready-event.json", {"event_id": "ready-event"})

    payload = json.loads(DashboardApp(run_dir).handle("GET", "/api/events").body)

    assert [event["event_id"] for event in payload["events"]] == ["ready-event"]


def test_events_api_filters_counts_sorts_and_paginates_full_result_set(
    run_dir: Path,
) -> None:
    matching = [
        {
            "event_id": f"match-{index:02d}",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": f"2026-07-20T14:{index:02d}:00+08:00",
            "level": "alarm" if index % 2 == 0 else "risk",
            "risk_peak": 0.5 + index / 100,
        }
        for index in range(25)
    ]
    other_events = [
        {
            "event_id": "other-hour",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-07-20T15:00:00+08:00",
            "level": "alarm",
            "risk_peak": 0.99,
        },
        {
            "event_id": "other-camera",
            "camera": "camera02",
            "status": "ready",
            "occurred_at": "2026-07-20T14:59:00+08:00",
            "level": "alarm",
            "risk_peak": 1.0,
        },
    ]
    write_ready_events(run_dir, [*matching, *other_events])
    app = DashboardApp(run_dir)
    query = "month=2026-07&date=2026-07-20&hour=14&camera=camera01&page_size=10"

    first = json.loads(app.handle("GET", "/api/events", query_string=query).body)
    second = json.loads(
        app.handle("GET", "/api/events", query_string=f"{query}&page=2").body
    )

    assert [event["event_id"] for event in first["events"]] == [
        f"match-{index:02d}" for index in range(24, 14, -1)
    ]
    assert [event["event_id"] for event in second["events"]] == [
        f"match-{index:02d}" for index in range(14, 4, -1)
    ]
    assert first["pagination"] == {
        "page": 1,
        "page_size": 10,
        "total": 25,
        "total_pages": 3,
    }
    assert second["pagination"]["page"] == 2
    assert first["selection"] == {
        "camera": "camera01",
        "month": "2026-07",
        "date": "2026-07-20",
        "hour": "14",
    }
    assert first["calendar"] == {"2026-07-20": 26}
    assert first["hours"] == {"2026-07-20T14": 25, "2026-07-20T15": 1}
    assert first["summary"] == second["summary"] == {
        "events": 25,
        "alarms": 13,
        "views": 1,
        "max_risk": 0.74,
    }


def test_events_api_can_hide_or_include_reviewed_false_positives(
    run_dir: Path,
) -> None:
    events = [
        {
            "event_id": "pending-event",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-07-20T14:01:00+08:00",
            "level": "alarm",
            "risk_peak": 0.91,
        },
        {
            "event_id": "false-positive-event",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-07-20T14:02:00+08:00",
            "level": "alarm",
            "risk_peak": 0.93,
        },
    ]
    write_ready_events(run_dir, events)
    write_json(
        run_dir / "dashboard" / "event_reviews.json",
        {
            "false-positive-event": {
                "result": "false_positive",
                "updated_at": "2026-07-20T14:03:00+08:00",
            }
        },
    )
    app = DashboardApp(run_dir)
    base_query = "date=2026-07-20&hour=14&page_size=20"

    hidden = json.loads(
        app.handle(
            "GET",
            "/api/events",
            query_string=f"{base_query}&include_false_positives=0",
        ).body
    )
    included = json.loads(
        app.handle(
            "GET",
            "/api/events",
            query_string=f"{base_query}&include_false_positives=1",
        ).body
    )

    assert [event["event_id"] for event in hidden["events"]] == ["pending-event"]
    assert hidden["pagination"]["total"] == 1
    assert [event["event_id"] for event in included["events"]] == [
        "false-positive-event",
        "pending-event",
    ]
    assert included["pagination"]["total"] == 2


def test_events_api_uses_event_id_to_stabilize_equal_timestamp_pages(
    run_dir: Path,
) -> None:
    events = [
        {
            "event_id": event_id,
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-07-20T14:00:00+08:00",
        }
        for event_id in ("event-d", "event-b", "event-c", "event-a")
    ]
    write_ready_events(run_dir, events)
    app = DashboardApp(run_dir)
    query = "date=2026-07-20&hour=14&page_size=2"

    first = json.loads(app.handle("GET", "/api/events", query_string=query).body)
    second = json.loads(
        app.handle("GET", "/api/events", query_string=f"{query}&page=2").body
    )

    assert [event["event_id"] for event in first["events"]] == [
        "event-a",
        "event-b",
    ]
    assert [event["event_id"] for event in second["events"]] == [
        "event-c",
        "event-d",
    ]


def test_events_api_converts_utc_timestamps_to_asia_shanghai(run_dir: Path) -> None:
    events = [
        {
            "event_id": "before-midnight-utc",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-07-20T15:59:59Z",
            "risk_peak": 0.4,
        },
        {
            "event_id": "after-midnight-beijing",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-07-20T16:00:00Z",
            "risk_peak": 0.8,
        },
    ]
    write_ready_events(run_dir, events)

    payload = json.loads(
        DashboardApp(run_dir).handle(
            "GET", "/api/events", query_string="date=2026-07-21&hour=00"
        ).body
    )

    assert [event["event_id"] for event in payload["events"]] == [
        "after-midnight-beijing"
    ]
    assert payload["calendar"] == {"2026-07-20": 1, "2026-07-21": 1}
    assert payload["hours"] == {"2026-07-21T00": 1}


def test_events_api_without_date_or_hour_selects_latest_available(
    run_dir: Path,
) -> None:
    events = [
        {
            "event_id": "old",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-06-30T23:59:00+08:00",
        },
        {
            "event_id": "latest",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-07-02T03:00:00+08:00",
        },
    ]
    write_ready_events(run_dir, events)

    payload = json.loads(DashboardApp(run_dir).handle("GET", "/api/events").body)

    assert payload["selection"] == {
        "camera": "",
        "month": "2026-07",
        "date": "2026-07-02",
        "hour": "03",
    }
    assert [event["event_id"] for event in payload["events"]] == ["latest"]


def test_events_api_only_validates_files_for_the_requested_page(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        {
            "event_id": f"event-{index:03d}",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": f"2026-07-20T14:{index % 60:02d}:{index // 60:02d}+08:00",
            "risk_peak": 0.9,
        }
        for index in range(120)
    ]
    write_ready_events(run_dir, events)
    app = DashboardApp(run_dir)
    original = app._decorate_indexed_record
    validated: list[str] = []

    def record_validation(indexed_record):
        validated.append(indexed_record.event_id)
        return original(indexed_record)

    monkeypatch.setattr(app, "_decorate_indexed_record", record_validation)

    response = app.handle(
        "GET",
        "/api/events",
        query_string=(
            "month=2026-07&date=2026-07-20&hour=14"
            "&camera=camera01&page=1&page_size=10"
        ),
    )

    assert response.status == 200
    assert len(json.loads(response.body)["events"]) == 10
    assert len(validated) == 10


def test_explicit_empty_selection_does_not_fall_back_and_page_is_clamped(
    run_dir: Path,
) -> None:
    events = [
        {
            "event_id": f"event-{index}",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": f"2026-07-20T14:{index:02d}:00+08:00",
        }
        for index in range(3)
    ]
    write_ready_events(run_dir, events)
    app = DashboardApp(run_dir)

    empty = json.loads(
        app.handle(
            "GET", "/api/events", query_string="date=2026-07-19&hour=08&page=9"
        ).body
    )
    clamped = json.loads(
        app.handle(
            "GET",
            "/api/events",
            query_string="date=2026-07-20&hour=14&page=9&page_size=2",
        ).body
    )

    assert empty["selection"]["date"] == "2026-07-19"
    assert empty["selection"]["hour"] == "08"
    assert empty["events"] == []
    assert empty["pagination"] == {
        "page": 1,
        "page_size": 20,
        "total": 0,
        "total_pages": 0,
    }
    assert clamped["pagination"]["page"] == 2
    assert [event["event_id"] for event in clamped["events"]] == ["event-0"]


@pytest.mark.parametrize(
    "query_string",
    (
        "month=2026-7",
        "month=2026-13",
        "date=2026-7-20",
        "date=2026-02-30",
        "hour=1",
        "hour=24",
        "page=0",
        "page=-1",
        "page=1.0",
        "page_size=0",
        "page_size=101",
        "page_size=abc",
        "include_false_positives=2",
        "include_false_positives=true",
        "include_false_positives=0&include_false_positives=1",
        "date=2026-07-20&date=2026-07-21",
    ),
)
def test_events_api_rejects_malformed_query_parameters(
    run_dir: Path, query_string: str
) -> None:
    response = DashboardApp(run_dir).handle(
        "GET", "/api/events", query_string=query_string
    )

    assert response.status == 400
    assert "error" in json.loads(response.body)


def test_malformed_head_events_query_has_get_content_length_and_empty_body(
    run_dir: Path,
) -> None:
    app = DashboardApp(run_dir)

    get_response = app.handle("GET", "/api/events", query_string="hour=24")
    head_response = app.handle("HEAD", "/api/events", query_string="hour=24")

    assert get_response.status == head_response.status == 400
    assert head_response.headers["Content-Length"] == str(len(get_response.body))
    assert head_response.headers["Content-Length"] == get_response.headers["Content-Length"]
    assert head_response.body == b""


def test_wsgi_query_string_is_forwarded_to_events_handler(run_dir: Path) -> None:
    events = [
        {
            "event_id": "camera-one",
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-07-20T14:00:00+08:00",
        },
        {
            "event_id": "camera-two",
            "camera": "camera02",
            "status": "ready",
            "occurred_at": "2026-07-20T14:01:00+08:00",
        },
    ]
    write_ready_events(run_dir, events)
    statuses: list[str] = []

    body = b"".join(
        DashboardApp(run_dir)(
            {
                "REQUEST_METHOD": "GET",
                "PATH_INFO": "/api/events",
                "QUERY_STRING": "date=2026-07-20&hour=14&camera=camera02",
                "wsgi.input": BytesIO(),
            },
            lambda status, _headers: statuses.append(status),
        )
    )

    assert statuses == ["200 OK"]
    assert [event["event_id"] for event in json.loads(body)["events"]] == [
        "camera-two"
    ]


def test_full_clip_request_returns_200_and_range_capability(run_dir: Path) -> None:
    clip = run_dir / "dashboard" / "clips" / "event-1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"0123456789")

    response = DashboardApp(run_dir).handle("GET", "/clips/event-1.mp4")

    assert response.status == 200
    assert not isinstance(response.body, bytes)
    assert response_bytes(response) == b"0123456789"
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.headers["Content-Length"] == "10"
    assert response.headers["Content-Type"] == "video/mp4"
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"


@pytest.mark.parametrize(
    ("range_header", "expected", "content_range"),
    [
        ("bytes=2-5", b"2345", "bytes 2-5/10"),
        ("bytes=6-", b"6789", "bytes 6-9/10"),
        ("bytes=-3", b"789", "bytes 7-9/10"),
    ],
)
def test_clip_byte_ranges_return_206(
    run_dir: Path, range_header: str, expected: bytes, content_range: str
) -> None:
    clip = run_dir / "dashboard" / "clips" / "event-1.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"0123456789")

    response = DashboardApp(run_dir).handle(
        "GET", "/clips/event-1.mp4", {"Range": range_header}
    )

    assert response.status == 206
    assert response_bytes(response) == expected
    assert response.headers["Content-Range"] == content_range
    assert response.headers["Content-Length"] == str(len(expected))
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"


@pytest.mark.parametrize(
    "range_header", ["bytes=20-30", "bytes=5-2", "items=0-1", "bytes=0-1,4-5"]
)
def test_invalid_or_unsatisfiable_ranges_return_416(
    run_dir: Path, range_header: str
) -> None:
    clip = run_dir / "dashboard" / "clips" / "event-1.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"0123456789")

    response = DashboardApp(run_dir).handle(
        "GET", "/clips/event-1.mp4", {"Range": range_header}
    )

    assert response.status == 416
    assert response.body == b""
    assert response.headers["Content-Range"] == "bytes */10"


def test_clip_path_traversal_and_unknown_routes_are_not_exposed(run_dir: Path) -> None:
    app = DashboardApp(run_dir)

    assert app.handle("GET", "/clips/../events.json").status == 404
    assert app.handle("GET", "/clips/%2e%2e%2fevents.json").status == 404
    assert app.handle("GET", "/clips/%252e%252e%252fevents.json").status == 404
    assert app.handle("POST", "/api/events").status == 405
    assert app.handle("GET", "/streams/camera01").status == 404


def test_clip_symlink_cannot_escape_current_dashboard(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = run_dir / "outside-secret.mp4"
    outside.write_bytes(b"must-not-be-served")
    clips = run_dir / "dashboard" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    link = clips / "event-link.mp4"
    try:
        link.symlink_to(outside)
    except OSError:
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda self: self == link or original_is_symlink(self),
        )

    assert DashboardApp(run_dir).handle("GET", "/clips/event-link.mp4").status == 404


def test_overlay_has_immutable_cache_policy(run_dir: Path) -> None:
    overlay = run_dir / "dashboard" / "clips" / "event-1.json"
    write_json(overlay, {"event_id": "event-1", "overlay": {}})

    response = DashboardApp(run_dir).handle("GET", "/clips/event-1.json")

    assert response.status == 200
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_large_full_clip_is_yielded_in_bounded_chunks(run_dir: Path) -> None:
    clip = run_dir / "dashboard" / "clips" / "large.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    expected = b"x" * 200_000
    clip.write_bytes(expected)

    response = DashboardApp(run_dir).handle("GET", "/clips/large.mp4")
    chunks = list(response.body)

    assert response.status == 200
    assert b"".join(chunks) == expected
    assert len(chunks) > 1
    assert max(map(len, chunks)) <= 64 * 1024


def test_frontend_generation_media_and_overlay_logic_executes_in_node() -> None:
    app_js = Path(__file__).parents[1] / "static" / "app.js"
    script = f"""
const assert = require('assert');
const logic = require({json.dumps(str(app_js))});

const refreshGate = logic.createGenerationGate();
const oldRefresh = refreshGate.begin();
const newRefresh = refreshGate.begin();
assert.strictEqual(refreshGate.isCurrent(oldRefresh), false);
assert.strictEqual(refreshGate.isCurrent(newRefresh), true);

const collecting = {{event_id: 'A', status: 'collecting'}};
const ready = {{event_id: 'A', status: 'ready', clip_url: '/clips/A.mp4', overlay_url: '/clips/A.json'}};
assert.notStrictEqual(logic.eventMediaKey(collecting), logic.eventMediaKey(ready));

const selectionGate = logic.createGenerationGate();
const selectA = selectionGate.begin();
const selectB = selectionGate.begin();
assert.strictEqual(selectionGate.isCurrent(selectA), false);
assert.strictEqual(selectionGate.isCurrent(selectB), true);

const timeline = [
  {{time_sec: 1.0, bbox: [1, 2, 3, 4]}},
  {{time_sec: 1.2, bbox: [2, 3, 4, 5]}},
];
assert.strictEqual(logic.selectOverlaySample(timeline, 0.99, 0.25), null);
assert.strictEqual(logic.selectOverlaySample(timeline, 1.1, 0.25).time_sec, 1.0);
assert.strictEqual(logic.selectOverlaySample(timeline, 1.46, 0.25), null);
assert.strictEqual(logic.selectOverlaySample([{{time_sec: 1.0}}], 1.6).time_sec, 1.0);
assert.deepStrictEqual(
  logic.overlayBoxesForSample({{
    bbox: [1, 2, 30, 40],
    track_id: "7",
    phone_boxes: [{{box: [10, 11, 16, 22], phone_score: 0.91}}],
  }}),
  [
    {{label: "person", bbox: [1, 2, 30, 40], track_id: "7"}},
    {{label: "phone", box: [10, 11, 16, 22], phone_score: 0.91, bbox: [10, 11, 16, 22]}},
  ],
);
assert.deepStrictEqual(logic.overlayBoxesForSample({{phone_boxes: [{{box: "bad"}}]}}), []);
const simultaneousTimeline = [
  {{time_sec: 1.0, track_id: "7", bbox: [1, 2, 3, 4]}},
  {{time_sec: 1.0, track_id: "9", bbox: [5, 6, 7, 8]}},
  {{time_sec: 1.2, track_id: "7", bbox: [2, 3, 4, 5]}},
  {{time_sec: 1.2, track_id: "9", bbox: [6, 7, 8, 9]}},
];
assert.deepStrictEqual(
  logic.selectOverlaySamples(simultaneousTimeline, 1.25, 0.25).map((sample) => sample.track_id),
  ["7", "9"],
);
assert.deepStrictEqual(logic.selectOverlaySamples(simultaneousTimeline, 1.46, 0.25), []);
assert.strictEqual(logic.eventLevel({{level: "alarm", status: "collecting"}}), "alarm");
assert.strictEqual(logic.eventLevel({{level: "risk", status: "ready"}}), "risk");
assert.strictEqual(logic.eventIsAlarm({{level: "risk", status: "ready"}}), false);
assert.strictEqual(logic.isReadyEvent({{status: "collecting", clip_url: "/a.mp4"}}), false);
assert.strictEqual(logic.isReadyEvent({{status: "ready", clip_url: "/a.mp4", overlay_url: "/a.json"}}), true);
assert.strictEqual(logic.isReadyEvent({{status: "ready", clip_url: "/a.mp4"}}), false);
assert.strictEqual(logic.beijingDateKey("not-a-timestamp"), "");
assert.strictEqual(logic.beijingHourKey("not-a-timestamp"), "");
assert.strictEqual(logic.beijingDateKey("2026-07-20T16:30:00Z"), "2026-07-21");
assert.strictEqual(logic.beijingHourKey("2026-07-20T16:30:00Z"), "2026-07-21T00");
assert.strictEqual(logic.formatTimestamp("2026-07-20T16:30:00Z"), "07/21 00:30:00");
const visibleEvents = [{{event_id: "A"}}, {{event_id: "B"}}];
assert.strictEqual(logic.selectVisibleEvent(visibleEvents, "B").event_id, "B");
assert.strictEqual(logic.selectVisibleEvent(visibleEvents, "missing").event_id, "A");
assert.strictEqual(logic.selectVisibleEvent([], "A"), null);

const timedEvents = [
  {{event_id: "old", camera: "camera01", status: "ready", clip_url: "/old.mp4", overlay_url: "/old.json", occurred_at: "2026-07-20T15:30:00Z"}},
  {{event_id: "latest-a", camera: "camera01", status: "ready", clip_url: "/a.mp4", overlay_url: "/a.json", occurred_at: "2026-07-20T16:30:00Z"}},
  {{event_id: "latest-b", camera: "camera02", status: "ready", clip_url: "/b.mp4", overlay_url: "/b.json", occurred_at: "2026-07-20T16:45:00Z"}},
  {{event_id: "invalid", camera: "camera01", status: "ready", clip_url: "/bad.mp4", overlay_url: "/bad.json", occurred_at: "bad"}},
];
assert.deepStrictEqual(logic.buildDateCounts(timedEvents), {{"2026-07-20": 1, "2026-07-21": 2}});
assert.deepStrictEqual(logic.availableHoursForDate(timedEvents, "2026-07-21"), ["2026-07-21T00"]);
assert.deepStrictEqual(logic.resolveTimeSelection(timedEvents, "", ""), {{selectedDate: "2026-07-21", selectedHour: "2026-07-21T00"}});
assert.deepStrictEqual(logic.resolveTimeSelection(timedEvents, "2026-07-20", "missing"), {{selectedDate: "2026-07-20", selectedHour: "2026-07-20T23"}});
assert.deepStrictEqual(
  logic.filterEventsBySelection(timedEvents, "camera01", "2026-07-21", "2026-07-21T00").map((event) => event.event_id),
  ["latest-a"],
);
assert.deepStrictEqual(logic.filterEventsBySelection(timedEvents, "missing", "2026-07-21", "2026-07-21T00"), []);

assert.strictEqual(logic.reviewResult({{}}), "pending");
assert.strictEqual(logic.reviewResult({{review_result: "confirmed"}}), "confirmed");
assert.strictEqual(logic.reviewResult({{review_result: "false_positive"}}), "false_positive");
assert.strictEqual(logic.reviewResult({{review_result: "unexpected"}}), "pending");
assert.strictEqual(logic.reviewLabel("pending"), "待复核");
assert.strictEqual(logic.reviewLabel("confirmed"), "确认报警");
assert.strictEqual(logic.reviewLabel("false_positive"), "已标误报");

const reviewedEvents = logic.updateEventReview([
  {{event_id: "A", level: "alarm", risk_peak: 0.91, review_result: "pending"}},
  {{event_id: "B", level: "risk", risk_peak: 0.42, review_result: "pending"}},
], "A", "confirmed");
assert.deepStrictEqual(reviewedEvents, [
  {{event_id: "A", level: "alarm", risk_peak: 0.91, review_result: "confirmed"}},
  {{event_id: "B", level: "risk", risk_peak: 0.42, review_result: "pending"}},
]);
assert.throws(() => logic.updateEventReview(reviewedEvents, "A", "bad"));

const localReviewOverrides = new Map([["A", "confirmed"]]);
const staleReviewPoll = [
  {{event_id: "A", review_result: "pending"}},
  {{event_id: "B", review_result: "pending"}},
];
assert.deepStrictEqual(logic.applyReviewOverrides(staleReviewPoll, localReviewOverrides), [
  {{event_id: "A", review_result: "confirmed"}},
  {{event_id: "B", review_result: "pending"}},
]);
assert.strictEqual(
  logic.reconcileReviewOverrides(staleReviewPoll, localReviewOverrides).get("A"),
  "confirmed",
);
const acknowledgedReviewPoll = [{{event_id: "A", review_result: "confirmed"}}];
assert.strictEqual(
  logic.reconcileReviewOverrides(acknowledgedReviewPoll, localReviewOverrides).has("A"),
  false,
);

const request = logic.createReviewRequest("event A/1", "false_positive");
assert.strictEqual(request.url, "/api/events/event%20A%2F1/review");
assert.strictEqual(request.options.method, "POST");
assert.deepStrictEqual(request.options.headers, {{"Content-Type": "application/json"}});
assert.deepStrictEqual(JSON.parse(request.options.body), {{result: "false_positive"}});
const classifiedRequest = logic.createReviewRequest("A", "false_positive", "fixed_phone");
assert.deepStrictEqual(
  JSON.parse(classifiedRequest.options.body),
  {{result: "false_positive", reason: "fixed_phone"}},
);
assert.throws(() => logic.createReviewRequest("A", "false_positive", "bad"));
assert.throws(() => logic.createReviewRequest("A", "bad"));

const initialState = logic.createDashboardState("2026-07-20T16:30:00Z");
assert.deepStrictEqual(initialState, {{
  selectedView: "",
  selectedDate: "",
  selectedHour: "",
  calendarMonth: "2026-07",
  page: 1,
  pageSize: 20,
  followLatest: true,
  riskBands: ["high", "medium"],
  showFalsePositives: false,
  selectedId: null,
}});

const liveQuery = new URL(logic.buildEventsQuery({{
  ...initialState,
  selectedView: "camera 1/东",
  selectedDate: "2026-07-20",
  selectedHour: "14",
  calendarMonth: "2026-07",
  page: 3,
}}), "http://dashboard.local");
assert.strictEqual(liveQuery.pathname, "/api/events");
assert.deepStrictEqual(Object.fromEntries(liveQuery.searchParams), {{
  month: "2026-07",
  camera: "camera 1/东",
  page: "3",
  page_size: "20",
  risk: "high,medium",
  include_false_positives: "0",
}});

const historicalQuery = new URL(logic.buildEventsQuery({{
  ...initialState,
  selectedView: "camera02",
  selectedDate: "2026-07-20",
  selectedHour: "03",
  calendarMonth: "2026-08",
  page: 2,
  followLatest: false,
}}), "http://dashboard.local");
assert.deepStrictEqual(Object.fromEntries(historicalQuery.searchParams), {{
  month: "2026-08",
  camera: "camera02",
  page: "2",
  page_size: "20",
  date: "2026-07-20",
  hour: "03",
  risk: "high,medium",
  include_false_positives: "0",
}});

const fixedState = {{
  ...initialState,
  selectedView: "camera01",
  selectedDate: "2026-07-20",
  selectedHour: "14",
  calendarMonth: "2026-07",
  page: 4,
  followLatest: false,
  selectedId: "event-old",
}};
assert.deepStrictEqual(logic.selectView(fixedState, "camera02"), {{
  ...fixedState,
  selectedView: "camera02",
  selectedDate: "",
  selectedHour: "",
  page: 1,
  followLatest: true,
  selectedId: null,
}});
assert.deepStrictEqual(logic.selectView(fixedState, ""), {{
  ...fixedState,
  selectedView: "",
  selectedDate: "",
  selectedHour: "",
  page: 1,
  followLatest: true,
  selectedId: null,
}});
assert.deepStrictEqual(logic.selectDate(fixedState, "2026-07-18"), {{
  ...fixedState,
  selectedDate: "2026-07-18",
  selectedHour: "",
  page: 1,
  followLatest: false,
  selectedId: null,
}});
assert.deepStrictEqual(logic.selectHour(fixedState, "09"), {{
  ...fixedState,
  selectedHour: "09",
  page: 1,
  followLatest: false,
  selectedId: null,
}});
assert.deepStrictEqual(logic.selectPage(fixedState, 2, {{date: "2026-07-19", hour: "23"}}), {{
  ...fixedState,
  selectedDate: "2026-07-19",
  selectedHour: "23",
  page: 2,
  followLatest: false,
  selectedId: null,
}});
assert.deepStrictEqual(logic.selectCalendarMonth(fixedState, "2026-08"), {{
  ...fixedState,
  calendarMonth: "2026-08",
}});
assert.deepStrictEqual(logic.toggleFalsePositiveVisibility(fixedState, true), {{
  ...fixedState,
  showFalsePositives: true,
  page: 1,
  selectedId: null,
}});
const falsePositiveQuery = new URL(logic.buildEventsQuery({{
  ...fixedState,
  showFalsePositives: true,
}}), "http://dashboard.local");
assert.strictEqual(falsePositiveQuery.searchParams.get("include_false_positives"), "1");

const responsePayload = {{
  events: [{{event_id: "B"}}, {{event_id: "A"}}],
  pagination: {{page: 2, page_size: 20, total: 23, total_pages: 2}},
  selection: {{camera: "camera01", month: "2026-07", date: "2026-07-20", hour: "14"}},
  calendar: {{"2026-07-20": 23}},
  hours: {{"2026-07-20T14": 23}},
  summary: {{events: 23, alarms: 9, views: 2, max_risk: 0.97}},
}};
const preserved = logic.applyEventsResponse({{...fixedState, selectedId: "A"}}, responsePayload, true);
assert.deepStrictEqual(preserved.events.map((event) => event.event_id), ["B", "A"]);
assert.strictEqual(preserved.state.selectedId, "A");
assert.deepStrictEqual(preserved.summary, responsePayload.summary);
assert.deepStrictEqual(preserved.pagination, responsePayload.pagination);
assert.deepStrictEqual(preserved.calendar, responsePayload.calendar);
assert.deepStrictEqual(preserved.hours, responsePayload.hours);
const transitioned = logic.applyEventsResponse(fixedState, responsePayload, false);
assert.strictEqual(transitioned.state.selectedId, "B");
assert.strictEqual(transitioned.state.selectedDate, "2026-07-20");
assert.strictEqual(transitioned.state.selectedHour, "14");
assert.strictEqual(transitioned.state.calendarMonth, "2026-07");
assert.strictEqual(logic.pollIntervalForState(initialState), 500);
assert.strictEqual(logic.pollIntervalForState(fixedState), 15000);
assert.strictEqual(logic.LIVE_EVENT_POLL_INTERVAL_MS, 500);
assert.strictEqual(logic.HISTORY_EVENT_POLL_INTERVAL_MS, 15000);
assert.strictEqual(logic.paginationLabel({{page: 2, total_pages: 4, total: 68}}), "第 2/4 页，共 68 条");
assert.strictEqual(logic.paginationLabel({{page: 1, total_pages: 0, total: 0}}), "第 0/0 页，共 0 条");
"""
    completed = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_frontend_independent_poll_schedulers_execute_in_node() -> None:
    app_js = Path(__file__).parents[1] / "static" / "app.js"
    script = f"""
const assert = require("assert");
const logic = require({json.dumps(str(app_js))});

function timerHarness() {{
  let nextId = 1;
  const timers = new Map();
  return {{
    timers,
    setTimer(callback, delay) {{
      const id = nextId++;
      timers.set(id, {{callback, delay}});
      return id;
    }},
    clearTimer(id) {{ timers.delete(id); }},
    takeOnly() {{
      assert.strictEqual(timers.size, 1);
      const [id, timer] = [...timers.entries()][0];
      timers.delete(id);
      return timer;
    }},
  }};
}}

(async () => {{
  const statusTimers = timerHarness();
  const statusLoop = logic.createPollingLoop(
    async () => {{}},
    () => logic.STATUS_POLL_INTERVAL_MS,
    statusTimers,
  );
  assert.strictEqual(statusLoop.refresh(), true);
  await new Promise(setImmediate);
  assert.strictEqual(statusTimers.takeOnly().delay, 500);
  statusLoop.stop();

  const historyTimers = timerHarness();
  const historicalState = {{...logic.createDashboardState("2026-07-20T00:00:00Z"), followLatest: false}};
  const historyLoop = logic.createPollingLoop(
    async () => {{}},
    () => logic.pollIntervalForState(historicalState),
    historyTimers,
  );
  assert.strictEqual(historyLoop.refresh(), true);
  await new Promise(setImmediate);
  assert.strictEqual(historyTimers.takeOnly().delay, 15000);
  historyLoop.stop();

  const liveTimers = timerHarness();
  const liveLoop = logic.createPollingLoop(
    async () => {{}},
    () => logic.pollIntervalForState(logic.createDashboardState("2026-07-20T00:00:00Z")),
    liveTimers,
  );
  assert.strictEqual(liveLoop.refresh(), true);
  await new Promise(setImmediate);
  assert.strictEqual(liveTimers.takeOnly().delay, 500);
  liveLoop.stop();

  const guardTimers = timerHarness();
  const resolvers = [];
  let guardedRuns = 0;
  const guardedLoop = logic.createPollingLoop(
    () => new Promise((resolve) => {{
      guardedRuns += 1;
      resolvers.push(resolve);
    }}),
    () => 500,
    guardTimers,
  );
  assert.strictEqual(guardedLoop.refresh(), true);
  assert.strictEqual(guardedLoop.refresh(), false);
  assert.strictEqual(guardedLoop.isInFlight(), true);
  assert.strictEqual(guardedRuns, 1);
  resolvers.shift()();
  await new Promise(setImmediate);
  const queued = guardTimers.takeOnly();
  assert.strictEqual(queued.delay, 0);
  queued.callback();
  assert.strictEqual(guardedRuns, 2);
  guardedLoop.stop();
  resolvers.shift()();

  const independentTimers = timerHarness();
  let releaseStatus;
  const blockedStatus = logic.createPollingLoop(
    () => new Promise((resolve) => {{ releaseStatus = resolve; }}),
    () => 500,
    independentTimers,
  );
  let eventRuns = 0;
  const independentEvents = logic.createPollingLoop(
    async () => {{ eventRuns += 1; }},
    () => 2000,
    independentTimers,
  );
  blockedStatus.refresh();
  assert.strictEqual(independentEvents.refresh(), true);
  assert.strictEqual(eventRuns, 1);
  blockedStatus.stop();
  independentEvents.stop();
  releaseStatus();
}})().catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""
    completed = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
