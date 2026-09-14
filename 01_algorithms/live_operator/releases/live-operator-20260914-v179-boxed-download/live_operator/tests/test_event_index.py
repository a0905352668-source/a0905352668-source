from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from live_operator import event_index as event_index_module
from live_operator.dashboard import DashboardApp
from live_operator.event_index import EventIndex, EventIndexError, IndexSummary


def test_integrity_check_runs_at_initialization_not_on_every_query(
    tmp_path: Path, monkeypatch
) -> None:
    statements: list[str] = []
    real_connect = event_index_module.sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(event_index_module.sqlite3, "connect", tracking_connect)
    index = EventIndex(tmp_path / "events.sqlite3")
    startup_checks = sum(
        statement.strip().upper().startswith("PRAGMA QUICK_CHECK")
        for statement in statements
    )

    assert startup_checks == 1
    index.refresh([], lambda _candidate: [])
    assert index.records() == []
    assert sum(
        statement.strip().upper().startswith("PRAGMA QUICK_CHECK")
        for statement in statements
    ) == startup_checks


def test_query_detects_and_recovers_database_replaced_with_invalid_bytes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "events.sqlite3"
    index = EventIndex(database)
    database.write_bytes(b"not a sqlite database")

    with pytest.raises(EventIndexError):
        index.records()

    assert list(tmp_path.glob("events.sqlite3.corrupt-*"))
    assert EventIndex(database).records() == []


def test_dashboard_reuses_decoded_rows_until_a_run_signature_changes(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "live_20260818_000000"
    dashboard = run_dir / "dashboard"
    clips = dashboard / "clips"
    clips.mkdir(parents=True)

    def publish(events: list[dict[str, object]]) -> None:
        (dashboard / "events.json").write_text(
            json.dumps(events), encoding="utf-8"
        )
        for event in events:
            event_id = str(event["event_id"])
            (clips / f"{event_id}.mp4").write_bytes(b"video")
            (clips / f"{event_id}.json").write_text(
                json.dumps({"event_id": event_id}), encoding="utf-8"
            )

    first = {
        "event_id": "camera01_person-1_alarm-1",
        "camera": "camera01",
        "status": "ready",
        "occurred_at": "2026-08-18T08:00:00+08:00",
    }
    publish([first])
    app = DashboardApp(run_dir)
    assert app.handle("GET", "/api/events").status == 200
    assert app._event_index is not None

    records_calls = 0
    summarize_calls = 0
    real_records = app._event_index.records
    real_summarize = app._summarize_indexed_record

    def tracking_records():
        nonlocal records_calls
        records_calls += 1
        return real_records()

    def tracking_summarize(record):
        nonlocal summarize_calls
        summarize_calls += 1
        return real_summarize(record)

    monkeypatch.setattr(app._event_index, "records", tracking_records)
    monkeypatch.setattr(app, "_summarize_indexed_record", tracking_summarize)

    assert app.handle("GET", "/api/events", query_string="page=1").status == 200
    assert app.handle("GET", "/api/events", query_string="page=2").status == 200
    assert records_calls == 0
    # Only the single page event is decorated per request; the 11k-row archive
    # summary is not rebuilt.
    assert summarize_calls == 2

    second = {
        "event_id": "camera01_person-2_alarm-2",
        "camera": "camera01",
        "status": "ready",
        "occurred_at": "2026-08-18T08:01:00+08:00",
    }
    publish([first, second])
    response = app.handle("GET", "/api/events")

    assert response.status == 200
    assert records_calls == 1
    assert summarize_calls == 6
    assert len(json.loads(response.body)["events"]) == 2


def test_large_dashboard_serves_stale_cache_while_refresh_runs_in_background(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "live_20260818_000000"
    dashboard = run_dir / "dashboard"
    clips = dashboard / "clips"
    clips.mkdir(parents=True)

    def publish(event_id: str) -> None:
        event = {
            "event_id": event_id,
            "camera": "camera01",
            "status": "ready",
            "occurred_at": "2026-08-18T08:00:00+08:00",
        }
        (dashboard / "events.json").write_text(
            json.dumps([event]), encoding="utf-8"
        )
        (clips / f"{event_id}.mp4").write_bytes(b"video")
        (clips / f"{event_id}.json").write_text(
            json.dumps({"event_id": event_id}), encoding="utf-8"
        )

    publish("camera01_person-1_alarm-1")
    app = DashboardApp(run_dir)
    initial = json.loads(app.handle("GET", "/api/events").body)
    assert initial["events"][0]["event_id"] == "camera01_person-1_alarm-1"
    app._event_index_summary = IndexSummary(0, 1, 0, 1000)

    refresh_started = threading.Event()
    release_refresh = threading.Event()

    def blocking_refresh(*_args):
        refresh_started.set()
        assert release_refresh.wait(2.0)
        return app._indexed_records_cache or []

    monkeypatch.setattr(app, "_refresh_index_cache", blocking_refresh)
    publish("camera01_person-2_alarm-2")

    response = json.loads(app.handle("GET", "/api/events").body)

    assert refresh_started.wait(1.0)
    assert response["events"][0]["event_id"] == "camera01_person-1_alarm-1"
    release_refresh.set()
    assert app._event_refresh_thread is not None
    app._event_refresh_thread.join(timeout=2.0)
