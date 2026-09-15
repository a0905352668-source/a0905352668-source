from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone

import pytest

from live_operator.dashboard import DashboardApp
from live_operator.vlm_review import VLM_EVIDENCE_REVISION
from live_operator.vlm_state import VLMReviewStateStore
from live_operator.vlm_state import VLMStateError


REQUEST_ID = "a" * 64
MODEL = "mage-vl-awq-test"
PROMPT = "prompt-v1"
EVIDENCE = "evidence-v1"


def _pending(store: VLMReviewStateStore, event: dict) -> dict:
    return store.set_vlm_filter_result(
        event,
        "pending",
        attempts=1,
        expected_attempts=0,
        request_id=REQUEST_ID,
        evidence_revision=EVIDENCE,
        prompt_revision=PROMPT,
        model_version=MODEL,
    )


def test_compact_state_round_trip_and_terminal_transition(tmp_path) -> None:
    state_path = tmp_path / "vlm_filter_states.json"
    store = VLMReviewStateStore(
        state_path,
        clock=lambda: datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc),
    )
    base = {"event_id": "camera01-person01-event01", "status": "ready"}

    pending = _pending(store, base)
    completed = store.set_vlm_filter_result(
        pending,
        "pass",
        attempts=1,
        expected_attempts=1,
        request_id=REQUEST_ID,
        evidence_revision=EVIDENCE,
        prompt_revision=PROMPT,
        model_version=MODEL,
        label="KEEP_NON_CALL_PHONE_USE",
        reviewed_at="2026-08-24T08:00:08+00:00",
        latency_seconds=8.0,
    )

    assert completed["vlm_filter_result"] == "pass"
    merged = VLMReviewStateStore(state_path).merge_events([base])[0]
    assert merged["vlm_filter_result"] == "pass"
    assert merged["vlm_filter_latency_seconds"] == 8.0
    assert state_path.stat().st_size < 4096
    assert json.loads(state_path.read_text())[base["event_id"]][
        "vlm_filter_result"
    ] == "pass"


def test_sidecar_overrides_stale_legacy_vlm_fields(tmp_path) -> None:
    store = VLMReviewStateStore(tmp_path / "vlm_filter_states.json")
    base = {
        "event_id": "camera01-person01-event02",
        "status": "ready",
        "vlm_filter_result": "filter",
        "vlm_filter_attempts": 1,
        "vlm_filter_evidence_revision": "old-evidence",
        "vlm_filter_expected_prompt_revision": "old-prompt",
        "vlm_filter_expected_model_version": "old-model",
    }
    revised = store.set_vlm_filter_result(
        base,
        "pending",
        attempts=1,
        expected_attempts=1,
        request_id=REQUEST_ID,
        evidence_revision=EVIDENCE,
        prompt_revision=PROMPT,
        model_version=MODEL,
    )

    assert revised["vlm_filter_result"] == "pending"
    merged = store.merge_events([base])[0]
    assert merged["vlm_filter_evidence_revision"] == EVIDENCE
    assert merged["vlm_filter_expected_prompt_revision"] == PROMPT


def test_stale_completion_does_not_replace_pending_state(tmp_path) -> None:
    store = VLMReviewStateStore(tmp_path / "vlm_filter_states.json")
    base = {"event_id": "camera01-person01-event03", "status": "ready"}
    _pending(store, base)

    with pytest.raises(ValueError, match="stale VLM filter completion"):
        store.set_vlm_filter_result(
            base,
            "pass",
            attempts=1,
            expected_attempts=1,
            request_id="b" * 64,
            evidence_revision=EVIDENCE,
            prompt_revision=PROMPT,
            model_version=MODEL,
            label="KEEP_NON_CALL_PHONE_USE",
            reviewed_at="2026-08-24T08:00:08+00:00",
        )

    assert store.merge_events([base])[0]["vlm_filter_result"] == "pending"


def test_dashboard_merges_compact_vlm_state_without_rewriting_events(tmp_path) -> None:
    run = tmp_path / "live_20260824_080000"
    dashboard = run / "dashboard"
    clips = dashboard / "clips"
    clips.mkdir(parents=True)
    event_id = "camera01-person01-event04"
    base = {
        "event_id": event_id,
        "camera": "camera01",
        "status": "ready",
        "level": "alarm",
        "occurred_at": "2026-08-24T08:00:00+08:00",
        "risk_peak": 0.9,
    }
    events_path = dashboard / "events.json"
    events_path.write_text(json.dumps([base]), encoding="utf-8")
    original = events_path.read_bytes()
    (clips / f"{event_id}.mp4").write_bytes(b"video")
    (clips / f"{event_id}.json").write_text(
        json.dumps({"event_id": event_id}), encoding="utf-8"
    )
    store = VLMReviewStateStore(dashboard / "vlm_filter_states.json")
    pending = store.set_vlm_filter_result(
        base,
        "pending",
        attempts=1,
        expected_attempts=0,
        request_id=REQUEST_ID,
        evidence_revision=VLM_EVIDENCE_REVISION,
        prompt_revision=PROMPT,
        model_version=MODEL,
    )
    store.set_vlm_filter_result(
        pending,
        "pass",
        attempts=1,
        expected_attempts=1,
        request_id=REQUEST_ID,
        evidence_revision=VLM_EVIDENCE_REVISION,
        prompt_revision=PROMPT,
        model_version=MODEL,
        label="KEEP_NON_CALL_PHONE_USE",
        reviewed_at="2026-08-24T08:00:08+00:00",
        latency_seconds=8.0,
    )

    records = DashboardApp(run)._scan_run_records(run)

    assert len(records) == 1
    assert records[0].payload["vlm_filter_result"] == "pass"
    assert events_path.read_bytes() == original


def test_dashboard_sidecar_update_does_not_invalidate_event_index(tmp_path) -> None:
    run = tmp_path / "live_20260824_081000"
    dashboard = run / "dashboard"
    clips = dashboard / "clips"
    clips.mkdir(parents=True)
    event_id = "camera01-person01-event05"
    base = {
        "event_id": event_id,
        "camera": "camera01",
        "status": "ready",
        "level": "alarm",
        "occurred_at": "2026-08-24T08:10:00+08:00",
        "risk_peak": 0.9,
    }
    (dashboard / "events.json").write_text(json.dumps([base]), encoding="utf-8")
    (clips / f"{event_id}.mp4").write_bytes(b"video")
    (clips / f"{event_id}.json").write_text(
        json.dumps({"event_id": event_id}), encoding="utf-8"
    )
    app = DashboardApp(run)
    original_signature = app._event_file_signature(dashboard)
    first = json.loads(
        app.handle(
            "GET",
            "/api/events",
            query_string="page=1&page_size=20&exclude_vlm_filtered=0",
        ).body
    )
    indexed_records = app._indexed_records_cache
    assert first["events"][0].get("vlm_filter_result") is None

    store = VLMReviewStateStore(dashboard / "vlm_filter_states.json")
    pending = store.set_vlm_filter_result(
        base,
        "pending",
        attempts=1,
        expected_attempts=0,
        request_id=REQUEST_ID,
        evidence_revision=VLM_EVIDENCE_REVISION,
        prompt_revision=PROMPT,
        model_version=MODEL,
    )
    store.set_vlm_filter_result(
        pending,
        "pass",
        attempts=1,
        expected_attempts=1,
        request_id=REQUEST_ID,
        evidence_revision=VLM_EVIDENCE_REVISION,
        prompt_revision=PROMPT,
        model_version=MODEL,
        label="KEEP_NON_CALL_PHONE_USE",
        reviewed_at="2026-08-24T08:10:08+00:00",
    )

    second = json.loads(
        app.handle(
            "GET",
            "/api/events",
            query_string="page=1&page_size=20&exclude_vlm_filtered=0",
        ).body
    )

    assert app._event_file_signature(dashboard) == original_signature
    assert app._indexed_records_cache is indexed_records
    assert second["events"][0]["vlm_filter_result"] == "pass"


def test_state_store_refuses_a_symlinked_lock_without_touching_its_target(tmp_path) -> None:
    """Following a lock symlink would let an operator chmod or lock an arbitrary file."""

    state_path = tmp_path / "vlm_filter_states.json"
    target = tmp_path / "unrelated-target"
    target.write_bytes(b"do not touch")
    os.chmod(target, 0o640)
    target_before = target.stat()
    lock_path = state_path.with_suffix(".json.lock")
    lock_path.symlink_to(target)
    store = VLMReviewStateStore(state_path)

    with pytest.raises(VLMStateError, match="lock"):
        store.set_vlm_filter_result(
            {"event_id": "camera01-person01-event-lock", "status": "ready"},
            "pending",
            attempts=1,
            expected_attempts=0,
            request_id=REQUEST_ID,
            evidence_revision=EVIDENCE,
            prompt_revision=PROMPT,
            model_version=MODEL,
        )

    target_after = target.stat()
    assert target.read_bytes() == b"do not touch"
    assert stat.S_IMODE(target_after.st_mode) == stat.S_IMODE(target_before.st_mode)
    assert target_after.st_mtime_ns == target_before.st_mtime_ns
