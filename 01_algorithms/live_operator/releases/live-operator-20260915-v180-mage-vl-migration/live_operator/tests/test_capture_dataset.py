from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from live_operator.capture_dataset import (
    build_manifest,
    split_by_source,
    validate_labels,
)


def _write_reviewed_event(
    review_pool: Path,
    runs_root: Path,
    *,
    event_id: str,
    camera: str = "camera08",
    source_run_id: str = "live_a",
    source_id: str | None = None,
    reviewed_at: str = "2026-09-15T08:00:00+00:00",
    first_stage_result: str | None = "pass",
) -> Path:
    event_dir = review_pool / "confirmed" / "2026-09-15" / event_id
    event_dir.mkdir(parents=True)
    (event_dir / "review.json").write_text(
        json.dumps(
            {
                "event_id": event_id,
                "result": "confirmed",
                "reason": None,
                "category": "phone",
                "reviewed_at": reviewed_at,
                "camera": camera,
                "occurred_at": "2026-09-15T07:59:55+00:00",
                "source_run_id": source_run_id,
            }
        ),
        encoding="utf-8",
    )
    (event_dir / "event.mp4").write_bytes(f"clip:{event_id}".encode())
    overlay = {"event_id": event_id, "bbox_timeline": []}
    if source_id is not None:
        overlay["source_id"] = source_id
    (event_dir / "overlay.json").write_text(json.dumps(overlay), encoding="utf-8")

    events_path = runs_root / source_run_id / "dashboard" / "events.json"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events = json.loads(events_path.read_text()) if events_path.exists() else []
    source_event = {
        "event_id": event_id,
        "camera": camera,
        "vlm_filter_expected_prompt_revision": "prompt-v1",
        "vlm_filter_evidence_revision": "evidence-v1",
        "vlm_filter_model_version": "model-v1",
    }
    if first_stage_result is not None:
        source_event["vlm_filter_result"] = first_stage_result
    events.append(source_event)
    events_path.write_text(json.dumps(events), encoding="utf-8")
    return event_dir


def test_manifest_is_deterministic_and_written_privately(tmp_path: Path) -> None:
    pool = tmp_path / "review_pool"
    runs = tmp_path / "runs"
    event_dir = _write_reviewed_event(pool, runs, event_id="event-a")

    first = build_manifest(pool, runs, tmp_path / "a.json", frozenset({"camera08"}), 10)
    second = build_manifest(pool, runs, tmp_path / "b.json", frozenset({"camera08"}), 10)

    assert first == second
    assert first.digest
    assert len(first.entries) == 1
    entry = first.entries[0]
    assert entry.event_id == "event-a"
    assert entry.first_stage_result == "pass"
    assert entry.prompt_revision == "prompt-v1"
    assert entry.evidence_revision == "evidence-v1"
    assert entry.model_version == "model-v1"
    assert entry.clip_path == str((event_dir / "event.mp4").resolve())
    assert entry.clip_size == len(b"clip:event-a")
    assert len(entry.clip_sha256) == 64
    assert len(entry.overlay_sha256) == 64
    assert os.stat(tmp_path / "a.json").st_mode & 0o777 == 0o600


def test_missing_source_event_is_explicitly_uncertain(tmp_path: Path) -> None:
    pool = tmp_path / "review_pool"
    runs = tmp_path / "runs"
    _write_reviewed_event(pool, runs, event_id="event-a")
    (runs / "live_a" / "dashboard" / "events.json").write_text("[]")

    manifest = build_manifest(
        pool, runs, tmp_path / "manifest.json", frozenset({"camera08"}), 10
    )

    entry = manifest.entries[0]
    assert entry.first_stage_result == "uncertain"
    assert entry.prompt_revision is None
    assert entry.evidence_revision is None
    assert entry.model_version is None


def test_camera_filter_and_limit_are_deterministic(tmp_path: Path) -> None:
    pool = tmp_path / "review_pool"
    runs = tmp_path / "runs"
    _write_reviewed_event(pool, runs, event_id="later", reviewed_at="2026-09-15T09:00:00Z")
    _write_reviewed_event(pool, runs, event_id="earlier", reviewed_at="2026-09-15T08:00:00Z")
    _write_reviewed_event(pool, runs, event_id="other", camera="camera09")

    manifest = build_manifest(
        pool, runs, tmp_path / "manifest.json", frozenset({"camera08"}), 1
    )

    assert [entry.event_id for entry in manifest.entries] == ["earlier"]


def test_symlinked_reviewed_event_is_rejected(tmp_path: Path) -> None:
    pool = tmp_path / "review_pool"
    runs = tmp_path / "runs"
    event_dir = _write_reviewed_event(pool, runs, event_id="event-a")
    link = pool / "confirmed" / "2026-09-15" / "event-link"
    link.symlink_to(event_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        build_manifest(pool, runs, tmp_path / "manifest.json", frozenset({"camera08"}), 10)


def test_labels_require_digest_known_event_valid_label_and_reason(tmp_path: Path) -> None:
    pool = tmp_path / "review_pool"
    runs = tmp_path / "runs"
    _write_reviewed_event(pool, runs, event_id="event-a", source_id="incident-1")
    manifest = build_manifest(
        pool, runs, tmp_path / "manifest.json", frozenset({"camera08"}), 10
    )
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_digest": manifest.digest,
                "labels": [
                    {
                        "event_id": "event-a",
                        "label": "CAPTURE_POSSIBLE",
                        "reason": "手机镜头朝向受保护屏幕",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    labeled = validate_labels(manifest, labels_path)

    assert labeled[0].entry.event_id == "event-a"
    assert labeled[0].ground_truth == "CAPTURE_POSSIBLE"
    assert labeled[0].reason == "手机镜头朝向受保护屏幕"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda body: body.update(manifest_digest="0" * 64), "digest"),
        (lambda body: body["labels"][0].update(event_id="missing"), "unknown event"),
        (lambda body: body["labels"][0].update(label="MAYBE"), "label"),
        (lambda body: body["labels"][0].update(reason="  "), "reason"),
    ],
)
def test_invalid_labels_are_rejected(tmp_path: Path, mutation, message: str) -> None:
    pool = tmp_path / "review_pool"
    runs = tmp_path / "runs"
    _write_reviewed_event(pool, runs, event_id="event-a")
    manifest = build_manifest(
        pool, runs, tmp_path / "manifest.json", frozenset({"camera08"}), 10
    )
    body = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "labels": [
            {"event_id": "event-a", "label": "NOT_PHONE", "reason": "人工复核"}
        ],
    }
    mutation(body)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_labels(manifest, labels_path)


def test_split_is_deterministic_and_keeps_sources_together(tmp_path: Path) -> None:
    pool = tmp_path / "review_pool"
    runs = tmp_path / "runs"
    _write_reviewed_event(pool, runs, event_id="event-a", source_id="incident-1")
    _write_reviewed_event(pool, runs, event_id="event-b", source_id="incident-1")
    _write_reviewed_event(pool, runs, event_id="event-c", source_id="incident-2")
    manifest = build_manifest(
        pool, runs, tmp_path / "manifest.json", frozenset({"camera08"}), 10
    )
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_digest": manifest.digest,
                "labels": [
                    {"event_id": entry.event_id, "label": "NOT_PHONE", "reason": "人工复核"}
                    for entry in manifest.entries
                ],
            }
        ),
        encoding="utf-8",
    )
    events = validate_labels(manifest, labels_path)

    first = split_by_source(events, seed="capture-v1")
    second = split_by_source(events, seed="capture-v1")

    assert first == second
    tuning_sources = {event.entry.source_id for event in first.tuning}
    evaluation_sources = {event.entry.source_id for event in first.evaluation}
    assert tuning_sources.isdisjoint(evaluation_sources)
    assert sorted(event.entry.event_id for event in first.tuning + first.evaluation) == [
        "event-a",
        "event-b",
        "event-c",
    ]
