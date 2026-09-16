from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import zipfile
import io

import pytest

from live_operator.capture_dataset import LabeledEvent, ManifestEntry
from live_operator.capture_offline import (
    CaptureReviewClient,
    OfflineEvaluator,
    build_parser,
)
from live_operator.capture_evidence import CAPTURE_EVIDENCE_REVISION
from live_operator.mage_vl_service import CAPTURE_PROMPT_REVISION
from live_operator.vlm_review import response_signature


def _entry(tmp_path: Path, event_id: str = "event-a") -> ManifestEntry:
    clip = tmp_path / f"{event_id}.mp4"
    overlay = tmp_path / f"{event_id}.json"
    review = tmp_path / f"{event_id}-review.json"
    clip.write_bytes(b"clip")
    overlay.write_text(
        json.dumps(
            {
                "event_id": event_id,
                "bbox_timeline": [
                    {
                        "track_id": "person-1",
                        "alarm": True,
                        "screen_id": "screen-1",
                        "bbox": [20, 20, 60, 80],
                        "frame_width": 100,
                        "frame_height": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review.write_text("{}", encoding="utf-8")
    return ManifestEntry(
        event_id=event_id,
        camera="camera08",
        source_run_id="live-a",
        source_id=f"incident-{event_id}",
        clip_path=str(clip),
        overlay_path=str(overlay),
        review_path=str(review),
        clip_size=4,
        clip_sha256="1" * 64,
        overlay_sha256="2" * 64,
        review_sha256="3" * 64,
        first_stage_result="pass",
        prompt_revision="prompt-v1",
        evidence_revision="evidence-v1",
        model_version="model-v1",
    )


def _visibility(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "camera08.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera": "camera08",
                "frame_width": 100,
                "frame_height": 100,
                "screens": [
                    {
                        "screen_id": "screen-1",
                        "screen_polygon": [[5, 5], [15, 5], [15, 15], [5, 15]],
                        "capture_position_zones": [
                            {
                                "zone_id": "possible-1",
                                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                            }
                        ],
                        "blocked_position_zones": [],
                        "unknown_position_zones": [],
                    }
                ],
                "occluders": [],
            }
        ),
        encoding="utf-8",
    )


def _foreground() -> dict:
    return {
        "state": "running",
        "aggregate_fps": 80.0,
        "cameras": [
            {"relay": f"camera{index:02d}", "status": "online", "fps": 10.0}
            for index in range(8)
        ],
        "alerts": [],
    }


def _mage() -> dict:
    return {
        "ready": True,
        "scheduler": {
            "active_kind": None,
            "production_waiting": 0,
            "quiet_remaining_seconds": 0.0,
        },
        "resource_health": {"memory_pressure": False},
    }


class Rig:
    def __init__(self, tmp_path: Path, events: tuple[LabeledEvent, ...] | None = None):
        self.foreground = _foreground()
        self.mage = _mage()
        self.capture_requests = 0
        self.now = 100.0
        self.visibility = tmp_path / "visibility"
        _visibility(self.visibility)
        if events is None:
            events = (LabeledEvent(_entry(tmp_path), "AWAY_FROM_SCREEN", "人工复核"),)
        self.events = events
        self.output = tmp_path / "output"
        self.evaluator = self._new_evaluator()

    def _new_evaluator(self) -> OfflineEvaluator:
        return OfflineEvaluator(
            events=self.events,
            visibility_dir=self.visibility,
            output_dir=self.output,
            manifest_digest="a" * 64,
            baseline_fps=80.0,
            expected_camera_count=8,
            status_fetcher=lambda: self.foreground,
            mage_health_fetcher=lambda: self.mage,
            capture_sender=self._send,
            clock=lambda: self.now,
        )

    def _send(self, entry: ManifestEntry, visibility_path: Path) -> dict:
        self.capture_requests += 1
        assert entry.event_id
        assert visibility_path.name == "camera08.json"
        return {
            "label": "IMPOSSIBLE_AWAY_FROM_SCREEN",
            "cancelled": False,
            "latency_seconds": 1.25,
            "candidate_labels": ["IMPOSSIBLE_AWAY_FROM_SCREEN"],
            "evidence_complete": True,
        }

    def set_breach(self, breach: str) -> None:
        if breach == "production_busy":
            self.mage["scheduler"]["active_kind"] = "production"
        elif breach == "fps_drop":
            self.foreground["aggregate_fps"] = 70.0
        elif breach == "camera_loss":
            self.foreground["cameras"][0]["status"] = "offline"
        elif breach == "vlm_unhealthy":
            self.mage["ready"] = False
        elif breach == "memory_pressure":
            self.mage["resource_health"]["memory_pressure"] = True


@pytest.mark.parametrize(
    "breach",
    ["production_busy", "fps_drop", "camera_loss", "vlm_unhealthy", "memory_pressure"],
)
def test_breach_pauses_before_claim(tmp_path: Path, breach: str) -> None:
    rig = Rig(tmp_path)
    rig.set_breach(breach)

    result = rig.evaluator.run_once()

    assert result.state == "paused"
    assert breach.split("_")[0] in result.reason
    assert rig.capture_requests == 0
    records = [json.loads(line) for line in (rig.output / "results.jsonl").read_text().splitlines()]
    assert records[-1]["record_type"] == "pause"
    assert "label" not in records[-1]


def test_end_to_end_resume_and_report(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    completed = rig.evaluator.run_once()
    resumed = rig._new_evaluator().run_once()
    report = rig._new_evaluator().write_report()

    assert completed.state == "completed"
    assert completed.event_id == "event-a"
    assert resumed.state == "done"
    assert rig.capture_requests == 1
    record = json.loads((rig.output / "results.jsonl").read_text().splitlines()[0])
    assert record["stage_two_label"] == "IMPOSSIBLE_AWAY_FROM_SCREEN"
    assert record["visibility"] == "possible"
    assert record["final_outcome"] == "FILTER_CAPTURE_IMPOSSIBLE"
    assert record["ground_truth"] == "AWAY_FROM_SCREEN"
    metrics = json.loads((rig.output / "metrics.json").read_text())
    assert metrics["aggregate"]["false_positive_reduction"] == 1.0
    assert metrics["aggregate"]["positive_retention"] is None
    assert report == rig.output / "report.html"
    assert report.is_file()
    assert os.stat(rig.output / "results.jsonl").st_mode & 0o777 == 0o600
    assert os.stat(rig.output / "checkpoint.json").st_mode & 0o777 == 0o600


def test_rate_limit_allows_at_most_one_request_per_30_seconds(tmp_path: Path) -> None:
    events = (
        LabeledEvent(_entry(tmp_path, "event-a"), "AWAY_FROM_SCREEN", "a"),
        LabeledEvent(_entry(tmp_path, "event-b"), "VIEWING_PHONE", "b"),
    )
    rig = Rig(tmp_path, events)

    assert rig.evaluator.run_once().state == "completed"
    rig.now += 29.0
    limited = rig.evaluator.run_once()
    rig.now += 1.0
    second = rig.evaluator.run_once()

    assert limited.state == "rate_limited"
    assert second.state == "completed"
    assert rig.capture_requests == 2


def test_cancelled_response_pauses_without_fabricating_label(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.evaluator.capture_sender = lambda _entry, _path: {
        "label": "UNCERTAIN",
        "cancelled": True,
        "latency_seconds": 0.2,
    }

    result = rig.evaluator.run_once()

    assert result.state == "paused"
    assert result.reason == "capture_cancelled"
    record = json.loads((rig.output / "results.jsonl").read_text().splitlines()[-1])
    assert record["record_type"] == "pause"
    assert "stage_two_label" not in record


def test_frontend_alert_pauses_before_claim(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.foreground["alerts"] = [{"code": "camera_low_fps"}]

    result = rig.evaluator.run_once()

    assert result.state == "paused"
    assert result.reason == "foreground_alerts"
    assert rig.capture_requests == 0


def test_capture_client_signs_stored_archive_and_validates_response(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_bytes(b"z" * 32)
    secret.chmod(0o600)
    entry = _entry(tmp_path)
    visibility = tmp_path / "visibility.json"
    visibility.write_text(
        json.dumps({"schema_version": 1, "camera": "camera08"}), encoding="utf-8"
    )
    observed = {}

    class Response:
        status = 200

        def __init__(self, body: bytes, headers: dict):
            self._body = body
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return self._body

    def opener(request, **_kwargs):
        observed["url"] = request.full_url
        observed["signature"] = request.headers["X-jiankong-signature"]
        archive_body = request.data
        with zipfile.ZipFile(io.BytesIO(archive_body)) as archive:
            observed["members"] = set(archive.namelist())
            metadata = json.loads(archive.read("request.json"))
        payload = {
            "schema_version": 1,
            "event_id": entry.event_id,
            "request_id": metadata["request_id"],
            "label": "CAPTURE_POSSIBLE",
            "cancelled": False,
            "model_version": "model-v2",
            "prompt_revision": CAPTURE_PROMPT_REVISION,
            "evidence_revision": CAPTURE_EVIDENCE_REVISION,
            "candidate_count": 1,
            "candidate_labels": ["CAPTURE_POSSIBLE"],
            "evidence_complete": True,
            "reviewed_at": "2026-09-16T00:00:00Z",
            "latency_seconds": 1.0,
        }
        response_body = json.dumps(payload, separators=(",", ":")).encode()
        return Response(
            response_body,
            {
                "X-Jiankong-Response-Signature": response_signature(
                    b"z" * 32, observed["signature"], response_body
                )
            },
        )

    client = CaptureReviewClient(
        endpoint="http://127.0.0.1:8879/v1/capture-review",
        shared_secret_file=secret,
        expected_model_version="model-v2",
        opener=opener,
        clock=lambda: 1_700_000_000,
    )

    response = client.send(entry, visibility)

    assert urlsplit(observed["url"]).path == "/v1/capture-review"
    assert observed["members"] == {
        "request.json",
        "clip.mp4",
        "overlay.json",
        "visibility.json",
    }
    assert len(observed["signature"]) == 64
    assert response["label"] == "CAPTURE_POSSIBLE"


def test_cli_exposes_all_offline_commands() -> None:
    parser = build_parser()

    for command in ("build-manifest", "validate-labels", "dry-run", "run", "report"):
        with pytest.raises(SystemExit) as result:
            parser.parse_args([command, "--help"])
        assert result.value.code == 0
