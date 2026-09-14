import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest

from live_operator.dashboard import DashboardApp
from live_operator import false_positive_dataset
from live_operator import review_artifacts


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_reviewable_event(tmp_path: Path, *, status: str = "ready") -> None:
    dashboard = tmp_path / "dashboard"
    write_json(
        dashboard / "events.json",
        [
            {
                "event_id": "event-1",
                "status": status,
                "occurred_at": "2026-07-20T14:00:00+08:00",
            }
        ],
    )
    clips = dashboard / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    (clips / "event-1.mp4").write_bytes(b"clip")
    write_json(clips / "event-1.json", {"event_id": "event-1"})


def test_event_summary_defaults_review_result_to_pending(tmp_path: Path) -> None:
    write_reviewable_event(tmp_path)

    response = DashboardApp(tmp_path).handle("GET", "/api/events")

    assert response.status == 200
    assert json.loads(response.body)["events"][0]["review_result"] == "pending"


def test_review_transitions_persist_and_are_returned_in_event_summaries(
    tmp_path: Path,
) -> None:
    write_reviewable_event(tmp_path)
    app = DashboardApp(tmp_path)

    for result in ("confirmed", "false_positive", "pending"):
        response = app.handle(
            "POST",
            "/api/events/event-1/review",
            body=json.dumps({"result": result}).encode("utf-8"),
        )

        assert response.status == 200
        response_payload = json.loads(response.body)
        assert {
            key: response_payload[key]
            for key in ("event_id", "review_result")
        } == {
            "event_id": "event-1",
            "review_result": result,
        }
        if result == "false_positive":
            assert response_payload["archived"] is True
            assert "archive_clip" not in response_payload
        else:
            assert "archived" not in response_payload
        persisted = json.loads(
            (tmp_path / "dashboard" / "event_reviews.json").read_text(
                encoding="utf-8"
            )
        )
        assert persisted["event-1"]["result"] == result
        datetime.fromisoformat(persisted["event-1"]["updated_at"])
        summary = json.loads(
            DashboardApp(tmp_path).handle(
                "GET",
                "/api/events",
                query_string="date=2026-07-20&hour=14&page=1&page_size=1",
            ).body
        )["events"][0]
        assert summary["review_result"] == result


def test_false_positive_review_archives_only_the_unboxed_clip(tmp_path: Path) -> None:
    write_reviewable_event(tmp_path)

    response = DashboardApp(tmp_path).handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"false_positive"}',
    )

    assert response.status == 200
    archive = tmp_path / "reviewed_false_positives" / "unknown" / "2026-07-20"
    assert (archive / "event-1.mp4").read_bytes() == b"clip"
    assert not list(archive.glob("*.json"))
    assert json.loads(response.body)["archived"] is True


def test_false_positive_review_archives_preannotated_person_roi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_id = "event-1"
    dashboard = tmp_path / "dashboard"
    write_json(
        dashboard / "events.json",
        [
            {
                "event_id": event_id,
                "camera": "camera04",
                "person_id": "17",
                "status": "ready",
                "occurred_at": "2026-07-20T14:00:05+08:00",
                "bbox_timeline": [
                    {
                        "time_sec": 120.0,
                        "frame_index": 960,
                        "bbox": [100.0, 50.0, 300.0, 350.0],
                        "track_id": "17",
                        "risk_score": 0.91,
                        "alarm": True,
                    }
                ],
            }
        ],
    )
    clips = dashboard / "clips"
    clips.mkdir(parents=True)
    (clips / f"{event_id}.mp4").write_bytes(b"raw-clip")
    write_json(
        clips / f"{event_id}.json",
        {
            "window_start": "2026-07-20T06:00:00+00:00",
            "window_end": "2026-07-20T06:00:10+00:00",
            "overlay": {
                "bbox_timeline": [
                    {
                        "time_sec": 5.0,
                        "frame_index": 960,
                        "bbox": [100.0, 50.0, 300.0, 350.0],
                        "track_id": "17",
                        "risk_score": 0.91,
                        "alarm": True,
                    }
                ]
            },
        },
    )
    inference = tmp_path / "inference"
    inference.mkdir()
    (inference / "frame_events.jsonl").write_text(
        json.dumps(
            {
                "camera": "camera04",
                "captured_at": "2026-07-20T06:00:05+00:00",
                "frame_index": 960,
                "width": 640,
                "height": 480,
                "persons": [
                    {"track_id": 17, "roi": [80.0, 40.0, 320.0, 380.0]}
                ],
                "phones": [
                    {
                        "track_id": 17,
                        "box": [180.0, 160.0, 220.0, 240.0],
                        "phone_score": 0.88,
                    },
                    {
                        "track_id": 17,
                        "box": [181.0, 161.0, 219.0, 239.0],
                        "phone_score": 0.72,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    extracted: dict[str, object] = {}

    def fake_extract(_ffmpeg, _clip, destination, *, clip_time_sec, roi):
        extracted["clip_time_sec"] = clip_time_sec
        extracted["roi"] = roi
        destination.write_bytes(b"jpeg")

    monkeypatch.setattr(false_positive_dataset, "_extract_crop", fake_extract)
    monkeypatch.setattr(false_positive_dataset.shutil, "which", lambda _name: "ffmpeg")
    response = DashboardApp(tmp_path).handle(
        "POST",
        f"/api/events/{event_id}/review",
        body=b'{"result":"false_positive"}',
    )

    assert response.status == 200
    assert json.loads(response.body)["person_roi_archived"] is True
    roi_dir = tmp_path.parent / "reviewed_false_positive_labelme"
    assert (roi_dir / f"{event_id}__roi-01.jpg").read_bytes() == b"jpeg"
    label = json.loads((roi_dir / f"{event_id}__roi-01.json").read_text(encoding="utf-8"))
    assert label["imagePath"] == f"{event_id}__roi-01.jpg"
    assert label["imageWidth"] == 240
    assert label["imageHeight"] == 340
    assert label["flags"] == {"reviewed_false_positive": True, "preannotated": True}
    assert len(label["shapes"]) == 1
    assert label["shapes"][0]["label"] == "phone"
    assert label["shapes"][0]["points"] == [
        [100.0, 120.0],
        [140.0, 120.0],
        [140.0, 200.0],
        [100.0, 200.0],
    ]
    assert extracted == {"clip_time_sec": 5.0, "roi": (80, 40, 320, 380)}


def test_false_positive_review_archive_is_idempotent_and_removed_when_reopened(
    tmp_path: Path,
) -> None:
    write_reviewable_event(tmp_path)
    app = DashboardApp(tmp_path)

    first = app.handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"false_positive"}',
    )
    archive_path = (
        tmp_path / "reviewed_false_positives" / "unknown" / "2026-07-20" / "event-1.mp4"
    )
    archive_path.write_bytes(b"already archived")

    second = app.handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"false_positive"}',
    )
    assert json.loads(second.body)["archived"] is True
    assert archive_path.read_bytes() == b"already archived"

    reopened = app.handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"pending"}',
    )
    assert reopened.status == 200
    assert not archive_path.exists()


@pytest.mark.parametrize(
    "path",
    (
        "/api/events/missing/review",
        "/api/events/%2E%2E%2Fevent/review",
    ),
)
def test_review_rejects_unknown_or_unsafe_event_ids(
    tmp_path: Path, path: str
) -> None:
    write_json(
        tmp_path / "dashboard" / "events.json",
        [
            {"event_id": "event-1", "status": "ready"},
            {"event_id": "../event", "status": "ready"},
        ],
    )

    response = DashboardApp(tmp_path).handle(
        "POST", path, body=b'{"result":"confirmed"}'
    )

    assert response.status == 404
    assert not (tmp_path / "dashboard" / "event_reviews.json").exists()


@pytest.mark.parametrize("status", ("collecting", "failed"))
def test_review_rejects_events_that_are_not_ready(
    tmp_path: Path, status: str
) -> None:
    write_reviewable_event(tmp_path, status=status)

    response = DashboardApp(tmp_path).handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"confirmed"}',
    )

    assert response.status == 409
    assert not (tmp_path / "dashboard" / "event_reviews.json").exists()


@pytest.mark.parametrize("missing_name", ("event-1.mp4", "event-1.json"))
def test_review_rejects_ready_events_with_missing_media(
    tmp_path: Path, missing_name: str
) -> None:
    write_reviewable_event(tmp_path)
    (tmp_path / "dashboard" / "clips" / missing_name).unlink()

    response = DashboardApp(tmp_path).handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"confirmed"}',
    )

    assert response.status == 409
    assert not (tmp_path / "dashboard" / "event_reviews.json").exists()


@pytest.mark.parametrize(
    "body",
    (
        b'{"result":"approved"}',
        b'{"result":null}',
        b'{"result":[]}',
        b"[]",
    ),
)
def test_review_rejects_invalid_results(tmp_path: Path, body: bytes) -> None:
    write_reviewable_event(tmp_path)

    response = DashboardApp(tmp_path).handle(
        "POST", "/api/events/event-1/review", body=body
    )

    assert response.status == 400
    assert not (tmp_path / "dashboard" / "event_reviews.json").exists()


@pytest.mark.parametrize("body", (b"", b"{", b"\xff"))
def test_review_rejects_malformed_json(tmp_path: Path, body: bytes) -> None:
    write_reviewable_event(tmp_path)

    response = DashboardApp(tmp_path).handle(
        "POST", "/api/events/event-1/review", body=body
    )

    assert response.status == 400
    assert not (tmp_path / "dashboard" / "event_reviews.json").exists()


def test_wsgi_review_request_reads_json_body(tmp_path: Path) -> None:
    write_reviewable_event(tmp_path)
    body = b'{"result":"confirmed"}'
    statuses: list[str] = []

    response_body = b"".join(
        DashboardApp(tmp_path)(
            {
                "REQUEST_METHOD": "POST",
                "PATH_INFO": "/api/events/event-1/review",
                "CONTENT_LENGTH": str(len(body)),
                "wsgi.input": BytesIO(body),
            },
            lambda status, _headers: statuses.append(status),
        )
    )

    assert statuses == ["200 OK"]
    assert json.loads(response_body)["review_result"] == "confirmed"


def test_false_positive_reason_is_persisted_and_returned(tmp_path: Path) -> None:
    write_reviewable_event(tmp_path)

    response = DashboardApp(tmp_path).handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"false_positive","reason":"model_misdetect"}',
    )

    assert response.status == 200
    assert json.loads(response.body)["review_reason"] == "model_misdetect"
    reviews = json.loads(
        (tmp_path / "dashboard" / "event_reviews.json").read_text(encoding="utf-8")
    )
    assert reviews["event-1"]["reason"] == "model_misdetect"
    summary = json.loads(DashboardApp(tmp_path).handle("GET", "/api/events").body)[
        "events"
    ][0]
    assert summary["review_reason"] == "model_misdetect"


@pytest.mark.parametrize("reason", ("fixed", "", 1, [], None))
def test_false_positive_rejects_invalid_explicit_reason(
    tmp_path: Path, reason: object
) -> None:
    write_reviewable_event(tmp_path)

    response = DashboardApp(tmp_path).handle(
        "POST",
        "/api/events/event-1/review",
        body=json.dumps({"result": "false_positive", "reason": reason}).encode(),
    )

    if reason is None:
        assert response.status == 200
    else:
        assert response.status == 400


def test_reviewed_event_pool_archives_confirmed_evidence(tmp_path: Path) -> None:
    write_reviewable_event(tmp_path)

    response = DashboardApp(tmp_path).handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"confirmed"}',
    )

    assert response.status == 200
    archived = (
        tmp_path.parent
        / "reviewed_event_pool"
        / "confirmed"
        / "2026-07-20"
        / "event-1"
    )
    assert (archived / "event.mp4").read_bytes() == b"clip"
    assert json.loads((archived / "overlay.json").read_text())["event_id"] == "event-1"
    assert json.loads((archived / "review.json").read_text())["result"] == "confirmed"


def test_fixed_phone_review_creates_three_templates_and_supports_management(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_reviewable_event(tmp_path)
    overlay_path = tmp_path / "dashboard" / "clips" / "event-1.json"
    write_json(
        overlay_path,
        {
            "overlay": {
                "bbox_timeline": [
                    {
                        "time_sec": float(index),
                        "frame_index": index,
                        "frame_width": 640,
                        "frame_height": 480,
                        "bbox": [50, 50, 300, 400],
                        "phone_boxes": [
                            {
                                "box": [180 + index, 200, 200 + index, 240],
                                "phone_score": 0.9,
                            }
                        ],
                    }
                    for index in (1, 2, 3)
                ]
            }
        },
    )

    def fake_extract(_ffmpeg, _clip, destination, *, clip_time_sec, roi):
        destination.write_bytes(f"{clip_time_sec}:{roi}".encode())

    monkeypatch.setattr(false_positive_dataset, "_extract_crop", fake_extract)
    monkeypatch.setattr(false_positive_dataset.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(review_artifacts.shutil, "which", lambda _name: "ffmpeg")

    response = DashboardApp(tmp_path).handle(
        "POST",
        "/api/events/event-1/review",
        body=b'{"result":"false_positive","reason":"fixed_phone"}',
    )

    assert response.status == 200
    assert json.loads(response.body)["fixed_template_created"] is True
    listing = json.loads(
        DashboardApp(tmp_path).handle("GET", "/api/fixed-objects").body
    )["templates"]
    template = next(item for item in listing if item["template_id"] == "event-1")
    assert template["sample_count"] == 3
    assert template["thumbnail_url"] == "/fixed-objects/event-1/sample-01.jpg"
    image = DashboardApp(tmp_path).handle(
        "GET", "/fixed-objects/event-1/sample-01.jpg"
    )
    assert image.status == 200

    deleted = DashboardApp(tmp_path).handle(
        "DELETE", "/api/fixed-objects/event-1"
    )
    assert deleted.status == 200
    assert json.loads(deleted.body)["deleted"] is True
