from __future__ import annotations

import hashlib
import http.client
import io
import json
from pathlib import Path
import threading
import zipfile

import pytest

from live_operator.capture_evidence import CAPTURE_EVIDENCE_REVISION
from live_operator.mage_vl_service import (
    CAPTURE_PROMPT,
    CAPTURE_PROMPT_REVISION,
    EARLY_RESCUE_PROMPT,
    FOCUS_PROMPT,
    NATIVE_PROMPT,
    PROMPT_REVISION,
    BoundedReviewHTTPServer,
    MageVLReviewer,
    ReviewApplication,
    capture_request_id,
    make_handler,
    select_candidate_sequences,
    _parser,
    _generation_cancel_mask,
)
from live_operator.vlm_review import (
    VLM_EVIDENCE_REVISION,
    evidence_request_id,
    request_signature,
)


SECRET = b"s" * 32
MODEL_VERSION = "fake-model-v1"


class FakeReviewer:
    model_version = MODEL_VERSION
    model_fingerprint = hashlib.sha256(b"fake-model").hexdigest()

    def __init__(self) -> None:
        self.phone_calls = 0
        self.capture_calls = 0
        self.cancelled_calls = 0

    def review(self, video_path: Path, overlay_path: Path) -> dict:
        self.phone_calls += 1
        return {
            "result": "pass",
            "label": "KEEP_NON_CALL_PHONE_USE",
            "model_version": self.model_version,
            "prompt_revision": PROMPT_REVISION,
            "evidence_revision": VLM_EVIDENCE_REVISION,
            "candidate_count": 1,
            "candidate_labels": ["KEEP_NON_CALL_PHONE_USE"],
            "evidence_complete": True,
        }

    def review_capture(
        self,
        video_path: Path,
        overlay_path: Path,
        visibility_path: Path,
        cancel_event: threading.Event,
    ) -> dict:
        self.capture_calls += 1
        if cancel_event.is_set():
            self.cancelled_calls += 1
            return {
                "label": "UNCERTAIN",
                "cancelled": True,
                "model_version": self.model_version,
                "prompt_revision": CAPTURE_PROMPT_REVISION,
                "evidence_revision": CAPTURE_EVIDENCE_REVISION,
                "candidate_count": 0,
                "candidate_labels": [],
                "evidence_complete": False,
            }
        return {
            "label": "CAPTURE_POSSIBLE",
            "cancelled": False,
            "model_version": self.model_version,
            "prompt_revision": CAPTURE_PROMPT_REVISION,
            "evidence_revision": CAPTURE_EVIDENCE_REVISION,
            "candidate_count": 1,
            "candidate_labels": ["CAPTURE_POSSIBLE"],
            "evidence_complete": True,
        }


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _stage_one_archive(tmp_path: Path, event_id: str = "event-one") -> bytes:
    clip = tmp_path / "clip.mp4"
    overlay = tmp_path / "overlay.json"
    clip.write_bytes(b"stage-one-clip")
    overlay.write_text(json.dumps({"event_id": event_id}), encoding="utf-8")
    request_id = evidence_request_id(
        event_id,
        clip,
        overlay,
        model_version=MODEL_VERSION,
        prompt_revision=PROMPT_REVISION,
        evidence_revision=VLM_EVIDENCE_REVISION,
    )
    return _archive(
        {
            "request.json": json.dumps(
                {
                    "schema_version": 1,
                    "event_id": event_id,
                    "request_id": request_id,
                    "prompt_revision": PROMPT_REVISION,
                    "evidence_revision": VLM_EVIDENCE_REVISION,
                }
            ).encode(),
            "clip.mp4": clip.read_bytes(),
            "overlay.json": overlay.read_bytes(),
        }
    )


def _capture_archive(tmp_path: Path, event_id: str = "event-capture") -> bytes:
    clip = tmp_path / f"{event_id}.mp4"
    overlay = tmp_path / f"{event_id}-overlay.json"
    visibility = tmp_path / f"{event_id}-visibility.json"
    clip.write_bytes(b"capture-clip")
    overlay.write_text(json.dumps({"event_id": event_id}), encoding="utf-8")
    visibility.write_text(
        json.dumps({"schema_version": 1, "camera": "camera08"}), encoding="utf-8"
    )
    request_id = capture_request_id(
        event_id,
        clip,
        overlay,
        visibility,
        model_version=MODEL_VERSION,
        prompt_revision=CAPTURE_PROMPT_REVISION,
        evidence_revision=CAPTURE_EVIDENCE_REVISION,
    )
    return _archive(
        {
            "request.json": json.dumps(
                {
                    "schema_version": 1,
                    "event_id": event_id,
                    "request_id": request_id,
                    "prompt_revision": CAPTURE_PROMPT_REVISION,
                    "evidence_revision": CAPTURE_EVIDENCE_REVISION,
                }
            ).encode(),
            "clip.mp4": clip.read_bytes(),
            "overlay.json": overlay.read_bytes(),
            "visibility.json": visibility.read_bytes(),
        }
    )


def _application(tmp_path: Path, reviewer: FakeReviewer, priority_clock) -> ReviewApplication:
    return ReviewApplication(
        reviewer=reviewer,
        shared_secret=SECRET,
        cache_dir=tmp_path / "cache",
        max_request_bytes=1024 * 1024,
        clock=lambda: 1_700_000_000,
        priority_clock=priority_clock,
        offline_quiet_seconds=30,
    )


def _post(application: ReviewApplication, path: str, body: bytes) -> tuple[int, dict, dict]:
    server = BoundedReviewHTTPServer(("127.0.0.1", 0), make_handler(application))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        timestamp = "1700000000"
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/zip",
                "Content-Length": str(len(body)),
                "X-Jiankong-Timestamp": timestamp,
                "X-Jiankong-Signature": request_signature(SECRET, timestamp, body),
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        headers = dict(response.getheaders())
        connection.close()
        return response.status, payload, headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_stage_one_contract_and_cache_remain_unchanged(tmp_path: Path) -> None:
    assert PROMPT_REVISION == "600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993"
    assert VLM_EVIDENCE_REVISION == "person-roi20-span5s-pre4-native-focus-temporal-early8-jpeg92-v20"
    reviewer = FakeReviewer()
    now = [31.0]
    application = _application(tmp_path, reviewer, lambda: now[0])
    body = _stage_one_archive(tmp_path)

    first = application.review_archive(body)
    second = application.review_archive(body)

    assert first["label"] == "KEEP_NON_CALL_PHONE_USE"
    assert second["label"] == "KEEP_NON_CALL_PHONE_USE"
    assert reviewer.phone_calls == 1
    assert (tmp_path / "cache" / "event-one.json").is_file()
    assert not (tmp_path / "cache" / "capture" / "event-one.json").exists()


def test_capture_endpoint_uses_same_reviewer_and_distinct_cache(tmp_path: Path) -> None:
    reviewer = FakeReviewer()
    now = [0.0]
    application = _application(tmp_path, reviewer, lambda: now[0])
    now[0] = 31.0
    body = _capture_archive(tmp_path)

    status, payload, _headers = _post(application, "/v1/capture-review", body)
    cached_status, cached_payload, _ = _post(application, "/v1/capture-review", body)

    assert status == cached_status == 200
    assert payload["label"] == cached_payload["label"] == "CAPTURE_POSSIBLE"
    assert reviewer.capture_calls == 1
    assert reviewer.phone_calls == 0
    assert (tmp_path / "cache" / "capture" / "event-capture.json").is_file()


def test_production_endpoint_is_still_v1_review(tmp_path: Path) -> None:
    reviewer = FakeReviewer()
    application = _application(tmp_path, reviewer, lambda: 31.0)

    status, payload, _headers = _post(
        application, "/v1/review", _stage_one_archive(tmp_path)
    )

    assert status == 200
    assert payload["label"] == "KEEP_NON_CALL_PHONE_USE"
    assert reviewer.phone_calls == 1


def test_offline_endpoint_waits_for_quiet_period(tmp_path: Path) -> None:
    reviewer = FakeReviewer()
    now = [0.0]
    application = _application(tmp_path, reviewer, lambda: now[0])

    status, payload, headers = _post(
        application, "/v1/capture-review", _capture_archive(tmp_path)
    )

    assert status == 503
    assert payload["error"] == "review service is busy"
    assert headers["Retry-After"] == "30"
    assert reviewer.capture_calls == 0


def test_cancelled_capture_review_is_not_cached(tmp_path: Path) -> None:
    reviewer = FakeReviewer()
    application = _application(tmp_path, reviewer, lambda: 31.0)
    body = _capture_archive(tmp_path)
    cancelled = threading.Event()
    cancelled.set()

    first = application.capture_archive(body, cancelled)
    second = application.capture_archive(body, cancelled)

    assert first["label"] == second["label"] == "UNCERTAIN"
    assert reviewer.capture_calls == 2
    assert not (tmp_path / "cache" / "capture" / "event-capture.json").exists()


def test_health_exposes_capture_identity_and_scheduler(tmp_path: Path) -> None:
    reviewer = FakeReviewer()
    application = _application(tmp_path, reviewer, lambda: 31.0)

    health = application.health()

    assert health["capture_prompt_revision"] == CAPTURE_PROMPT_REVISION
    assert health["capture_evidence_revision"] == CAPTURE_EVIDENCE_REVISION
    assert health["scheduler"]["active_kind"] is None


def test_service_defaults_to_immediate_nonpreemptive_validation() -> None:
    args = _parser().parse_args(
        [
            "--model", "/model",
            "--model-version", "model-v1",
            "--shared-secret-file", "/secret",
            "--cache-dir", "/cache",
            "--gpu-lock-file", "/lock",
        ]
    )

    assert args.offline_quiet_seconds == 0.0


def test_generation_cancel_mask_is_one_boolean_per_batch_item() -> None:
    calls = []

    class FakeTorch:
        bool = "bool"

        @staticmethod
        def full(shape, value, *, dtype, device):
            calls.append((shape, value, dtype, device))
            return "mask"

    class InputIds:
        shape = (3, 20)
        device = "cuda:0"

    cancelled = threading.Event()

    assert _generation_cancel_mask(FakeTorch, InputIds(), cancelled) == "mask"
    assert calls == [((3,), False, "bool", "cuda:0")]


def test_stage_one_selects_twenty_frames_with_four_before_alarm() -> None:
    entries = [
        {
            "track_id": "person-1",
            "time_sec": index * 0.25,
            "frame_index": index,
            "alarm": index == 8,
        }
        for index in range(40)
    ]

    sequences = select_candidate_sequences(
        {"bbox_timeline": entries},
        frame_count=20,
        target_span_seconds=5.0,
        pre_alarm_frames=4,
        video_fps=4.0,
        video_frame_count=40,
    )

    assert len(sequences) == 1
    selected = sequences[0].entries
    assert len(selected) == 20
    assert [entry["frame_index"] for entry in selected[:4]] == [4, 5, 6, 7]
    assert selected[4]["frame_index"] == 8
    assert len({entry["frame_index"] for entry in selected}) == 20


def test_reviewer_initializes_production_and_capture_prompts_but_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    fingerprint = hashlib.sha256(b"config.json\0{}\0").hexdigest()

    class Processor:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, **_kwargs):
            self.calls.append(messages)
            return f"chat-{len(self.calls)}"

    processor = Processor()
    model = object()
    loads = []

    def fake_load(*_args):
        loads.append(True)
        return processor, model

    monkeypatch.setattr(MageVLReviewer, "_load_model", staticmethod(fake_load))
    reviewer = MageVLReviewer(
        model_path=model_path,
        model_version=f"fake-{fingerprint[:12]}",
        gpu_weight_memory="1GiB",
        cpu_memory="1GiB",
    )

    assert loads == [True]
    assert reviewer.model is model
    assert reviewer.chat_text == "chat-1"
    assert reviewer.focus_chat_text == "chat-2"
    assert reviewer.early_rescue_chat_text == "chat-3"
    assert reviewer.capture_chat_text == "chat-4"
    prompt_texts = [call[0]["content"][1]["text"] for call in processor.calls]
    assert prompt_texts == [
        NATIVE_PROMPT,
        FOCUS_PROMPT,
        EARLY_RESCUE_PROMPT,
        CAPTURE_PROMPT,
    ]
