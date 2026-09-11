import json
import os
import subprocess
import sys
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from live_operator.events import EventCoordinator, SourceGenerationMismatch, write_source_generation


RUN_STARTED_AT = datetime(2026, 7, 17, 9, 30, tzinfo=timezone.utc)


def _frame(camera, frame, time_sec, people):
    return {"stream_index": camera, "frame_index": frame, "time_sec": time_sec, "persons": people}


def _person(person_id, alarm, risk, box):
    return {"track_id": person_id, "alarm": alarm, "risk_score": risk, "box": box}


def _append(path: Path, *records, trailing_newline=True):
    with path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(json.dumps(record))
            if trailing_newline or index < len(records) - 1:
                handle.write("\n")


def _coordinator(source, output, **kwargs):
    generation_file = source.with_name("source_generation.json")
    if not generation_file.exists():
        _write_generation(generation_file, "generation-1", RUN_STARTED_AT)
    return EventCoordinator(
        source,
        output,
        generation_file=generation_file,
        run_started_at=RUN_STARTED_AT,
        infer_fps=10.0,
        **kwargs,
    )


def _write_generation(path, generation_id, started_at):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "generation_id": generation_id,
        "started_at": started_at.isoformat(timespec="microseconds"),
    }) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@pytest.mark.parametrize("generation_id", [
    "../escape",
    "nested/path",
    r"nested\path",
    "   ",
    "a" * 129,
    "valid..but-forbidden",
    "control\x01char",
])
def test_generation_id_rejects_path_traversal_and_unsafe_tokens(tmp_path, generation_id):
    generation_file = tmp_path / "source_generation.json"
    with pytest.raises(ValueError, match="generation_id"):
        write_source_generation(generation_file, generation_id, RUN_STARTED_AT)

    generation_file.write_text(json.dumps({
        "generation_id": generation_id,
        "started_at": RUN_STARTED_AT.isoformat(),
    }), encoding="utf-8")
    coordinator = EventCoordinator(
        tmp_path / "frame_events.jsonl",
        tmp_path / "dashboard",
        generation_file=generation_file,
        run_started_at=RUN_STARTED_AT,
        infer_fps=10,
    )
    with pytest.raises(ValueError, match="generation_id"):
        coordinator.poll()


def test_source_generation_writer_round_trip_is_atomic_and_private(tmp_path):
    generation_file = tmp_path / "run" / "source_generation.json"
    started_at = datetime(2026, 7, 17, 17, 0, tzinfo=timezone.utc)
    write_source_generation(generation_file, "producer_20260717-01", started_at)

    assert json.loads(generation_file.read_text(encoding="utf-8")) == {
        "generation_id": "producer_20260717-01",
        "started_at": "2026-07-17T17:00:00.000000+00:00",
    }
    assert list(generation_file.parent.glob("*.tmp")) == []
    assert list(generation_file.parent.glob(".*.tmp")) == []
    if os.name != "nt":
        assert stat.S_IMODE(generation_file.stat().st_mode) == 0o600

    source = generation_file.with_name("frame_events.jsonl")
    _append(source, _frame(0, 1, 0.5, [_person(1, True, 0.8, [0, 0, 1, 1])]))
    event = EventCoordinator(
        source,
        tmp_path / "dashboard",
        generation_file=generation_file,
        run_started_at=RUN_STARTED_AT,
        infer_fps=10,
        stable_frames=1,
    ).poll()[0]
    assert event["source_generation"] == "producer_20260717-01"
    assert event["occurred_at"] == "2026-07-17T17:00:00.500000+00:00"


def test_real_writer_fields_map_camera_and_run_relative_time(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    literal_writer_record = {
        "stream_index": 0,
        "frame_index": 11,
        "frame_id": 12,
        "time_sec": 1.1,
        "width": 1920,
        "height": 1080,
        "input_video": "rtsp://127.0.0.1:8554/camera01",
        "accepted_count": 1,
        "alarm_track_count": 1,
        "persons": [{
            "person_index": 0,
            "track_id": 7,
            "box": [1, 2, 3, 4],
            "raw_box": [1, 2, 3, 4],
            "visible": True,
            "risk": True,
            "alarm": True,
            "suspect": True,
            "state": "S4_ALARM",
            "risk_score": 0.9,
        }],
    }
    _append(source, literal_writer_record)

    event = _coordinator(source, output, stable_frames=1).poll()[0]

    assert event["camera"] == "camera01"
    assert event["event_stream_time_sec"] == 1.1
    assert event["occurred_at"] == "2026-07-17T09:30:01.100000+00:00"
    assert event["risk_peak"] == pytest.approx(0.9)
    assert len(event["bbox_timeline"]) == 1
    assert event["bbox_timeline"][0] == {
        "time_sec": 1.1,
        "frame_index": 11,
        "bbox": [1, 2, 3, 4],
        "track_id": "7",
        "risk_score": 0.9,
        "alarm": True,
        "state": "S4_ALARM",
    }
    assert event["level"] == "alarm"
    assert event["event_id"].startswith("camera-camera01_person-7_source-")
    assert "_alarm-" in event["event_id"]


def test_event_timeline_embeds_person_roi_and_matched_phone_prelabels(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    record = _frame(
        3,
        42,
        5.25,
        [
            {
                **_person(17, True, 0.91, [100, 50, 300, 350]),
                "roi": [80, 40, 320, 380],
            }
        ],
    )
    record.update(
        {
            "width": 640,
            "height": 480,
            "phones": [
                {
                    "track_id": 17,
                    "box": [180, 160, 220, 240],
                    "confidence": 0.75,
                    "phone_score": 0.88,
                    "accepted": True,
                },
                {
                    "track_id": 99,
                    "box": [400, 100, 430, 160],
                    "confidence": 0.80,
                },
            ],
        }
    )
    _append(source, record)

    event = _coordinator(source, output, stable_frames=1).poll()[0]

    sample = event["bbox_timeline"][0]
    assert sample["roi"] == [80, 40, 320, 380]
    assert sample["frame_width"] == 640
    assert sample["frame_height"] == 480
    assert sample["phone_boxes"] == [
        {
            "box": [180, 160, 220, 240],
            "confidence": 0.75,
            "phone_score": 0.88,
            "accepted": True,
        }
    ]


def test_captured_at_controls_event_time_and_is_preserved_in_timeline(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    captured_at = "2026-07-17T17:42:03.125Z"
    record = _frame(
        0,
        11,
        1.1,
        [_person(7, True, 0.9, [1, 2, 3, 4])],
    )
    record["captured_at"] = captured_at
    _append(source, record)

    event = _coordinator(source, output, stable_frames=1).poll()[0]

    expected = "2026-07-17T17:42:03.125000+00:00"
    assert event["occurred_at"] == expected
    assert event["bbox_timeline"][0]["captured_at"] == expected
    expected_ms = int(datetime.fromisoformat(expected).timestamp() * 1000)
    assert event["event_id"].endswith(f"_alarm-{expected_ms}")


@pytest.mark.parametrize(
    "captured_at",
    ["not-a-timestamp", "2026-07-17T17:42:03.125000"],
    ids=["invalid", "timezone-naive"],
)
def test_invalid_captured_at_keeps_generation_relative_event_time(
    tmp_path, captured_at
):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    record = _frame(0, 11, 1.1, [_person(7, True, 0.9, [1, 2, 3, 4])])
    record["captured_at"] = captured_at
    _append(source, record)

    event = _coordinator(source, output, stable_frames=1).poll()[0]

    assert event["occurred_at"] == "2026-07-17T09:30:01.100000+00:00"
    assert "captured_at" not in event["bbox_timeline"][0]


def test_event_overlay_keeps_pre_alarm_and_post_alarm_track_context(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    coordinator = _coordinator(source, output, stable_frames=2)
    _append(
        source,
        _frame(0, 1, 0.1, [_person(7, False, 0.2, [1, 1, 11, 21])]),
        _frame(0, 2, 0.2, [_person(7, False, 0.4, [2, 1, 12, 21])]),
        _frame(0, 3, 0.3, [_person(7, True, 0.8, [3, 1, 13, 21])]),
        _frame(0, 4, 0.4, [_person(7, True, 0.9, [4, 1, 14, 21])]),
    )

    event = coordinator.poll()[0]
    assert [sample["frame_index"] for sample in event["bbox_timeline"]] == [1, 2, 3, 4]
    assert [sample["alarm"] for sample in event["bbox_timeline"]] == [False, False, True, True]

    _append(source, _frame(0, 5, 0.5, [_person(7, False, 0.3, [5, 1, 15, 21])]))
    event = coordinator.poll()[0]
    assert [sample["frame_index"] for sample in event["bbox_timeline"]] == [1, 2, 3, 4, 5]
    assert event["bbox_timeline"][-1]["alarm"] is False


def test_poll_only_rewrites_state_when_no_event_changed(tmp_path, monkeypatch):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    coordinator = _coordinator(source, output, stable_frames=1)
    _append(source, _frame(0, 1, 0.1, [_person(7, True, 0.9, [1, 1, 11, 21])]))
    coordinator.poll()

    writes = []
    original = EventCoordinator._atomic_json

    def recording_write(path, value):
        writes.append(Path(path))
        return original(path, value)

    monkeypatch.setattr(EventCoordinator, "_atomic_json", staticmethod(recording_write))
    _append(source, _frame(1, 1, 0.1, [_person(9, False, 0.2, [2, 2, 12, 22])]))
    coordinator.poll()

    assert writes == [output / ".events_state.json"]


def test_missing_tracks_and_expired_cooldowns_are_pruned_from_state(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    coordinator = _coordinator(source, output, stable_frames=1, cooldown_sec=10)
    _append(source, _frame(0, 1, 0.1, [_person(7, True, 0.9, [1, 1, 11, 21])]))
    coordinator.poll()

    _append(source, _frame(0, 102, 10.2, []))
    coordinator.poll()
    state = json.loads((output / ".events_state.json").read_text(encoding="utf-8"))

    assert state["tracks"] == {}
    assert state["last_alarm"] == {}


def test_risk_only_frames_do_not_create_frontend_event(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(
        source,
        _frame(0, 1, 0.1, [_person(7, False, 0.75, [1, 1, 11, 21])]),
        _frame(0, 2, 0.2, [_person(7, False, 0.82, [2, 1, 12, 21])]),
        _frame(0, 3, 0.3, [_person(7, False, 0.91, [3, 1, 13, 21])]),
    )

    assert _coordinator(source, output, stable_frames=2).poll() == []


def test_requires_timezone_aware_start_and_positive_fps(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        EventCoordinator(tmp_path / "in", tmp_path / "out", generation_file=tmp_path / "gen", run_started_at=datetime(2026, 1, 1), infer_fps=10)
    with pytest.raises(ValueError, match="infer_fps"):
        EventCoordinator(tmp_path / "in", tmp_path / "out", generation_file=tmp_path / "gen", run_started_at=RUN_STARTED_AT, infer_fps=0)


def test_aggregates_by_camera_and_person_with_peak_and_timeline(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(
        source,
        _frame(0, 1, 1.0, [_person(7, True, 0.6, [1, 2, 3, 4])]),
        _frame(1, 1, 1.0, [_person(7, True, 0.7, [5, 6, 7, 8])]),
        _frame(0, 2, 1.1, [_person(7, True, 0.9, [2, 3, 3, 4])]),
        _frame(1, 2, 1.1, [_person(7, True, 0.8, [6, 7, 7, 8])]),
    )
    events = _coordinator(source, output, stable_frames=2, cooldown_sec=30).poll()

    assert {(event["camera"], event["person_id"]) for event in events} == {("camera01", "7"), ("camera02", "7")}
    first = next(event for event in events if event["camera"] == "camera01")
    assert first["risk_peak"] == 0.9
    assert [point["time_sec"] for point in first["bbox_timeline"]] == [1.0, 1.1]
    assert json.loads((output / "events.json").read_text(encoding="utf-8")) == events
    overlay = json.loads((output / "events" / first["event_id"] / "overlay.json").read_text(encoding="utf-8"))
    assert overlay["bbox_timeline"] == first["bbox_timeline"]


def test_simultaneous_alarm_tracks_share_one_camera_event_and_keep_both_boxes(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(
        source,
        _frame(
            0,
            1,
            1.0,
            [
                _person(7, True, 0.81, [1, 2, 11, 22]),
                _person(9, True, 0.84, [21, 2, 31, 22]),
            ],
        ),
        _frame(
            0,
            2,
            1.1,
            [
                _person(7, True, 0.88, [2, 2, 12, 22]),
                _person(9, True, 0.93, [22, 2, 32, 22]),
            ],
        ),
    )

    events = _coordinator(source, output, stable_frames=2, cooldown_sec=30).poll()

    assert len(events) == 1
    event = events[0]
    assert event["camera"] == "camera01"
    assert event["person_id"] == "7"
    assert event["person_ids"] == ["7", "9"]
    assert event["alarm_track_count"] == 2
    assert event["risk_peak"] == pytest.approx(0.93)
    frame_two = [
        point for point in event["bbox_timeline"] if point["frame_index"] == 2
    ]
    assert [point["track_id"] for point in frame_two] == ["7", "9"]
    assert [point["bbox"] for point in frame_two] == [
        [2, 2, 12, 22],
        [22, 2, 32, 22],
    ]
    overlay = json.loads(
        (output / "events" / event["event_id"] / "overlay.json").read_text(
            encoding="utf-8"
        )
    )
    assert overlay["person_ids"] == ["7", "9"]
    assert overlay["alarm_track_count"] == 2
    assert overlay["bbox_timeline"] == event["bbox_timeline"]


def test_alarm_tracks_outside_ten_second_group_window_create_separate_events(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(
        source,
        _frame(0, 1, 1.0, [_person(7, True, 0.81, [1, 2, 11, 22])]),
        _frame(0, 112, 11.1, [_person(9, True, 0.93, [21, 2, 31, 22])]),
    )

    events = _coordinator(source, output, stable_frames=1, cooldown_sec=30).poll()

    assert len(events) == 2
    assert [event["person_ids"] for event in events] == [["7"], ["9"]]


def test_alarm_tracks_within_ten_second_group_window_share_one_event(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(
        source,
        _frame(0, 1, 1.0, [_person(7, True, 0.81, [1, 2, 11, 22])]),
        _frame(0, 96, 9.6, [_person(9, True, 0.93, [21, 2, 31, 22])]),
    )

    events = _coordinator(source, output, stable_frames=1, cooldown_sec=30).poll()

    assert len(events) == 1
    assert events[0]["person_ids"] == ["7", "9"]


def test_reidentified_track_with_same_box_is_suppressed_during_cooldown(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(
        source,
        _frame(0, 1, 0.0, [_person(7, True, 0.81, [10, 20, 30, 40])]),
        _frame(0, 2, 0.1, [_person(7, False, 0.0, [10, 20, 30, 40])]),
        _frame(0, 121, 12.0, [_person(99, True, 0.93, [11, 20, 30, 40])]),
    )

    events = _coordinator(source, output, stable_frames=1, cooldown_sec=60).poll()

    assert len(events) == 1
    assert events[0]["person_ids"] == ["7"]


def test_reidentified_track_at_a_different_location_is_not_suppressed(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(
        source,
        _frame(0, 1, 0.0, [_person(7, True, 0.81, [10, 20, 30, 40])]),
        _frame(0, 2, 0.1, [_person(7, False, 0.0, [10, 20, 30, 40])]),
        _frame(0, 121, 12.0, [_person(99, True, 0.93, [200, 20, 30, 40])]),
    )

    events = _coordinator(source, output, stable_frames=1, cooldown_sec=60).poll()

    assert len(events) == 2
    assert [event["person_ids"] for event in events] == [["7"], ["99"]]


@pytest.mark.parametrize("break_kind", ["gap", "non_monotonic", "absent"])
def test_stable_alarm_requires_strictly_consecutive_present_frames(tmp_path, break_kind):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    records = [_frame(0, 10, 1.0, [_person(2, True, 0.5, [0, 0, 1, 1])])]
    if break_kind == "gap":
        records.append(_frame(0, 12, 1.2, [_person(2, True, 0.6, [0, 0, 1, 1])]))
    elif break_kind == "non_monotonic":
        records.append(_frame(0, 10, 1.1, [_person(2, True, 0.6, [0, 0, 1, 1])]))
    else:
        records.extend([
            _frame(0, 11, 1.1, [_person(9, True, 0.4, [0, 0, 1, 1])]),
            _frame(0, 12, 1.2, [_person(2, True, 0.6, [0, 0, 1, 1])]),
        ])
    _append(source, *records)
    assert _coordinator(source, output, stable_frames=2).poll() == []


def test_partial_line_and_offset_survive_restart(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    first = _frame("camera_a", 10, 2.0, [_person("p1", True, 0.5, [1, 1, 2, 2])])
    second = _frame("camera_a", 11, 2.1, [_person("p1", True, 0.8, [2, 2, 2, 2])])
    _append(source, first)
    encoded = json.dumps(second)
    with source.open("a", encoding="utf-8") as handle:
        handle.write(encoded[:20])
    _coordinator(source, output, stable_frames=2).poll()
    with source.open("a", encoding="utf-8") as handle:
        handle.write(encoded[20:] + "\n")

    events = _coordinator(source, output, stable_frames=2).poll()
    assert len(events) == 1
    assert events[0]["risk_peak"] == 0.8
    assert _coordinator(source, output, stable_frames=2).poll() == events


def test_cooldown_suppresses_repeat_then_allows_later_event(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    coordinator = _coordinator(source, output, stable_frames=2, cooldown_sec=10)
    _append(source,
        _frame(0, 1, 0.0, [_person(3, True, 0.5, [0, 0, 1, 1])]),
        _frame(0, 2, 0.1, [_person(3, True, 0.6, [0, 0, 1, 1])]),
        _frame(0, 3, 1.0, [_person(3, False, 0.0, [0, 0, 1, 1])]),
        _frame(0, 4, 2.0, [_person(3, True, 0.7, [0, 0, 1, 1])]),
        _frame(0, 5, 2.1, [_person(3, True, 0.8, [0, 0, 1, 1])]))
    assert len(coordinator.poll()) == 1
    _append(source,
        _frame(0, 11, 11.0, [_person(3, False, 0.0, [0, 0, 1, 1])]),
        _frame(0, 12, 12.0, [_person(3, True, 0.9, [0, 0, 1, 1])]),
        _frame(0, 13, 12.1, [_person(3, True, 1.0, [0, 0, 1, 1])]))
    assert len(coordinator.poll()) == 2


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_publish_failure_replay_is_idempotent(tmp_path, monkeypatch, failure_call):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(source, _frame(0, 1, 1.0, [_person(1, True, 0.9, [0, 0, 1, 1])]))
    coordinator = _coordinator(source, output, stable_frames=1)
    original = EventCoordinator._atomic_json
    calls = 0

    def flaky(path, value):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("injected publish failure")
        return original(path, value)

    monkeypatch.setattr(EventCoordinator, "_atomic_json", staticmethod(flaky))
    with pytest.raises(OSError, match="injected"):
        coordinator.poll()
    monkeypatch.setattr(EventCoordinator, "_atomic_json", staticmethod(original))
    coordinator.close()

    events = _coordinator(source, output, stable_frames=1).poll()
    assert len(events) == 1
    assert len({event["event_id"] for event in events}) == 1


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_multiframe_publish_replay_merges_timeline_idempotently(tmp_path, monkeypatch, failure_call):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(source,
        _frame(0, 1, 0.1, [_person(1, True, 0.5, [0, 0, 1, 1])]),
        _frame(0, 2, 0.2, [_person(1, True, 0.7, [1, 1, 2, 2])]),
        _frame(0, 3, 0.3, [_person(1, True, 0.9, [2, 2, 3, 3])]))
    coordinator = _coordinator(source, output, stable_frames=2)
    original = EventCoordinator._atomic_json
    calls = 0

    def flaky(path, value):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("injected publish failure")
        return original(path, value)

    monkeypatch.setattr(EventCoordinator, "_atomic_json", staticmethod(flaky))
    with pytest.raises(OSError, match="injected"):
        coordinator.poll()
    monkeypatch.setattr(EventCoordinator, "_atomic_json", staticmethod(original))
    coordinator.close()

    event = _coordinator(source, output, stable_frames=2).poll()[0]
    assert [(point["frame_index"], point["time_sec"]) for point in event["bbox_timeline"]] == [
        (1, 0.1), (2, 0.2), (3, 0.3)
    ]
    assert event["risk_peak"] == 0.9


def test_atomic_json_cleans_temporary_file_after_replace_failure(tmp_path, monkeypatch):
    target = tmp_path / "events.json"
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        EventCoordinator._atomic_json(target, [])
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("replacement_size", ["smaller", "regrown"])
def test_source_replacement_resets_offset_even_when_size_recovers(tmp_path, replacement_size):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(source, _frame(0, 1, 0.1, [_person(1, True, 0.5, [0, 0, 1, 1])]))
    coordinator = _coordinator(source, output, stable_frames=1)
    assert len(coordinator.poll()) == 1
    replacement = _frame(1, 1, 0.1, [_person(2, True, 0.8, [1, 1, 2, 2])])
    text = json.dumps(replacement) + "\n"
    if replacement_size == "regrown":
        replacement["padding"] = "x" * 500
        text = json.dumps(replacement) + "\n"
    _write_generation(source.with_name("source_generation.json"), "generation-2", datetime(2026, 7, 17, 9, 45, tzinfo=timezone.utc))
    source.write_text(text, encoding="utf-8")

    events = coordinator.poll()
    assert {(event["camera"], event["person_id"]) for event in events} == {("camera01", "1"), ("camera02", "2")}


def test_source_generation_prevents_same_alarm_identity_collision(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    first_record = _frame(0, 1, 0.1, [_person(1, True, 0.5, [0, 0, 1, 1])])
    _append(source, first_record)
    coordinator = _coordinator(source, output, stable_frames=1)
    first = coordinator.poll()[0]
    coordinator.set_status(first["event_id"], "ready")
    replacement = _frame(0, 1, 0.1, [_person(1, True, 0.9, [5, 5, 6, 6])])
    replacement["generation_marker"] = "replacement"
    generation_started_at = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)
    _write_generation(source.with_name("source_generation.json"), "generation-2", generation_started_at)
    source.write_text(json.dumps(replacement) + "\n", encoding="utf-8")

    events = coordinator.poll()
    assert len(events) == 2
    assert events[0]["event_id"] != events[1]["event_id"]
    assert events[0]["source_generation"] != events[1]["source_generation"]
    assert events[0]["status"] == "ready"
    assert events[1]["status"] == "collecting"
    assert events[1]["generation_started_at"] == "2026-07-17T10:00:00.000000+00:00"
    assert events[1]["occurred_at"] == "2026-07-17T10:00:00.100000+00:00"


def test_mid_poll_source_change_without_sidecar_update_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(source, _frame(0, 1, 0.1, [_person(1, True, 0.5, [0, 0, 1, 1])]))
    coordinator = _coordinator(source, output, stable_frames=1)
    replacement = _frame(1, 1, 0.1, [_person(2, True, 0.9, [1, 1, 2, 2])])
    replacement["generation_marker"] = "changed-during-poll"

    def replace_source(_handle):
        source.write_text(json.dumps(replacement) + "\n", encoding="utf-8")

    monkeypatch.setattr(coordinator, "_after_source_validation", replace_source)
    with pytest.raises(SourceGenerationMismatch):
        coordinator.poll()
    monkeypatch.setattr(coordinator, "_after_source_validation", lambda _handle: None)
    _write_generation(source.with_name("source_generation.json"), "generation-2", datetime(2026, 7, 17, 9, 50, tzinfo=timezone.utc))
    events = coordinator.poll()
    assert [(event["camera"], event["person_id"]) for event in events] == [("camera02", "2")]


def test_sidecar_started_at_not_run_start_controls_occurrence_time(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    generation_start = datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc)
    _write_generation(source.with_name("source_generation.json"), "late-generation", generation_start)
    _append(source, _frame(0, 1, 2.5, [_person(1, True, 0.8, [0, 0, 1, 1])]))

    event = _coordinator(source, output, stable_frames=1).poll()[0]
    assert event["generation_started_at"] == "2026-07-17T15:00:00.000000+00:00"
    assert event["occurred_at"] == "2026-07-17T15:00:02.500000+00:00"
    assert "source-late-generation" in event["event_id"]


def test_same_inode_same_first_line_new_sidecar_creates_new_generation(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    record = _frame(0, 1, 0.1, [_person(1, True, 0.5, [0, 0, 1, 1])])
    line = json.dumps(record) + "\n"
    source.write_text(line, encoding="utf-8")
    coordinator = _coordinator(source, output, stable_frames=1)
    first = coordinator.poll()[0]
    _write_generation(source.with_name("source_generation.json"), "generation-2", datetime(2026, 7, 17, 16, 0, tzinfo=timezone.utc))
    source.write_text(line, encoding="utf-8")

    events = coordinator.poll()
    assert len(events) == 2
    assert events[0]["source_generation"] == "generation-1"
    assert events[1]["source_generation"] == "generation-2"


def test_copytruncate_without_generation_update_stops_consumption(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(source, _frame(0, 1, 0.1, [_person(1, True, 0.5, [0, 0, 1, 1])]))
    coordinator = _coordinator(source, output, stable_frames=1)
    assert len(coordinator.poll()) == 1
    source.write_text(json.dumps(_frame(1, 1, 0.1, [_person(2, True, 0.8, [1, 1, 2, 2])])) + "\n", encoding="utf-8")

    with pytest.raises(SourceGenerationMismatch):
        coordinator.poll()
    assert len(json.loads((output / "events.json").read_text(encoding="utf-8"))) == 1


def test_normal_append_during_poll_is_consumed_without_starvation(tmp_path, monkeypatch):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(source, _frame(0, 1, 0.1, [_person(1, True, 0.5, [0, 0, 1, 1])]))
    coordinator = _coordinator(source, output, stable_frames=1)
    appended = False

    def append_during_poll(_handle):
        nonlocal appended
        if not appended:
            appended = True
            _append(source, _frame(0, 2, 0.2, [_person(1, True, 0.6, [1, 1, 2, 2])]))

    monkeypatch.setattr(coordinator, "_after_source_validation", append_during_poll)
    event = coordinator.poll()[0]
    assert [point["frame_index"] for point in event["bbox_timeline"]] == [1, 2]


def test_empty_source_at_start_accepts_first_normal_append(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    source.touch()
    coordinator = _coordinator(source, output, stable_frames=2)

    assert coordinator.poll() == []
    _append(
        source,
        _frame(0, 1, 0.1, [_person(1, True, 0.7, [0, 0, 1, 1])]),
        _frame(0, 2, 0.2, [_person(1, True, 0.9, [1, 1, 2, 2])]),
    )

    event = coordinator.poll()[0]
    assert event["camera"] == "camera01"
    assert event["person_id"] == "1"
    assert event["risk_peak"] == 0.9


def test_state_failure_then_append_replays_with_same_generation_and_event_id(tmp_path, monkeypatch):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(source,
        _frame(0, 1, 0.1, [_person(1, True, 0.5, [0, 0, 1, 1])]),
        _frame(0, 2, 0.2, [_person(1, True, 0.7, [1, 1, 2, 2])]),
        _frame(0, 3, 0.3, [_person(1, True, 0.9, [2, 2, 3, 3])]))
    coordinator = _coordinator(source, output, stable_frames=2)
    original = EventCoordinator._atomic_json
    calls = 0

    def fail_state(path, value):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("state checkpoint failed")
        return original(path, value)

    monkeypatch.setattr(EventCoordinator, "_atomic_json", staticmethod(fail_state))
    with pytest.raises(OSError, match="checkpoint"):
        coordinator.poll()
    published_id = json.loads((output / "events.json").read_text(encoding="utf-8"))[0]["event_id"]
    monkeypatch.setattr(EventCoordinator, "_atomic_json", staticmethod(original))
    _append(source, _frame(0, 4, 0.4, [_person(1, True, 1.0, [3, 3, 4, 4])]))
    coordinator.close()

    events = _coordinator(source, output, stable_frames=2).poll()
    assert len(events) == 1
    assert events[0]["event_id"] == published_id
    assert [point["frame_index"] for point in events[0]["bbox_timeline"]] == [1, 2, 3, 4]


def test_failed_status_error_and_terminal_event_are_immutable(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    _append(source, _frame(0, 1, 1.0, [_person(1, True, 0.5, [0, 0, 1, 1])]))
    coordinator = _coordinator(source, output, stable_frames=1)
    event_id = coordinator.poll()[0]["event_id"]
    failed = coordinator.set_status(event_id, "failed", error="clip unavailable")
    assert failed["status"] == "failed"
    assert failed["error"] == "clip unavailable"
    _append(source, _frame(0, 2, 1.1, [_person(1, True, 1.0, [1, 1, 2, 2])]))

    coordinator.close()
    restarted = _coordinator(source, output, stable_frames=1)
    after = restarted.poll()[0]
    assert after == failed
    with pytest.raises(ValueError, match="terminal"):
        restarted.set_status(event_id, "ready")


def test_poll_and_set_status_are_serialized(tmp_path, monkeypatch):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    coordinator = _coordinator(source, output, stable_frames=1)
    _append(source, _frame(0, 1, 1.0, [_person(1, True, 0.5, [0, 0, 1, 1])]))
    event_id = coordinator.poll()[0]["event_id"]
    _append(source, _frame(0, 2, 1.1, [_person(1, True, 0.6, [0, 0, 1, 1])]))
    entered = threading.Event()
    release = threading.Event()
    original = coordinator._consume

    def paused(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(coordinator, "_consume", paused)
    poll_thread = threading.Thread(target=coordinator.poll)
    status_thread = threading.Thread(target=lambda: coordinator.set_status(event_id, "ready"))
    poll_thread.start()
    assert entered.wait(5)
    status_thread.start()
    release.set()
    poll_thread.join(5)
    status_thread.join(5)
    assert not poll_thread.is_alive() and not status_thread.is_alive()
    assert json.loads((output / "events.json").read_text(encoding="utf-8"))[0]["status"] == "ready"


def test_output_directory_has_single_coordinator_owner(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    first = _coordinator(source, output)
    with pytest.raises(RuntimeError, match="already has an EventCoordinator owner"):
        _coordinator(source, output)
    first.close()
    with pytest.raises(RuntimeError, match="closed"):
        first.poll()
    with pytest.raises(RuntimeError, match="closed"):
        first.set_status("missing", "failed")
    second = _coordinator(source, output)
    second.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock integration test")
def test_posix_owner_lock_rejects_second_process(tmp_path):
    source = tmp_path / "frame_events.jsonl"
    output = tmp_path / "dashboard"
    first = _coordinator(source, output)
    script = """
from datetime import datetime, timezone
from live_operator.events import EventCoordinator
try:
    EventCoordinator(r'{source}', r'{output}', generation_file=r'{generation}', run_started_at=datetime.now(timezone.utc), infer_fps=10)
except RuntimeError:
    raise SystemExit(0)
raise SystemExit(3)
""".format(source=source, output=output, generation=source.with_name("source_generation.json"))
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), check=False)
    first.close()
    assert completed.returncode == 0
