"""Incrementally coordinate per-person alarm events from DeepStream JSONL."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from live_operator.clips import POST_EVENT_SECONDS
from live_operator.storage_paths import resolve_run_storage
from live_operator.vlm_review import VLM_FILTER_RESULTS, VLM_LABEL_BY_RESULT


GENERATION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
EVENT_CONTEXT_SECONDS = 5.0
# A camera alarm may contain several people entering the same scene slightly
# apart.  Five seconds keeps them in one review item while the clip itself
# retains 3 seconds before and 12 seconds after the first stable alarm.
EVENT_GROUP_WINDOW_SECONDS = 5.0
_REIDENTIFICATION_IOU_THRESHOLD = 0.50
_VLM_REQUEST_ID = re.compile(r"[A-Fa-f0-9]{64}\Z")
_VLM_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class SourceGenerationMismatch(RuntimeError):
    """The JSONL source changed without a matching producer generation update."""


class EventArchiveError(RuntimeError):
    """Archive history cannot be trusted; explicit operator recovery is required."""


def _validate_generation_id(generation_id: Any) -> str:
    if (
        not isinstance(generation_id, str)
        or GENERATION_ID_PATTERN.fullmatch(generation_id) is None
        or ".." in generation_id
    ):
        raise ValueError(
            "generation_id must be a safe 1-128 character token without '..' or path separators"
        )
    return generation_id


def write_source_generation(path: str | Path, generation_id: str, started_at: datetime) -> None:
    """Atomically publish private producer-generation metadata for EventCoordinator."""
    generation_id = _validate_generation_id(generation_id)
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("started_at must be timezone-aware")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            json.dump({
                "generation_id": generation_id,
                "started_at": started_at.isoformat(timespec="microseconds"),
            }, handle, ensure_ascii=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        if os.name != "nt":
            os.chmod(target, 0o600)
            directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class EventCoordinator:
    """Consume complete appended JSONL records without blocking the producer."""

    PREFIX_BYTES = 4096

    def __init__(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        generation_file: str | Path,
        run_started_at: datetime,
        infer_fps: float,
        stable_frames: int = 2,
        run_dir: str | Path | None = None,
        cooldown_sec: float = 180.0,
        stream_cameras: Mapping[int, str] | None = None,
    ) -> None:
        if run_started_at.tzinfo is None or run_started_at.utcoffset() is None:
            raise ValueError("run_started_at must be timezone-aware")
        if infer_fps <= 0:
            raise ValueError("infer_fps must be positive")
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        if cooldown_sec < 0:
            raise ValueError("cooldown_sec cannot be negative")
        self.source = Path(source)
        self.generation_file = Path(generation_file)
        self.output_dir = Path(output_dir)
        self.storage_run_dir = Path(run_dir) if run_dir is not None else None
        self.run_started_at = run_started_at
        self.infer_fps = float(infer_fps)
        self.stable_frames = stable_frames
        self.cooldown_sec = cooldown_sec
        self.stream_cameras = {
            int(index): str(camera)
            for index, camera in (stream_cameras or {}).items()
            if isinstance(index, int) and isinstance(camera, str) and camera
        }
        self.state_path = self.output_dir / ".events_state.json"
        self.events_path = self.output_dir / "events.json"
        self._archive_initialized = False
        self._saved_checkpoint_bytes: bytes | None = None
        self._lock = threading.RLock()
        self._owner_handle = None
        self._acquire_owner()

    def close(self) -> None:
        """Release this output directory's single-coordinator ownership lock."""
        with self._lock:
            handle = self._owner_handle
            if handle is None:
                return
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
                self._owner_handle = None

    def __enter__(self) -> "EventCoordinator":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _acquire_owner(self) -> None:
        self._validate_storage()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / ".events_owner.lock"
        handle = lock_path.open("a+b")
        try:
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RuntimeError(f"{self.output_dir} already has an EventCoordinator owner") from exc
        self._owner_handle = handle

    def poll(self) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_owner()
            return self._poll_locked()

    def _poll_locked(self) -> list[dict[str, Any]]:
        state = self._load_state()
        events = self._load_events()
        event_by_id = {event["event_id"]: event for event in events}
        dirty_event_ids: set[str] = set()
        generation_id, generation_started_at = self._read_generation()

        saved_generation = state.get("generation")
        generation_changed = saved_generation is not None and saved_generation != generation_id
        if generation_changed:
            state = self._empty_state()
        state["generation"] = generation_id
        state["generation_started_at"] = generation_started_at.isoformat(timespec="microseconds")

        if not self.source.exists():
            self._publish(events, state, dirty_event_ids, events_changed=False)
            return events

        with self.source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            initial_evidence = self._fd_identity(handle, before)
            source_changed = self._source_changed(state, handle, before)
            if source_changed and not generation_changed:
                raise SourceGenerationMismatch("frame_events.jsonl changed without a new generation_id")
            if state["offset"] > before.st_size:
                raise SourceGenerationMismatch("frame_events.jsonl truncated without a new generation_id")

            self._after_source_validation(handle)
            read_offset = state["offset"]
            handle.seek(read_offset)
            data = handle.read()
            after = os.fstat(handle.fileno())
            final_evidence = self._fd_identity(handle, after)

            try:
                path_stat = self.source.stat()
            except FileNotFoundError:
                latest_generation, _ = self._read_generation()
                if latest_generation == generation_id:
                    raise SourceGenerationMismatch("frame_events.jsonl disappeared without a new generation_id")
                return events
            complete_bytes = data.rfind(b"\n") + 1
            changed_during_read = self._fd_changed_during_read(
                handle,
                before,
                after,
                initial_evidence,
                final_evidence,
                path_stat,
                read_offset,
                data[:complete_bytes],
            )
            latest_generation, _ = self._read_generation()
            if latest_generation != generation_id:
                return events
            if changed_during_read:
                raise SourceGenerationMismatch("frame_events.jsonl changed during poll without a new generation_id")

            if complete_bytes:
                for raw_line in data[:complete_bytes].splitlines():
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    self._consume(
                        record, state, events, event_by_id, dirty_event_ids
                    )
                state["offset"] += complete_bytes

            state["source"] = final_evidence
            state["checkpoint_tail"] = self._checkpoint_tail(handle, state["offset"])

        self._publish(
            events,
            state,
            dirty_event_ids,
            events_changed=bool(dirty_event_ids),
        )
        return events

    def _after_source_validation(self, _handle: Any) -> None:
        """Test hook at the source validation/read boundary."""

    def _ensure_owner(self) -> None:
        self._validate_storage()
        if self._owner_handle is None:
            raise RuntimeError("EventCoordinator is closed")

    def _validate_storage(self) -> None:
        if self.storage_run_dir is not None:
            if resolve_run_storage(self.storage_run_dir).metadata_dir != self.output_dir.absolute():
                raise ValueError("event metadata authority changed")

    def set_status(self, event_id: str, status: str, *, error: str | None = None) -> dict[str, Any]:
        """Atomically move a collecting event to ``ready`` or ``failed``."""
        with self._lock:
            self._ensure_owner()
            if status not in {"ready", "failed"}:
                raise ValueError("status must be ready or failed")
            events = self._load_events()
            event = next((item for item in events if item.get("event_id") == event_id), None)
            if event is None:
                raise KeyError(event_id)
            if event.get("status") != "collecting":
                raise ValueError("terminal event cannot transition again")
            event["status"] = status
            if error is not None:
                event["error"] = error
            self._publish(events, self._load_state(), set())
            return event

    def set_vlm_filter_result(
        self,
        event_id: str,
        result: str,
        *,
        attempts: int,
        expected_attempts: int,
        request_id: str,
        evidence_revision: str,
        prompt_revision: str,
        model_version: str,
        label: str | None = None,
        reviewed_at: str | None = None,
        latency_seconds: float | None = None,
        retry_at: float | None = None,
        retryable: bool | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Atomically persist one asynchronous large-model review transition."""

        if result not in VLM_FILTER_RESULTS:
            raise ValueError("invalid VLM filter result")
        if (
            type(attempts) is not int
            or type(expected_attempts) is not int
            or attempts < 0
            or expected_attempts < 0
            or attempts > 1000
            or expected_attempts > 1000
        ):
            raise ValueError("invalid VLM filter attempt count")
        if _VLM_REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("invalid VLM filter request id")
        if (
            _VLM_REVISION.fullmatch(evidence_revision) is None
            or _VLM_REVISION.fullmatch(prompt_revision) is None
            or _VLM_REVISION.fullmatch(model_version) is None
        ):
            raise ValueError("invalid VLM filter revision")
        if retry_at is not None and (
            not isinstance(retry_at, (int, float))
            or isinstance(retry_at, bool)
            or not math.isfinite(float(retry_at))
            or float(retry_at) < 0
        ):
            raise ValueError("invalid VLM filter retry time")
        if retryable is not None and type(retryable) is not bool:
            raise ValueError("invalid VLM filter retryable flag")
        if result != "error" and retryable is not None:
            raise ValueError("VLM retryable flag is only valid for errors")
        if latency_seconds is not None and (
            not isinstance(latency_seconds, (int, float))
            or isinstance(latency_seconds, bool)
            or not math.isfinite(float(latency_seconds))
            or not 0 <= float(latency_seconds) <= 3600
        ):
            raise ValueError("invalid VLM filter latency")
        with self._lock:
            self._ensure_owner()
            events = self._load_events()
            event = next((item for item in events if item.get("event_id") == event_id), None)
            if event is None:
                raise KeyError(event_id)
            if event.get("status") != "ready":
                raise ValueError("VLM review requires a ready event")
            current_result = event.get("vlm_filter_result")
            if current_result is not None and current_result not in VLM_FILTER_RESULTS:
                raise ValueError("event has an invalid VLM filter state")
            current_attempts_value = event.get("vlm_filter_attempts", 0)
            current_attempts = (
                current_attempts_value
                if type(current_attempts_value) is int and current_attempts_value >= 0
                else 0
            )
            if current_attempts != expected_attempts:
                raise ValueError("stale VLM filter attempt")
            current_request_id = event.get("vlm_filter_request_id")
            current_evidence_revision = event.get("vlm_filter_evidence_revision")
            current_prompt_revision = event.get("vlm_filter_expected_prompt_revision")
            current_model_version = event.get("vlm_filter_expected_model_version")

            if result == "pending":
                same_request = (
                    current_result == "pending"
                    and current_request_id == request_id
                    and current_evidence_revision == evidence_revision
                    and current_prompt_revision == prompt_revision
                    and current_model_version == model_version
                    and attempts == current_attempts
                )
                if same_request:
                    return event
                revision_changed = (
                    current_result in {"pending", "pass", "filter", "uncertain", "error"}
                    and (
                        current_evidence_revision != evidence_revision
                        or current_prompt_revision != prompt_revision
                        or current_model_version != model_version
                    )
                )
                if current_result == "pending" and not revision_changed:
                    raise ValueError("VLM review is already pending")
                if current_result in {"pass", "filter", "uncertain"} and not revision_changed:
                    raise ValueError("completed VLM review is immutable")
                expected_next_attempt = (
                    1 if current_result is None or revision_changed else current_attempts + 1
                )
                if attempts != expected_next_attempt:
                    raise ValueError("invalid next VLM filter attempt")
                event["vlm_filter_result"] = "pending"
                event["vlm_filter_attempts"] = attempts
                event["vlm_filter_request_id"] = request_id
                event["vlm_filter_evidence_revision"] = evidence_revision
                event["vlm_filter_expected_prompt_revision"] = prompt_revision
                event["vlm_filter_expected_model_version"] = model_version
                event["vlm_filter_updated_at"] = datetime.now(timezone.utc).isoformat()
                for field in (
                    "vlm_filter_label",
                    "vlm_filter_model_version",
                    "vlm_filter_reviewed_at",
                    "vlm_filter_latency_seconds",
                    "vlm_filter_error",
                    "vlm_filter_retry_at",
                    "vlm_filter_retryable",
                ):
                    event.pop(field, None)
                self._publish(events, self._load_state(), set())
                return event

            if (
                current_result != "pending"
                or attempts != current_attempts
                or current_request_id != request_id
                or current_evidence_revision != evidence_revision
                or current_prompt_revision != prompt_revision
                or current_model_version != model_version
            ):
                raise ValueError("stale VLM filter completion")
            event["vlm_filter_result"] = result
            event["vlm_filter_updated_at"] = datetime.now(timezone.utc).isoformat()
            if result in {"pass", "filter", "uncertain"}:
                if label != VLM_LABEL_BY_RESULT[result] or not isinstance(
                    reviewed_at, str
                ):
                    raise ValueError("completed VLM review requires result metadata")
                try:
                    reviewed_time = datetime.fromisoformat(
                        reviewed_at.replace("Z", "+00:00")
                    )
                except ValueError as value_error:
                    raise ValueError("invalid VLM review timestamp") from value_error
                if reviewed_time.tzinfo is None or reviewed_time.utcoffset() is None:
                    raise ValueError("VLM review timestamp must be timezone-aware")
                event["vlm_filter_label"] = label
                event["vlm_filter_model_version"] = model_version
                event["vlm_filter_reviewed_at"] = reviewed_at
                if latency_seconds is not None:
                    event["vlm_filter_latency_seconds"] = float(latency_seconds)
                event.pop("vlm_filter_error", None)
                event.pop("vlm_filter_retry_at", None)
                event.pop("vlm_filter_retryable", None)
            elif result == "error":
                if (
                    not isinstance(error, str)
                    or not error
                    or error != error.strip()
                    or len(error) > 1000
                ):
                    raise ValueError("failed VLM review requires an error")
                if retryable is None:
                    raise ValueError("failed VLM review requires retry metadata")
                if retryable is False and retry_at is not None:
                    raise ValueError("permanent VLM failure cannot have a retry time")
                event["vlm_filter_error"] = error
                event["vlm_filter_retryable"] = retryable
                if retry_at is not None:
                    event["vlm_filter_retry_at"] = float(retry_at)
                else:
                    event.pop("vlm_filter_retry_at", None)
            self._publish(events, self._load_state(), set())
            return event

    def _consume(
        self,
        record: dict[str, Any],
        state: dict[str, Any],
        events: list[dict[str, Any]],
        event_by_id: dict[str, dict[str, Any]],
        dirty_event_ids: set[str],
    ) -> None:
        camera = self._camera(record)
        frame_index = int(self._first(record, "frame_index", "frame_id", default=0))
        time_sec = float(self._first(
            record,
            "time_sec",
            "timestamp",
            "pts_sec",
            default=frame_index / self.infer_fps,
        ))
        captured_at = self._parse_captured_at(record.get("captured_at"))
        captured_at_text = (
            captured_at.isoformat(timespec="microseconds")
            if captured_at is not None
            else None
        )
        people = self._first(record, "persons", "people", default=[])
        if not isinstance(people, list):
            return
        phones = record.get("phones", [])
        if not isinstance(phones, list):
            phones = []

        present_keys = set()
        for person in people:
            if not isinstance(person, dict):
                continue
            raw_id = self._first(person, "person_id", "track_id", "id", default=None)
            if raw_id is not None:
                present_keys.add(self._track_key(camera, str(raw_id)))
        for key, track in list(state["tracks"].items()):
            track_camera, _ = json.loads(key)
            if track_camera == camera and key not in present_keys:
                last_time_sec = track.get("last_time_sec")
                if (
                    isinstance(last_time_sec, (int, float))
                    and time_sec - float(last_time_sec) > EVENT_CONTEXT_SECONDS
                ):
                    del state["tracks"][key]
                else:
                    self._reset_alarm_episode(track, last_frame=frame_index)
        for key, last_alarm in list(state["last_alarm"].items()):
            track_camera, _ = json.loads(key)
            if (
                track_camera == camera
                and time_sec - float(last_alarm) >= self.cooldown_sec
            ):
                del state["last_alarm"][key]

        for person in people:
            if not isinstance(person, dict):
                continue
            raw_id = self._first(person, "person_id", "track_id", "id", default=None)
            if raw_id is None:
                continue
            person_id = str(raw_id)
            key = self._track_key(camera, person_id)
            track = state["tracks"].setdefault(key, self._new_track())
            previous_frame = track.get("last_frame")
            if previous_frame is not None and frame_index != int(previous_frame) + 1:
                self._reset_alarm_episode(track)

            strict_alarm = bool(self._first(person, "stable_alarm", "alarm", default=False))
            if not strict_alarm and str(person.get("state", "")).lower() == "alarm":
                strict_alarm = True
            review_candidate = False
            alarm = strict_alarm or review_candidate
            bbox = self._first(person, "bbox", "box", "raw_box", default=None)
            # DeepStream emits both a boolean risk flag and a numeric risk score.
            # Prefer the numeric value so True is not displayed as a score of 1.00.
            risk = float(self._first(person, "risk_score", "score", "risk", default=0.0))
            track["last_frame"] = frame_index
            track["last_time_sec"] = time_sec
            sample = {
                "time_sec": time_sec,
                "frame_index": frame_index,
                "bbox": bbox,
                "track_id": person_id,
                "risk_score": risk,
                "alarm": alarm,
                "state": str(person.get("state", "")),
            }
            if review_candidate:
                sample["strict_alarm"] = strict_alarm
                sample["review_candidate"] = True
            # Persist the decision evidence with the clip overlay.  The
            # dashboard can then explain an alarm from its own immutable event
            # artifact instead of consulting mutable live process state.
            for field in (
                "screen_id",
                "gated_reject_reason",
                "gated_handheld_review_hits",
                "handheld_phone_hits",
                "handheld_phone_stable_count",
                "window_hits",
                "legacy_window_hits",
                "static_suppressed",
                "static_pending",
                "static_context_reason",
                "static_exit_reason",
            ):
                value = person.get(field)
                if isinstance(value, (str, int, float, bool)):
                    sample[field] = value
            person_roi = person.get("roi")
            if isinstance(person_roi, list) and len(person_roi) == 4:
                sample["roi"] = person_roi
                if isinstance(record.get("width"), int):
                    sample["frame_width"] = record["width"]
                if isinstance(record.get("height"), int):
                    sample["frame_height"] = record["height"]
            matched_phones = []
            for phone in phones:
                if not isinstance(phone, dict):
                    continue
                phone_track_id = self._first(phone, "track_id", "person_id", default=None)
                phone_box = self._first(phone, "bbox", "box", default=None)
                if str(phone_track_id) != person_id or not isinstance(phone_box, list):
                    continue
                matched_phones.append(
                    {
                        "box": phone_box,
                        "confidence": phone.get("confidence"),
                        "phone_score": phone.get("phone_score"),
                        "accepted": phone.get("accepted"),
                    }
                )
            if matched_phones:
                sample["phone_boxes"] = matched_phones
            if captured_at_text is not None:
                sample["captured_at"] = captured_at_text
            if bbox is not None:
                self._append_context(track, sample, time_sec)

            current_id = track.get("event_id")
            if current_id and current_id in event_by_id:
                current_event = event_by_id[current_id]
                capture_until = (
                    float(current_event.get("event_stream_time_sec", time_sec))
                    + EVENT_CONTEXT_SECONDS
                )
                if current_event.get("status") == "collecting" and time_sec <= capture_until:
                    if strict_alarm:
                        self._promote_strict_alarm(current_event)
                    self._update_event(current_event, sample)
                    dirty_event_ids.add(str(current_event["event_id"]))
                    if not alarm:
                        self._reset_alarm_episode(track, last_frame=frame_index)
                    continue
                track["event_id"] = None
                track["episode_emitted"] = False

            if not alarm:
                self._reset_alarm_episode(track, last_frame=frame_index)
                continue

            track["consecutive"] += 1
            track["risk_peak"] = max(float(track["risk_peak"]), risk)
            if bbox is not None:
                track["timeline"].append(sample)
            # A gated review candidate already represents a multi-frame policy
            # decision.  Requiring the coordinator's additional two-frame
            # debounce can discard a valid one-frame policy pulse.  Strict
            # alarms keep the original debounce unchanged.
            required_stable_frames = (
                1 if review_candidate and not strict_alarm else self.stable_frames
            )
            if track["consecutive"] < required_stable_frames or track["episode_emitted"]:
                continue

            last_alarm = state["last_alarm"].get(key)
            if last_alarm is not None and time_sec - float(last_alarm) < self.cooldown_sec:
                track["episode_emitted"] = True
                continue

            # Track IDs can change even when the same person has not moved.
            # Reuse the existing per-person cooldown when the latest alarm box
            # substantially overlaps a recent event from this camera.  This
            # suppresses duplicate review items without suppressing a second
            # person elsewhere in the same view.
            if self._recent_camera_event_has_same_subject(
                events,
                camera=camera,
                generation=str(state["generation"]),
                time_sec=time_sec,
                bbox=bbox,
                person_id=person_id,
            ):
                track["episode_emitted"] = True
                state["last_alarm"][key] = time_sec
                continue

            generation = str(state["generation"])
            generation_started_at = datetime.fromisoformat(state["generation_started_at"])
            occurred_at = captured_at or generation_started_at + timedelta(seconds=time_sec)
            event_id = self._event_id(camera, person_id, generation, occurred_at)
            existing = event_by_id.get(event_id)
            if existing is None:
                existing = self._shared_camera_event(
                    events,
                    camera=camera,
                    generation=generation,
                    time_sec=time_sec,
                )
                if existing is not None:
                    event_id = str(existing["event_id"])
            if existing is None:
                existing = {
                    "event_id": event_id,
                    "camera": camera,
                    "person_id": person_id,
                    "person_ids": [],
                    "alarm_track_count": 0,
                    "source_generation": generation,
                    "generation_started_at": generation_started_at.isoformat(timespec="microseconds"),
                    "status": "collecting",
                    "level": "alarm" if strict_alarm else "review",
                    "detector_state": "S4_ALARM" if strict_alarm else "REVIEW",
                    "detector_trigger": "strict_alarm" if strict_alarm else "handheld_review",
                    "occurred_at": occurred_at.isoformat(timespec="microseconds"),
                    "event_stream_time_sec": time_sec,
                    "clip_post_seconds": POST_EVENT_SECONDS,
                    "stable_alarm_time_sec": time_sec,
                    "first_frame_index": frame_index,
                    "last_frame_index": frame_index,
                    "risk_peak": 0.0,
                    "bbox_timeline": [],
                }
                events.append(existing)
                event_by_id[event_id] = existing
            elif strict_alarm:
                self._promote_strict_alarm(existing)
            self._add_event_person(existing, person_id)
            for context_sample in track["context_timeline"]:
                self._update_event(existing, context_sample)
            dirty_event_ids.add(event_id)
            track["event_id"] = event_id
            track["episode_emitted"] = True
            state["last_alarm"][key] = time_sec

    @staticmethod
    def _promote_strict_alarm(event: dict[str, Any]) -> None:
        if event.get("detector_trigger") != "strict_alarm":
            event["level"] = "alarm"
            event["detector_state"] = "S4_ALARM"
            event["detector_trigger"] = "strict_alarm"

    @staticmethod
    def _shared_camera_event(
        events: list[dict[str, Any]],
        *,
        camera: str,
        generation: str,
        time_sec: float,
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            if (
                event.get("camera") != camera
                or event.get("source_generation") != generation
                or event.get("status") != "collecting"
            ):
                continue
            anchor = event.get("event_stream_time_sec")
            if not isinstance(anchor, (int, float)):
                continue
            delta = time_sec - float(anchor)
            if 0.0 <= delta <= EVENT_GROUP_WINDOW_SECONDS:
                return event
        return None

    def _recent_camera_event_has_same_subject(
        self,
        events: list[dict[str, Any]],
        *,
        camera: str,
        generation: str,
        time_sec: float,
        bbox: Any,
        person_id: str,
    ) -> bool:
        """Return true when a tracker-ID replacement matches a recent subject."""
        current_box = self._normalized_box(bbox)
        if current_box is None:
            return False
        for event in reversed(events):
            if (
                event.get("camera") != camera
                or event.get("source_generation") != generation
            ):
                continue
            timeline = event.get("bbox_timeline")
            if not isinstance(timeline, list):
                continue
            for sample in reversed(timeline):
                if not isinstance(sample, dict):
                    continue
                # A coordinator restart may replay the same track before its
                # state checkpoint is restored.  That must update the original
                # event rather than being treated as a tracker-ID replacement.
                if str(sample.get("track_id", "")) == person_id:
                    continue
                sample_time = sample.get("time_sec")
                if not isinstance(sample_time, (int, float)):
                    continue
                delta = time_sec - float(sample_time)
                if delta < 0:
                    continue
                if delta >= self.cooldown_sec:
                    break
                previous_box = self._normalized_box(sample.get("bbox"))
                if previous_box is not None and self._box_iou(current_box, previous_box) >= _REIDENTIFICATION_IOU_THRESHOLD:
                    return True
        return False

    @staticmethod
    def _normalized_box(value: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            left, top, width, height = (float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return left, top, left + width, top + height

    @staticmethod
    def _box_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        overlap_left = max(first[0], second[0])
        overlap_top = max(first[1], second[1])
        overlap_right = min(first[2], second[2])
        overlap_bottom = min(first[3], second[3])
        overlap_width = max(0.0, overlap_right - overlap_left)
        overlap_height = max(0.0, overlap_bottom - overlap_top)
        overlap_area = overlap_width * overlap_height
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - overlap_area
        return overlap_area / union if union > 0 else 0.0

    @staticmethod
    def _add_event_person(event: dict[str, Any], person_id: str) -> None:
        raw_ids = event.get("person_ids")
        person_ids = []
        if isinstance(raw_ids, list):
            person_ids.extend(str(value) for value in raw_ids)
        primary = event.get("person_id")
        if primary is not None and str(primary) not in person_ids:
            person_ids.insert(0, str(primary))
        if person_id not in person_ids:
            person_ids.append(person_id)
        event["person_ids"] = person_ids
        event["alarm_track_count"] = len(person_ids)

    @staticmethod
    def _update_event(event: dict[str, Any], sample: dict[str, Any]) -> None:
        time_sec = float(sample["time_sec"])
        frame_index = sample["frame_index"]
        bbox = sample.get("bbox")
        track_id = str(sample.get("track_id", ""))
        risk = float(sample.get("risk_score", 0.0))
        event["risk_peak"] = max(float(event.get("risk_peak", 0.0)), risk)
        event["first_frame_index"] = min(
            int(event.get("first_frame_index", frame_index)), int(frame_index)
        )
        event["last_frame_index"] = max(
            int(event.get("last_frame_index", frame_index)), int(frame_index)
        )
        if bbox is not None:
            point = dict(sample)
            key = (int(frame_index), float(time_sec), track_id)
            for index, existing in enumerate(event["bbox_timeline"]):
                existing_key = (
                    int(existing["frame_index"]),
                    float(existing["time_sec"]),
                    str(existing.get("track_id", "")),
                )
                if existing_key == key:
                    event["bbox_timeline"][index] = point
                    break
            else:
                event["bbox_timeline"].append(point)
                event["bbox_timeline"].sort(
                    key=lambda item: (
                        int(item["frame_index"]),
                        float(item["time_sec"]),
                        str(item.get("track_id", "")),
                    )
                )

    def _publish(
        self,
        events: list[dict[str, Any]],
        state: dict[str, Any],
        dirty_event_ids: set[str] | None = None,
        *,
        events_changed: bool = True,
    ) -> None:
        self._validate_storage()
        # Even a no-event first poll must publish an archive before its checkpoint.
        # Otherwise the next missing archive cannot be distinguished from data loss.
        if events_changed or not self._archive_initialized:
            self._atomic_json(self.events_path, events)
            self._archive_initialized = True
        events_to_publish = (
            events
            if dirty_event_ids is None
            else [
                event
                for event in events
                if str(event.get("event_id", "")) in dirty_event_ids
            ]
        )
        for event in events_to_publish:
            overlay_path = self.output_dir / "events" / event["event_id"] / "overlay.json"
            self._atomic_json(overlay_path, {
                "event_id": event["event_id"],
                "camera": event["camera"],
                "person_id": event["person_id"],
                "person_ids": event.get("person_ids", [event["person_id"]]),
                "alarm_track_count": int(event.get("alarm_track_count", 1)),
                "source_generation": event["source_generation"],
                "generation_started_at": event["generation_started_at"],
                "occurred_at": event["occurred_at"],
                "event_stream_time_sec": event["event_stream_time_sec"],
                "risk_peak": event["risk_peak"],
                "bbox_timeline": event["bbox_timeline"],
            })
        checkpoint_bytes = (json.dumps(state, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
        # Only coalesce a checkpoint this owner has successfully persisted.
        # Never skip a failed save or trust an in-memory value after removal.
        if checkpoint_bytes != self._saved_checkpoint_bytes or not self.state_path.exists():
            self._atomic_json(self.state_path, state)
            self._saved_checkpoint_bytes = checkpoint_bytes

    def _source_changed(self, state: dict[str, Any], handle: Any, stat: os.stat_result) -> bool:
        saved = state.get("source")
        if not isinstance(saved, dict):
            return False
        if saved.get("dev") != stat.st_dev or saved.get("ino") != stat.st_ino:
            return True
        current = self._fd_identity(handle, stat)
        if saved.get("first_line_sha256") != current["first_line_sha256"]:
            checkpoint = state.get("checkpoint_tail")
            source_was_unconsumed = (
                int(state.get("offset", 0)) == 0
                and isinstance(checkpoint, dict)
                and int(checkpoint.get("length", 0)) == 0
            )
            if not source_was_unconsumed:
                return True
        checkpoint = state.get("checkpoint_tail")
        if not isinstance(checkpoint, dict):
            return False
        start = int(checkpoint.get("start", 0))
        length = int(checkpoint.get("length", 0))
        if start < 0 or length < 0 or start + length > stat.st_size:
            return True
        handle.seek(start)
        content = handle.read(length)
        return hashlib.sha256(content).hexdigest() != checkpoint.get("sha256")

    def _fd_identity(self, handle: Any, stat: os.stat_result) -> dict[str, Any]:
        handle.seek(0)
        first_line = handle.readline()
        return {
            "dev": stat.st_dev,
            "ino": stat.st_ino,
            "first_line_sha256": hashlib.sha256(first_line).hexdigest(),
        }

    def _checkpoint_tail(self, handle: Any, offset: int) -> dict[str, Any]:
        start = max(0, offset - self.PREFIX_BYTES)
        length = offset - start
        handle.seek(start)
        content = handle.read(length)
        return {"start": start, "length": length, "sha256": hashlib.sha256(content).hexdigest()}

    @staticmethod
    def _fd_changed_during_read(
        handle: Any,
        before: os.stat_result,
        after: os.stat_result,
        initial: dict[str, Any],
        final: dict[str, Any],
        path_stat: os.stat_result,
        read_offset: int,
        checkpoint_bytes: bytes,
    ) -> bool:
        path_identity = (path_stat.st_dev, path_stat.st_ino)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            return True
        if (after.st_dev, after.st_ino) != path_identity:
            return True
        if initial["first_line_sha256"] != final["first_line_sha256"]:
            empty_source_started_growing = before.st_size == 0 and read_offset == 0
            if not empty_source_started_growing:
                return True
        handle.seek(read_offset)
        return handle.read(len(checkpoint_bytes)) != checkpoint_bytes

    def _load_state(self) -> dict[str, Any]:
        loaded = self._read_json(self.state_path, None)
        if not isinstance(loaded, dict):
            return self._empty_state()
        loaded.setdefault("offset", 0)
        loaded.setdefault("tracks", {})
        loaded.setdefault("last_alarm", {})
        loaded.setdefault("source", None)
        loaded.setdefault("generation", None)
        loaded.setdefault("generation_started_at", None)
        loaded.setdefault("checkpoint_tail", None)
        return loaded

    def _read_generation(self) -> tuple[str, datetime]:
        metadata = self._read_json(self.generation_file, None)
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid producer generation metadata: {self.generation_file}")
        generation_id = metadata.get("generation_id")
        started_at_raw = metadata.get("started_at")
        generation_id = _validate_generation_id(generation_id)
        if not isinstance(started_at_raw, str):
            raise ValueError("producer generation started_at must be a timezone-aware ISO datetime")
        try:
            started_at = datetime.fromisoformat(started_at_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("producer generation started_at must be a timezone-aware ISO datetime") from exc
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("producer generation started_at must be timezone-aware")
        return generation_id, started_at

    def _load_events(self) -> list[dict[str, Any]]:
        recovery = (
            f"event archive {self.events_path} is unavailable or invalid; "
            "publication stopped. Restore the verified current archive consistent "
            "with its checkpoint, preserving newer status/review annotations; "
            "do not reset the checkpoint or substitute an empty archive."
        )
        try:
            content = self.events_path.read_text(encoding="utf-8")
            self._archive_initialized = True
            loaded = json.loads(content)
        except FileNotFoundError as exc:
            # lstat distinguishes a genuinely absent file from a dangling symlink,
            # and propagates storage errors instead of treating them as absence.
            if self._archive_initialized:
                raise EventArchiveError(recovery) from exc
            try:
                for path in (self.events_path, self.state_path, self.output_dir / "events"):
                    try:
                        path.lstat()
                    except FileNotFoundError:
                        continue
                    self._archive_initialized = True
                    raise EventArchiveError(recovery) from exc
            except OSError as stat_error:
                self._archive_initialized = True
                raise EventArchiveError(recovery) from stat_error
            return []
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self._archive_initialized = True
            raise EventArchiveError(recovery) from exc

        if not isinstance(loaded, list):
            raise EventArchiveError(recovery)
        seen: set[str] = set()
        for event in loaded:
            if not isinstance(event, dict):
                raise EventArchiveError(recovery)
            event_id = event.get("event_id")
            if (
                not isinstance(event_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", event_id) is None
                or event_id in seen
                or event.get("status") not in ("collecting", "ready", "failed")
                or any(
                    not isinstance(event.get(field), str) or not event[field]
                    for field in ("camera", "person_id", "source_generation",
                                  "generation_started_at", "occurred_at")
                )
                or not isinstance(event.get("bbox_timeline"), list)
                or any(not isinstance(sample, dict) for sample in event["bbox_timeline"])
                or any(
                    not isinstance(event.get(field), (int, float))
                    or isinstance(event[field], bool)
                    or not math.isfinite(event[field])
                    for field in ("event_stream_time_sec", "risk_peak")
                )
            ):
                raise EventArchiveError(recovery)
            seen.add(event_id)
        self._archive_initialized = True
        return loaded

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "offset": 0,
            "tracks": {},
            "last_alarm": {},
            "source": None,
            "generation": None,
            "generation_started_at": None,
            "checkpoint_tail": None,
        }

    @staticmethod
    def _new_track() -> dict[str, Any]:
        return {
            "consecutive": 0,
            "timeline": [],
            "context_timeline": [],
            "risk_peak": 0.0,
            "event_id": None,
            "episode_emitted": False,
            "last_frame": None,
            "last_time_sec": None,
        }

    @staticmethod
    def _append_context(track: dict[str, Any], sample: dict[str, Any], time_sec: float) -> None:
        context = track.setdefault("context_timeline", [])
        context.append(dict(sample))
        cutoff = time_sec - EVENT_CONTEXT_SECONDS
        track["context_timeline"] = [
            point for point in context if float(point.get("time_sec", time_sec)) >= cutoff
        ]

    @staticmethod
    def _reset_alarm_episode(track: dict[str, Any], *, last_frame: int | None = None) -> None:
        track["consecutive"] = 0
        track["timeline"] = []
        track["risk_peak"] = 0.0
        if not track.get("event_id"):
            track["episode_emitted"] = False
        if last_frame is not None:
            track["last_frame"] = last_frame

    @classmethod
    def _reset_track(cls, track: dict[str, Any], *, last_frame: int | None = None) -> None:
        track.update(cls._new_track())
        track["last_frame"] = last_frame

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _first(mapping: dict[str, Any], *names: str, default: Any) -> Any:
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
        return default

    def _camera(self, record: dict[str, Any]) -> str:
        explicit = EventCoordinator._first(record, "camera", "camera_id", default=None)
        if explicit is not None:
            return str(explicit)
        stream_index = record.get("stream_index")
        if isinstance(stream_index, int) and stream_index in self.stream_cameras:
            return self.stream_cameras[stream_index]
        if isinstance(stream_index, int) and 0 <= stream_index <= 6:
            return f"camera{stream_index + 1:02d}"
        return str(EventCoordinator._first(record, "stream_index", "input_video", default="unknown"))

    @staticmethod
    def _track_key(camera: str, person_id: str) -> str:
        return json.dumps([camera, person_id], separators=(",", ":"))

    @staticmethod
    def _parse_captured_at(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            captured_at = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if captured_at.tzinfo is None or captured_at.utcoffset() is None:
            return None
        return captured_at

    def _event_id(
        self,
        camera: str,
        person_id: str,
        generation: str,
        occurred_at: datetime,
    ) -> str:
        clean = lambda value: re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"
        occurred_ms = int(round(occurred_at.timestamp() * 1000.0))
        return f"camera-{clean(camera)}_person-{clean(person_id)}_source-{generation}_alarm-{occurred_ms}"
