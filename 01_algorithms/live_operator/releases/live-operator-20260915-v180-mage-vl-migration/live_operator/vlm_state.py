"""Compact durable state for asynchronous large-model event reviews.

The frame-level event archive grows quickly. VLM transitions are small and
frequent, so they live in a separate atomic sidecar instead of rewriting the
whole archive for every pending/completed state change.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from live_operator.vlm_review import VLM_FILTER_RESULTS, VLM_LABEL_BY_RESULT
from live_operator.storage_paths import resolve_run_storage


DEFAULT_VLM_STATE_FILENAME = "vlm_filter_states.json"
_MAX_STATE_BYTES = 64 * 1024 * 1024
_SAFE_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_VLM_REQUEST_ID = re.compile(r"[A-Fa-f0-9]{64}\Z")
_VLM_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

VLM_STATE_FIELDS = frozenset(
    {
        "vlm_filter_result",
        "vlm_filter_attempts",
        "vlm_filter_request_id",
        "vlm_filter_retryable",
        "vlm_filter_retry_at",
        "vlm_filter_error",
        "vlm_filter_expected_model_version",
        "vlm_filter_expected_prompt_revision",
        "vlm_filter_evidence_revision",
        "vlm_filter_label",
        "vlm_filter_model_version",
        "vlm_filter_reviewed_at",
        "vlm_filter_latency_seconds",
        "vlm_filter_updated_at",
    }
)


class VLMStateError(RuntimeError):
    """The compact state sidecar could not be safely read or updated."""


def _transition(
    event: Mapping[str, Any],
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
    updated_at: str,
) -> dict[str, Any]:
    """Apply the existing request-fenced transition to a copied event."""

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
    if any(
        _VLM_REVISION.fullmatch(value) is None
        for value in (evidence_revision, prompt_revision, model_version)
    ):
        raise ValueError("invalid VLM filter revision")
    if retry_at is not None and (
        isinstance(retry_at, bool)
        or not isinstance(retry_at, (int, float))
        or not math.isfinite(float(retry_at))
        or float(retry_at) < 0
    ):
        raise ValueError("invalid VLM filter retry time")
    if retryable is not None and type(retryable) is not bool:
        raise ValueError("invalid VLM filter retryable flag")
    if result != "error" and retryable is not None:
        raise ValueError("VLM retryable flag is only valid for errors")
    if latency_seconds is not None and (
        isinstance(latency_seconds, bool)
        or not isinstance(latency_seconds, (int, float))
        or not math.isfinite(float(latency_seconds))
        or not 0 <= float(latency_seconds) <= 3600
    ):
        raise ValueError("invalid VLM filter latency")

    updated = dict(event)
    if updated.get("status") != "ready":
        raise ValueError("VLM review requires a ready event")
    current_result = updated.get("vlm_filter_result")
    if current_result is not None and current_result not in VLM_FILTER_RESULTS:
        raise ValueError("event has an invalid VLM filter state")
    attempts_value = updated.get("vlm_filter_attempts", 0)
    current_attempts = (
        attempts_value if type(attempts_value) is int and attempts_value >= 0 else 0
    )
    if current_attempts != expected_attempts:
        raise ValueError("stale VLM filter attempt")
    current_request_id = updated.get("vlm_filter_request_id")
    current_evidence_revision = updated.get("vlm_filter_evidence_revision")
    current_prompt_revision = updated.get("vlm_filter_expected_prompt_revision")
    current_model_version = updated.get("vlm_filter_expected_model_version")

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
            return updated
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
        expected_next = 1 if current_result is None or revision_changed else current_attempts + 1
        if attempts != expected_next:
            raise ValueError("invalid next VLM filter attempt")
        updated.update(
            {
                "vlm_filter_result": "pending",
                "vlm_filter_attempts": attempts,
                "vlm_filter_request_id": request_id,
                "vlm_filter_evidence_revision": evidence_revision,
                "vlm_filter_expected_prompt_revision": prompt_revision,
                "vlm_filter_expected_model_version": model_version,
                "vlm_filter_updated_at": updated_at,
            }
        )
        for field in (
            "vlm_filter_label",
            "vlm_filter_model_version",
            "vlm_filter_reviewed_at",
            "vlm_filter_latency_seconds",
            "vlm_filter_error",
            "vlm_filter_retry_at",
            "vlm_filter_retryable",
        ):
            updated.pop(field, None)
        return updated

    if (
        current_result != "pending"
        or attempts != current_attempts
        or current_request_id != request_id
        or current_evidence_revision != evidence_revision
        or current_prompt_revision != prompt_revision
        or current_model_version != model_version
    ):
        raise ValueError("stale VLM filter completion")
    updated["vlm_filter_result"] = result
    updated["vlm_filter_updated_at"] = updated_at
    if result in {"pass", "filter", "uncertain"}:
        if label != VLM_LABEL_BY_RESULT[result] or not isinstance(reviewed_at, str):
            raise ValueError("completed VLM review requires result metadata")
        try:
            reviewed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as value_error:
            raise ValueError("invalid VLM review timestamp") from value_error
        if reviewed_time.tzinfo is None or reviewed_time.utcoffset() is None:
            raise ValueError("VLM review timestamp must be timezone-aware")
        updated["vlm_filter_label"] = label
        updated["vlm_filter_model_version"] = model_version
        updated["vlm_filter_reviewed_at"] = reviewed_at
        if latency_seconds is not None:
            updated["vlm_filter_latency_seconds"] = float(latency_seconds)
        updated.pop("vlm_filter_error", None)
        updated.pop("vlm_filter_retry_at", None)
        updated.pop("vlm_filter_retryable", None)
    else:
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
        updated["vlm_filter_error"] = error
        updated["vlm_filter_retryable"] = retryable
        if retry_at is not None:
            updated["vlm_filter_retry_at"] = float(retry_at)
        else:
            updated.pop("vlm_filter_retry_at", None)
    return updated


class VLMReviewStateStore:
    """Store small per-event VLM overlays with atomic replacement and locking."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.storage_run_dir = Path(run_dir) if run_dir is not None else None
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._thread_lock = threading.RLock()

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        self._validate_storage()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            with self.lock_path.open("a+b") as handle:
                if os.name != "nt":
                    # chmod dirties inode metadata even when the mode is
                    # unchanged. Reads must not generate journal writes.
                    if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) != 0o600:
                        os.fchmod(handle.fileno(), 0o600)
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    self._validate_storage()
                    yield
                finally:
                    if os.name != "nt":
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _validate_storage(self) -> None:
        if self.storage_run_dir is not None:
            if resolve_run_storage(self.storage_run_dir).metadata_dir != self.path.absolute().parent:
                raise ValueError("VLM metadata authority changed")

    def read(self) -> dict[str, dict[str, Any]]:
        self._validate_storage()
        try:
            self.path.lstat()
        except FileNotFoundError:
            # Legacy history may have no dashboard at all and be read-only.
            # An absent optional sidecar needs neither a directory nor a lock.
            self._validate_storage()
            return {}
        with self._exclusive():
            return self._read_unlocked()

    def merge_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        states = self.read()
        return [
            self._merge_event(event, states.get(str(event.get("event_id", ""))))
            for event in events
        ]

    def set_vlm_filter_result(
        self,
        event: Mapping[str, Any],
        result: str,
        **transition: Any,
    ) -> dict[str, Any]:
        event_id = str(event.get("event_id", ""))
        if _SAFE_EVENT_ID.fullmatch(event_id) is None:
            raise ValueError("invalid VLM event id")
        with self._exclusive():
            states = self._read_unlocked()
            current = self._merge_event(event, states.get(event_id))
            updated = _transition(
                current,
                result,
                updated_at=self._clock().isoformat(),
                **transition,
            )
            states[event_id] = {
                field: updated[field] for field in VLM_STATE_FIELDS if field in updated
            }
            self._write_unlocked(states)
            return updated

    @staticmethod
    def _merge_event(
        event: Mapping[str, Any], state: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        merged = dict(event)
        if state is None:
            return merged
        for field in VLM_STATE_FIELDS:
            merged.pop(field, None)
        merged.update(state)
        return merged

    def _read_unlocked(self) -> dict[str, dict[str, Any]]:
        try:
            details = self.path.lstat()
        except FileNotFoundError:
            return {}
        except OSError as error:
            raise VLMStateError("VLM state is unavailable") from error
        if not stat.S_ISREG(details.st_mode) or details.st_size > _MAX_STATE_BYTES:
            raise VLMStateError("VLM state is invalid")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VLMStateError("VLM state is invalid") from error
        if not isinstance(payload, dict):
            raise VLMStateError("VLM state is invalid")
        states: dict[str, dict[str, Any]] = {}
        for event_id, value in payload.items():
            if (
                not isinstance(event_id, str)
                or _SAFE_EVENT_ID.fullmatch(event_id) is None
                or not isinstance(value, dict)
                or any(field not in VLM_STATE_FIELDS for field in value)
            ):
                raise VLMStateError("VLM state is invalid")
            states[event_id] = dict(value)
        return states

    def _write_unlocked(self, states: Mapping[str, Mapping[str, Any]]) -> None:
        self._validate_storage()
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                descriptor = None
                json.dump(states, handle, ensure_ascii=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            if os.name != "nt":
                os.chmod(self.path, 0o600)
                directory = os.open(
                    self.path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
