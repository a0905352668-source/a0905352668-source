"""Fenced operator requeue for retryable Mage-VL HTTP 500 overlays."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import uuid
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from live_operator.storage_paths import resolve_run_storage
from live_operator.vlm_review import VLMReviewConfig
from live_operator.vlm_state import (
    DEFAULT_VLM_STATE_FILENAME,
    VLMReviewStateStore,
    _MAX_STATE_BYTES,
    _SAFE_EVENT_ID,
    VLM_STATE_FIELDS,
)


HTTP_500_ERROR = "VLM review service returned HTTP 500"


class VLMRequeueError(RuntimeError):
    """The requeue operation could not prove its required safety fence."""


@dataclass(frozen=True)
class RequeuePlan:
    """An immutable dry-run snapshot that must still match at apply time."""

    run_dir: Path
    metadata_dir: Path
    sidecar_path: Path
    source_sha256: str
    candidate_event_ids: tuple[str, ...]
    candidate_ids_sha256: str
    expected_model_version: str
    expected_prompt_revision: str
    expected_evidence_revision: str
    error_text: str
    config: VLMReviewConfig = field(repr=False, compare=False)


@dataclass(frozen=True)
class RequeueResult:
    """The durable results of applying one validated requeue plan."""

    removed_event_ids: tuple[str, ...]
    backup_sha256: str

    @property
    def removed_count(self) -> int:
        return len(self.removed_event_ids)


def _read_sidecar(path: Path) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Read one stable sidecar image using the state store's validation rules."""

    try:
        before = path.lstat()
    except OSError as error:
        raise VLMRequeueError("VLM state is unavailable") from error
    if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_STATE_BYTES:
        raise VLMRequeueError("VLM state is invalid")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VLMRequeueError("VLM state is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size > _MAX_STATE_BYTES
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise VLMRequeueError("VLM state is invalid")
        pieces: list[bytes] = []
        remaining = int(opened.st_size)
        while remaining:
            piece = os.read(descriptor, min(65536, remaining))
            if not piece:
                raise VLMRequeueError("VLM state is invalid")
            pieces.append(piece)
            remaining -= len(piece)
        source = b"".join(pieces)
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VLMRequeueError("VLM state is invalid") from error
    if not isinstance(payload, dict):
        raise VLMRequeueError("VLM state is invalid")
    states: dict[str, dict[str, Any]] = {}
    for event_id, value in payload.items():
        if (
            not isinstance(event_id, str)
            or _SAFE_EVENT_ID.fullmatch(event_id) is None
            or not isinstance(value, dict)
            or any(field not in VLM_STATE_FIELDS for field in value)
        ):
            raise VLMRequeueError("VLM state is invalid")
        states[event_id] = dict(value)
    return source, states


def _candidate_ids(
    states: Mapping[str, Mapping[str, Any]], config: VLMReviewConfig
) -> tuple[str, ...]:
    return tuple(
        sorted(
            event_id
            for event_id, state in states.items()
            if state.get("vlm_filter_result") == "error"
            and state.get("vlm_filter_error") == HTTP_500_ERROR
            and state.get("vlm_filter_retryable") is True
            and state.get("vlm_filter_expected_model_version")
            == config.expected_model_version
            and state.get("vlm_filter_expected_prompt_revision")
            == config.expected_prompt_revision
            and state.get("vlm_filter_evidence_revision")
            == config.expected_evidence_revision
        )
    )


def _candidate_ids_sha256(candidate_event_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(candidate_event_ids).encode("utf-8")).hexdigest()


def plan_http500_requeue(run_dir: Path, config: VLMReviewConfig) -> RequeuePlan:
    """Return a non-mutating, hash-fenced plan for HTTP-500 overlay removal."""

    run_path = Path(run_dir).absolute()
    metadata_dir = resolve_run_storage(run_path).metadata_dir
    sidecar_path = metadata_dir / DEFAULT_VLM_STATE_FILENAME
    source, states = _read_sidecar(sidecar_path)
    candidate_event_ids = _candidate_ids(states, config)
    return RequeuePlan(
        run_dir=run_path,
        metadata_dir=metadata_dir.absolute(),
        sidecar_path=sidecar_path.absolute(),
        source_sha256=hashlib.sha256(source).hexdigest(),
        candidate_event_ids=candidate_event_ids,
        candidate_ids_sha256=_candidate_ids_sha256(candidate_event_ids),
        expected_model_version=config.expected_model_version,
        expected_prompt_revision=config.expected_prompt_revision,
        expected_evidence_revision=config.expected_evidence_revision,
        error_text=HTTP_500_ERROR,
        config=config,
    )


def _validate_plan(plan: RequeuePlan) -> None:
    """Reject any plan that cannot still identify its exact dry-run intent."""

    storage = resolve_run_storage(plan.run_dir)
    if storage.metadata_dir.absolute() != plan.metadata_dir.absolute():
        raise VLMRequeueError("VLM metadata authority changed")
    if plan.sidecar_path.absolute() != (
        plan.metadata_dir / DEFAULT_VLM_STATE_FILENAME
    ).absolute():
        raise VLMRequeueError("VLM sidecar authority changed")
    if (
        plan.expected_model_version != plan.config.expected_model_version
        or plan.expected_prompt_revision != plan.config.expected_prompt_revision
        or plan.expected_evidence_revision != plan.config.expected_evidence_revision
    ):
        raise VLMRequeueError("VLM configuration revisions changed")
    if plan.error_text != HTTP_500_ERROR:
        raise VLMRequeueError("VLM error predicate changed")
    if plan.candidate_event_ids != tuple(sorted(set(plan.candidate_event_ids))):
        raise VLMRequeueError("VLM selected set is invalid")
    if plan.candidate_ids_sha256 != _candidate_ids_sha256(plan.candidate_event_ids):
        raise VLMRequeueError("VLM selected-set fingerprint changed")


def _write_backup(backup_path: Path, source: bytes) -> str:
    """Create one private, exclusive byte-for-byte source backup."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(backup_path, flags, 0o600)
    except OSError as error:
        raise VLMRequeueError("backup path must be new and unavailable to symlinks") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise VLMRequeueError("backup must be a regular file")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        view = memoryview(source)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short backup write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise VLMRequeueError("backup could not be written") from error
    finally:
        os.close(descriptor)
    return hashlib.sha256(source).hexdigest()


def _write_remaining_states(path: Path, states: Mapping[str, Mapping[str, Any]]) -> None:
    """Atomically publish the compact state file, then persist its directory entry."""

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            json.dump(states, handle, ensure_ascii=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def apply_http500_requeue(plan: RequeuePlan, backup_path: Path) -> RequeueResult:
    """Remove exactly the still-matching overlays under the state-store lock."""

    _validate_plan(plan)
    store = VLMReviewStateStore(plan.sidecar_path, run_dir=plan.run_dir)
    with store._exclusive():
        _validate_plan(plan)
        source, states = _read_sidecar(plan.sidecar_path)
        source_sha256 = hashlib.sha256(source).hexdigest()
        if source_sha256 != plan.source_sha256:
            raise VLMRequeueError("VLM requeue source changed after dry run")
        selected = _candidate_ids(states, plan.config)
        if (
            selected != plan.candidate_event_ids
            or _candidate_ids_sha256(selected) != plan.candidate_ids_sha256
        ):
            raise VLMRequeueError("VLM selected set changed after dry run")
        backup_sha256 = _write_backup(Path(backup_path), source)
        selected_set = set(selected)
        remaining = {
            event_id: state
            for event_id, state in states.items()
            if event_id not in selected_set
        }
        _write_remaining_states(plan.sidecar_path, remaining)
        return RequeueResult(
            removed_event_ids=selected,
            backup_sha256=backup_sha256,
        )


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description="fenced Mage-VL HTTP-500 requeue")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--backup-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Emit an auditable dry-run JSON object or apply an explicit matching plan."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.apply and (
        arguments.expected_source_sha256 is None or arguments.backup_path is None
    ):
        parser.error("--apply requires --expected-source-sha256 and --backup-path")
    if not arguments.apply and (
        arguments.expected_source_sha256 is not None or arguments.backup_path is not None
    ):
        parser.error("--expected-source-sha256 and --backup-path require --apply")
    try:
        plan = plan_http500_requeue(arguments.run_dir, VLMReviewConfig.load(arguments.config))
        if not arguments.apply:
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "source_sha256": plan.source_sha256,
                        "candidate_count": len(plan.candidate_event_ids),
                        "candidate_ids_sha256": plan.candidate_ids_sha256,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.expected_source_sha256 != plan.source_sha256:
            raise VLMRequeueError("expected source SHA256 does not match dry run")
        result = apply_http500_requeue(plan, arguments.backup_path)
        print(
            json.dumps(
                {
                    "mode": "apply",
                    "removed_count": result.removed_count,
                    "backup_sha256": result.backup_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, VLMRequeueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
