"""Frozen historical datasets for offline screen-capture evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


_SCHEMA_VERSION = 1
_LABELS = frozenset(
    {
        "CAPTURE_POSSIBLE",
        "SUSPECTED_CAPTURE",
        "VIEWING_PHONE",
        "FLAT_OR_DOWN",
        "AWAY_FROM_SCREEN",
        "BLOCKED",
        "NOT_PHONE",
    }
)
_FIRST_STAGE_RESULTS = frozenset({"pass", "filter", "uncertain"})


@dataclass(frozen=True)
class ManifestEntry:
    event_id: str
    camera: str
    source_run_id: str
    source_id: str
    clip_path: str
    overlay_path: str
    review_path: str
    clip_size: int
    clip_sha256: str
    overlay_sha256: str
    review_sha256: str
    first_stage_result: str
    prompt_revision: str | None
    evidence_revision: str | None
    model_version: str | None


@dataclass(frozen=True)
class FrozenManifest:
    schema_version: int
    entries: tuple[ManifestEntry, ...]
    digest: str


@dataclass(frozen=True)
class LabeledEvent:
    entry: ManifestEntry
    ground_truth: str
    reason: str


@dataclass(frozen=True)
class DatasetSplit:
    tuning: tuple[LabeledEvent, ...]
    evaluation: tuple[LabeledEvent, ...]


def build_manifest(
    review_pool: Path,
    runs_root: Path,
    output: Path,
    enabled_cameras: frozenset[str],
    per_camera_limit: int,
) -> FrozenManifest:
    review_pool = Path(review_pool)
    runs_root = Path(runs_root)
    output = Path(output)
    if type(per_camera_limit) is not int or per_camera_limit <= 0:
        raise ValueError("per_camera_limit must be a positive integer")
    if not enabled_cameras or any(
        not isinstance(camera, str) or not camera for camera in enabled_cameras
    ):
        raise ValueError("enabled_cameras must contain non-empty strings")

    event_dirs = _reviewed_event_directories(review_pool)
    source_cache: dict[str, dict[str, dict[str, Any]]] = {}
    candidates: list[tuple[str, str, str, Path, dict[str, Any], dict[str, Any]]] = []
    seen_event_ids: set[str] = set()

    for event_dir in event_dirs:
        review_path = event_dir / "review.json"
        clip_path = event_dir / "event.mp4"
        overlay_path = event_dir / "overlay.json"
        if not all(path.exists() for path in (review_path, clip_path, overlay_path)):
            continue
        if any(path.is_symlink() or not path.is_file() for path in (review_path, clip_path, overlay_path)):
            raise ValueError(f"reviewed event contains a symlink or non-file: {event_dir}")
        review = _read_object(review_path, "review")
        overlay = _read_object(overlay_path, "overlay")
        event_id = _required_string(review, "event_id")
        camera = _required_string(review, "camera")
        source_run_id = _required_string(review, "source_run_id")
        reviewed_at = _required_string(review, "reviewed_at")
        if camera not in enabled_cameras:
            continue
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate event_id: {event_id}")
        seen_event_ids.add(event_id)
        candidates.append(
            (camera, reviewed_at, event_id, event_dir, review, overlay)
        )

    selected: list[tuple[str, str, str, Path, dict[str, Any], dict[str, Any]]] = []
    camera_counts: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda item: item[:3]):
        camera = candidate[0]
        if camera_counts.get(camera, 0) >= per_camera_limit:
            continue
        camera_counts[camera] = camera_counts.get(camera, 0) + 1
        selected.append(candidate)

    entries: list[ManifestEntry] = []
    for camera, _, event_id, event_dir, review, overlay in selected:
        source_run_id = str(review["source_run_id"])
        source_events = source_cache.get(source_run_id)
        if source_events is None:
            source_events = _load_source_events(runs_root, source_run_id)
            source_cache[source_run_id] = source_events
        source_event = source_events.get(event_id)
        first_result, prompt_revision, evidence_revision, model_version = (
            _first_stage_metadata(source_event)
        )
        explicit_source = overlay.get("source_id")
        source_id = (
            explicit_source.strip()
            if isinstance(explicit_source, str) and explicit_source.strip()
            else f"{source_run_id}:{event_id}"
        )
        clip_path = (event_dir / "event.mp4").resolve()
        overlay_path = (event_dir / "overlay.json").resolve()
        review_path = (event_dir / "review.json").resolve()
        entries.append(
            ManifestEntry(
                event_id=event_id,
                camera=camera,
                source_run_id=source_run_id,
                source_id=source_id,
                clip_path=str(clip_path),
                overlay_path=str(overlay_path),
                review_path=str(review_path),
                clip_size=clip_path.stat().st_size,
                clip_sha256=_sha256_file(clip_path),
                overlay_sha256=_sha256_file(overlay_path),
                review_sha256=_sha256_file(review_path),
                first_stage_result=first_result,
                prompt_revision=prompt_revision,
                evidence_revision=evidence_revision,
                model_version=model_version,
            )
        )

    canonical_entries = [asdict(entry) for entry in entries]
    digest = hashlib.sha256(_canonical_json(canonical_entries)).hexdigest()
    manifest = FrozenManifest(
        schema_version=_SCHEMA_VERSION,
        entries=tuple(entries),
        digest=digest,
    )
    _atomic_private_json(
        output,
        {
            "schema_version": manifest.schema_version,
            "digest": manifest.digest,
            "entries": canonical_entries,
        },
    )
    return manifest


def validate_labels(
    manifest: FrozenManifest, labels_path: Path
) -> tuple[LabeledEvent, ...]:
    body = _read_object(Path(labels_path), "labels")
    if body.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("invalid labels schema_version")
    if body.get("manifest_digest") != manifest.digest:
        raise ValueError("labels manifest digest does not match")
    labels = body.get("labels")
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    entries = {entry.event_id: entry for entry in manifest.entries}
    labeled: list[LabeledEvent] = []
    seen: set[str] = set()
    for index, label_record in enumerate(labels):
        if not isinstance(label_record, dict):
            raise ValueError(f"label at index {index} must be an object")
        event_id = _required_string(label_record, "event_id")
        if event_id not in entries:
            raise ValueError(f"unknown event in labels: {event_id}")
        if event_id in seen:
            raise ValueError(f"duplicate label for event: {event_id}")
        seen.add(event_id)
        ground_truth = label_record.get("label")
        if ground_truth not in _LABELS:
            raise ValueError(f"invalid label for event: {event_id}")
        reason = label_record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"label reason is required for event: {event_id}")
        labeled.append(
            LabeledEvent(
                entry=entries[event_id],
                ground_truth=ground_truth,
                reason=reason.strip(),
            )
        )
    missing = set(entries) - seen
    if missing:
        raise ValueError(f"missing labels for events: {', '.join(sorted(missing))}")
    return tuple(sorted(labeled, key=lambda event: event.entry.event_id))


def split_by_source(
    events: tuple[LabeledEvent, ...], seed: str
) -> DatasetSplit:
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    tuning: list[LabeledEvent] = []
    evaluation: list[LabeledEvent] = []
    for event in sorted(events, key=lambda item: item.entry.event_id):
        bucket = hashlib.sha256(
            f"{seed}\0{event.entry.source_id}".encode("utf-8")
        ).digest()[0]
        (tuning if bucket < 64 else evaluation).append(event)
    return DatasetSplit(tuning=tuple(tuning), evaluation=tuple(evaluation))


def _reviewed_event_directories(review_pool: Path) -> tuple[Path, ...]:
    if review_pool.is_symlink():
        raise ValueError("review pool must not be a symlink")
    if not review_pool.is_dir():
        raise ValueError(f"review pool does not exist: {review_pool}")
    event_dirs: list[Path] = []
    for root, directories, files in os.walk(review_pool, followlinks=False):
        root_path = Path(root)
        for directory in directories:
            if (root_path / directory).is_symlink():
                raise ValueError(f"review pool contains a symlink: {root_path / directory}")
        if {"review.json", "event.mp4", "overlay.json"}.issubset(files):
            event_dirs.append(root_path)
    return tuple(sorted(event_dirs))


def _load_source_events(runs_root: Path, source_run_id: str) -> dict[str, dict[str, Any]]:
    events_path = runs_root / source_run_id / "dashboard" / "events.json"
    if events_path.is_symlink() or not events_path.is_file():
        return {}
    try:
        body = json.loads(events_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(body, list):
        return {}
    events: dict[str, dict[str, Any]] = {}
    for item in body:
        if not isinstance(item, dict):
            continue
        event_id = item.get("event_id")
        if isinstance(event_id, str) and event_id and event_id not in events:
            events[event_id] = item
    return events


def _first_stage_metadata(
    source_event: dict[str, Any] | None,
) -> tuple[str, str | None, str | None, str | None]:
    if source_event is None or source_event.get("vlm_filter_result") not in _FIRST_STAGE_RESULTS:
        return "uncertain", None, None, None
    return (
        str(source_event["vlm_filter_result"]),
        _optional_string(source_event.get("vlm_filter_expected_prompt_revision")),
        _optional_string(source_event.get("vlm_filter_evidence_revision")),
        _optional_string(source_event.get("vlm_filter_model_version")),
    )


def _required_string(body: dict[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} JSON: {path}") from error
    if not isinstance(body, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return body


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(body: object) -> bytes:
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_private_json(path: Path, body: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(_canonical_json(body))
            target.write(b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
