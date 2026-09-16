"""Resumable, production-guarded offline screen-capture evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import hmac
from html import escape
import io
import json
import os
from pathlib import Path
import ssl
import stat
import tempfile
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
import zipfile

from live_operator.capture_dataset import (
    FrozenManifest,
    LabeledEvent,
    ManifestEntry,
    build_manifest,
    validate_labels,
)
from live_operator.capture_evidence import CAPTURE_EVIDENCE_REVISION
from live_operator.capture_policy import StageTwoLabel, aggregate_capture_decision
from live_operator.capture_visibility import (
    VisibilityRelation,
    evaluate_visibility,
    load_visibility_sidecar,
)
from live_operator.mage_vl_service import (
    CAPTURE_PROMPT_REVISION,
    capture_request_id,
)
from live_operator.vlm_review import (
    load_shared_secret,
    request_signature,
    response_signature,
)


_POSITIVE_LABELS = frozenset({"CAPTURE_POSSIBLE", "SUSPECTED_CAPTURE"})
_MAX_RESPONSE_BYTES = 256 * 1024


@dataclass(frozen=True)
class RunResult:
    state: str
    reason: str
    event_id: str | None = None


class CaptureRequestError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CaptureReviewClient:
    def __init__(
        self,
        *,
        endpoint: str,
        shared_secret_file: Path,
        expected_model_version: str,
        tls_ca_file: Path | None = None,
        timeout_seconds: float = 120.0,
        max_request_bytes: int = 64 * 1024 * 1024,
        opener: Callable[..., Any] = urlopen,
        clock: Callable[[], float] = time.time,
    ) -> None:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path.rstrip("/") != "/v1/capture-review"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("capture endpoint must be an http(s) /v1/capture-review URL")
        if parsed.scheme == "http" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("remote capture endpoint must use HTTPS")
        if parsed.scheme == "https" and tls_ca_file is None:
            raise ValueError("HTTPS capture endpoint requires a CA file")
        self.endpoint = endpoint
        self.expected_model_version = expected_model_version
        self.timeout_seconds = float(timeout_seconds)
        self.max_request_bytes = int(max_request_bytes)
        self._secret = load_shared_secret(shared_secret_file)
        self._opener = opener
        self._clock = clock
        self._ssl_context = (
            ssl.create_default_context(cafile=str(tls_ca_file))
            if parsed.scheme == "https"
            else None
        )

    def health(self) -> Mapping[str, Any]:
        parsed = urlsplit(self.endpoint)
        url = urlunsplit((parsed.scheme, parsed.netloc, "/healthz", "", ""))
        timestamp = str(int(self._clock()))
        signature = request_signature(self._secret, timestamp, b"")
        request = Request(
            url,
            method="GET",
            headers={
                "X-Jiankong-Timestamp": timestamp,
                "X-Jiankong-Signature": signature,
            },
        )
        payload = self._open_json(request, signature)
        if (
            payload.get("ready") is not True
            or payload.get("model_version") != self.expected_model_version
            or payload.get("capture_prompt_revision") != CAPTURE_PROMPT_REVISION
            or payload.get("capture_evidence_revision") != CAPTURE_EVIDENCE_REVISION
        ):
            raise CaptureRequestError("vlm_unhealthy")
        return payload

    def send(self, entry: ManifestEntry, visibility_path: Path) -> Mapping[str, Any]:
        clip_path = Path(entry.clip_path)
        overlay_path = Path(entry.overlay_path)
        visibility_path = Path(visibility_path)
        request_id = capture_request_id(
            entry.event_id,
            clip_path,
            overlay_path,
            visibility_path,
            model_version=self.expected_model_version,
            prompt_revision=CAPTURE_PROMPT_REVISION,
            evidence_revision=CAPTURE_EVIDENCE_REVISION,
        )
        metadata = {
            "schema_version": 1,
            "event_id": entry.event_id,
            "request_id": request_id,
            "prompt_revision": CAPTURE_PROMPT_REVISION,
            "evidence_revision": CAPTURE_EVIDENCE_REVISION,
        }
        source_bytes = {
            logical_name: self._read_regular(path)
            for logical_name, path in (
                ("clip.mp4", clip_path),
                ("overlay.json", overlay_path),
                ("visibility.json", visibility_path),
            )
        }
        if sum(len(content) for content in source_bytes.values()) + 4096 > self.max_request_bytes:
            raise CaptureRequestError("capture_request_too_large")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                "request.json",
                json.dumps(metadata, ensure_ascii=True, separators=(",", ":")),
            )
            for logical_name, content in source_bytes.items():
                archive.writestr(logical_name, content)
        body = buffer.getvalue()
        if not body or len(body) > self.max_request_bytes:
            raise CaptureRequestError("capture_request_too_large")
        timestamp = str(int(self._clock()))
        signature = request_signature(self._secret, timestamp, body)
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/zip",
                "Content-Length": str(len(body)),
                "X-Jiankong-Timestamp": timestamp,
                "X-Jiankong-Signature": signature,
            },
        )
        payload = self._open_json(request, signature)
        if (
            payload.get("event_id") != entry.event_id
            or payload.get("request_id") != request_id
            or payload.get("model_version") != self.expected_model_version
            or payload.get("prompt_revision") != CAPTURE_PROMPT_REVISION
            or payload.get("evidence_revision") != CAPTURE_EVIDENCE_REVISION
        ):
            raise CaptureRequestError("invalid_capture_response")
        return payload

    def _read_regular(self, path: Path) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise CaptureRequestError("invalid_capture_input") from error
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
                raise CaptureRequestError("invalid_capture_input")
            if details.st_size > self.max_request_bytes:
                raise CaptureRequestError("capture_request_too_large")
            chunks = []
            remaining = details.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise CaptureRequestError("invalid_capture_input")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _open_json(self, request: Request, request_digest: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self._ssl_context is not None:
            kwargs["context"] = self._ssl_context
        try:
            with self._opener(request, **kwargs) as response:
                if int(getattr(response, "status", 200)) != 200:
                    raise CaptureRequestError("capture_service_busy")
                body = response.read(_MAX_RESPONSE_BYTES + 1)
                signed = response.headers.get("X-Jiankong-Response-Signature")
        except HTTPError as error:
            if error.code == 503:
                raise CaptureRequestError("production_busy") from error
            raise CaptureRequestError("capture_http_error") from error
        except (OSError, TimeoutError, URLError) as error:
            raise CaptureRequestError("capture_timeout") from error
        if len(body) > _MAX_RESPONSE_BYTES:
            raise CaptureRequestError("invalid_capture_response")
        expected = response_signature(self._secret, request_digest, body)
        if not isinstance(signed, str) or not hmac.compare_digest(expected, signed):
            raise CaptureRequestError("invalid_capture_signature")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaptureRequestError("invalid_capture_response") from error
        if not isinstance(payload, dict):
            raise CaptureRequestError("invalid_capture_response")
        return payload


class OfflineEvaluator:
    def __init__(
        self,
        *,
        events: tuple[LabeledEvent, ...],
        visibility_dir: Path,
        output_dir: Path,
        manifest_digest: str,
        baseline_fps: float,
        expected_camera_count: int,
        status_fetcher: Callable[[], Mapping[str, Any]],
        mage_health_fetcher: Callable[[], Mapping[str, Any]],
        capture_sender: Callable[[ManifestEntry, Path], Mapping[str, Any]],
        clock: Callable[[], float] = time.time,
        min_request_interval_seconds: float = 30.0,
    ) -> None:
        self.events = tuple(events)
        self.visibility_dir = Path(visibility_dir)
        self.output_dir = Path(output_dir)
        self.manifest_digest = manifest_digest
        self.baseline_fps = float(baseline_fps)
        self.expected_camera_count = int(expected_camera_count)
        self.status_fetcher = status_fetcher
        self.mage_health_fetcher = mage_health_fetcher
        self.capture_sender = capture_sender
        self.clock = clock
        self.min_request_interval_seconds = float(min_request_interval_seconds)
        if (
            self.baseline_fps <= 0
            or self.expected_camera_count <= 0
            or self.min_request_interval_seconds < 30.0
        ):
            raise ValueError("invalid frozen foreground baseline")
        if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
            raise ValueError("invalid manifest digest")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.output_dir.is_symlink():
            raise ValueError("output directory must not be a symlink")
        if os.name == "posix":
            os.chmod(self.output_dir, 0o700)
        self.results_path = self.output_dir / "results.jsonl"
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self._ensure_private_append_file(self.results_path)
        checkpoint = self._load_checkpoint()
        self._completed = set(checkpoint.get("completed_event_ids", []))
        self._completed.update(self._completed_from_results())
        last_request = checkpoint.get("last_request_at")
        self._last_request_at = (
            float(last_request) if isinstance(last_request, (int, float)) else None
        )
        self._write_checkpoint()

    def run_once(self) -> RunResult:
        pending = next(
            (event for event in self.events if event.entry.event_id not in self._completed),
            None,
        )
        if pending is None:
            return RunResult("done", "manifest_complete")

        health_reason = self._health_breach()
        if health_reason is not None:
            self._append({"record_type": "pause", "reason": health_reason})
            return RunResult("paused", health_reason)

        now = float(self.clock())
        if (
            self._last_request_at is not None
            and now - self._last_request_at < self.min_request_interval_seconds
        ):
            return RunResult("rate_limited", "offline_rate_limit", pending.entry.event_id)

        visibility_path = self.visibility_dir / f"{pending.entry.camera}.json"
        try:
            config = load_visibility_sidecar(visibility_path)
            relation = self._visibility_relation(pending.entry, config)
        except (OSError, ValueError, json.JSONDecodeError):
            return self._input_error(pending, "invalid_visibility_sidecar")

        self._last_request_at = now
        self._write_checkpoint()
        try:
            response = self.capture_sender(pending.entry, visibility_path)
        except CaptureRequestError as error:
            self._append_pause(error.reason, pending.entry.event_id)
            return RunResult("paused", error.reason, pending.entry.event_id)
        except (OSError, TimeoutError, URLError):
            self._append_pause("capture_timeout", pending.entry.event_id)
            return RunResult("paused", "capture_timeout", pending.entry.event_id)

        if response.get("cancelled") is True:
            self._append_pause("capture_cancelled", pending.entry.event_id)
            return RunResult("paused", "capture_cancelled", pending.entry.event_id)
        raw_label = response.get("label")
        try:
            stage_two = StageTwoLabel(raw_label)
        except (TypeError, ValueError):
            self._append_pause("invalid_capture_response", pending.entry.event_id)
            return RunResult("paused", "invalid_capture_response", pending.entry.event_id)

        decision = aggregate_capture_decision(
            pending.entry.first_stage_result, relation, stage_two
        )
        record = {
            "record_type": "completed",
            "event_id": pending.entry.event_id,
            "camera": pending.entry.camera,
            "source_id": pending.entry.source_id,
            "baseline_result": self._baseline_result(pending.entry.first_stage_result),
            "first_stage_result": pending.entry.first_stage_result,
            "stage_two_label": stage_two.value,
            "visibility": relation.state,
            "visibility_zone_id": relation.zone_id,
            "visibility_occluder_id": relation.occluder_id,
            "final_outcome": decision.outcome.value,
            "decision_reason": decision.reason,
            "ground_truth": pending.ground_truth,
            "ground_truth_reason": pending.reason,
            "latency_seconds": response.get("latency_seconds"),
            "evidence_complete": response.get("evidence_complete"),
            "candidate_labels": response.get("candidate_labels", []),
        }
        self._append(record)
        self._completed.add(pending.entry.event_id)
        self._write_checkpoint()
        return RunResult("completed", "event_completed", pending.entry.event_id)

    def write_report(self) -> Path:
        records = self._records()
        completed = [record for record in records if record.get("record_type") == "completed"]
        pauses = [record for record in records if record.get("record_type") == "pause"]
        metrics = {
            "schema_version": 1,
            "manifest_digest": self.manifest_digest,
            "complete": len(self._completed) == len(self.events),
            "aggregate": self._metrics_for(completed),
            "per_camera": {
                camera: self._metrics_for(
                    [record for record in completed if record.get("camera") == camera]
                )
                for camera in sorted(
                    {str(record.get("camera")) for record in completed}
                )
            },
            "pause_count": len(pauses),
            "cancel_count": sum(
                record.get("reason") == "capture_cancelled" for record in pauses
            ),
            "uncertain_count": sum(
                record.get("stage_two_label") == "UNCERTAIN" for record in completed
            ),
        }
        _atomic_private_json(self.output_dir / "metrics.json", metrics)
        rows = "".join(
            "<tr>"
            + "".join(
                f"<td>{escape(str(record.get(key, '')))}</td>"
                for key in (
                    "camera",
                    "baseline_result",
                    "stage_two_label",
                    "visibility",
                    "final_outcome",
                    "ground_truth",
                )
            )
            + "</tr>"
            for record in completed
        )
        html = (
            "<!doctype html><meta charset='utf-8'><title>Capture offline report</title>"
            "<h1>Capture offline report</h1>"
            f"<pre>{escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre>"
            "<table><thead><tr><th>Camera</th><th>Baseline</th><th>Stage 2</th>"
            "<th>Visibility</th><th>Final</th><th>Ground truth</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        report_path = self.output_dir / "report.html"
        _atomic_private_bytes(report_path, html.encode("utf-8"))
        return report_path

    def _health_breach(self) -> str | None:
        try:
            foreground = self.status_fetcher()
        except Exception:
            return "foreground_unreachable"
        if not isinstance(foreground, Mapping) or foreground.get("state") != "running":
            return "foreground_unhealthy"
        cameras = foreground.get("cameras")
        if not isinstance(cameras, list) or len(cameras) != self.expected_camera_count:
            return "camera_loss"
        if any(
            not isinstance(camera, Mapping) or camera.get("status") != "online"
            for camera in cameras
        ):
            return "camera_loss"
        fps = foreground.get("aggregate_fps")
        if (
            not isinstance(fps, (int, float))
            or isinstance(fps, bool)
            or float(fps) < self.baseline_fps * 0.95
        ):
            return "fps_drop"
        if foreground.get("alerts") not in (None, []):
            return "foreground_alerts"
        try:
            mage = self.mage_health_fetcher()
        except Exception:
            return "vlm_unhealthy"
        if not isinstance(mage, Mapping) or mage.get("ready") is not True:
            return "vlm_unhealthy"
        resources = mage.get("resource_health", {})
        if isinstance(resources, Mapping) and resources.get("memory_pressure") is True:
            return "memory_pressure"
        scheduler = mage.get("scheduler")
        if not isinstance(scheduler, Mapping):
            return "vlm_unhealthy"
        if scheduler.get("active_kind") is not None or int(
            scheduler.get("production_waiting", 0) or 0
        ) > 0:
            return "production_busy"
        quiet = scheduler.get("quiet_remaining_seconds")
        if not isinstance(quiet, (int, float)) or float(quiet) > 0:
            return "production_quiet_period"
        return None

    def _visibility_relation(self, entry: ManifestEntry, config: Any) -> VisibilityRelation:
        overlay = json.loads(Path(entry.overlay_path).read_text(encoding="utf-8"))
        root = overlay.get("overlay", overlay) if isinstance(overlay, dict) else {}
        timeline = root.get("bbox_timeline", []) if isinstance(root, dict) else []
        sample = next(
            (
                item
                for item in timeline
                if isinstance(item, dict)
                and item.get("alarm") is True
                and isinstance(item.get("screen_id"), str)
            ),
            None,
        )
        if sample is None:
            return VisibilityRelation("unknown", "unknown", None, None)
        screen_id = str(sample["screen_id"])
        raw_box = sample.get("roi", sample.get("bbox"))
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            return VisibilityRelation(screen_id, "unknown", None, None)
        try:
            left, _top, right, bottom = (float(value) for value in raw_box)
            source_width = float(sample.get("frame_width", config.frame_width))
            source_height = float(sample.get("frame_height", config.frame_height))
            anchor = (
                (left + right) / 2 * config.frame_width / source_width,
                bottom * config.frame_height / source_height,
            )
            return evaluate_visibility(config, screen_id=screen_id, anchor=anchor)
        except (TypeError, ValueError, ZeroDivisionError):
            return VisibilityRelation(screen_id, "unknown", None, None)

    @staticmethod
    def _baseline_result(first_stage: str) -> str:
        if first_stage == "filter":
            return "FILTER_NOT_PHONE"
        return "KEEP_SUSPECTED_CAPTURE"

    def _input_error(self, event: LabeledEvent, reason: str) -> RunResult:
        self._append(
            {
                "record_type": "input_error",
                "event_id": event.entry.event_id,
                "camera": event.entry.camera,
                "reason": reason,
            }
        )
        self._completed.add(event.entry.event_id)
        self._write_checkpoint()
        return RunResult("input_error", reason, event.entry.event_id)

    def _append_pause(self, reason: str, event_id: str) -> None:
        self._append({"record_type": "pause", "reason": reason, "event_id": event_id})

    def _append(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        payload["observed_at"] = float(self.clock())
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.results_path, flags)
        try:
            os.write(
                descriptor,
                (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _records(self) -> list[dict[str, Any]]:
        records = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _completed_from_results(self) -> set[str]:
        return {
            str(record["event_id"])
            for record in self._records()
            if record.get("record_type") in {"completed", "input_error"}
            and isinstance(record.get("event_id"), str)
        }

    def _load_checkpoint(self) -> dict[str, Any]:
        try:
            body = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(body, dict) or body.get("manifest_digest") != self.manifest_digest:
            raise ValueError("checkpoint manifest digest mismatch")
        return body

    def _write_checkpoint(self) -> None:
        _atomic_private_json(
            self.checkpoint_path,
            {
                "schema_version": 1,
                "manifest_digest": self.manifest_digest,
                "completed_event_ids": sorted(self._completed),
                "last_request_at": self._last_request_at,
            },
        )

    @staticmethod
    def _ensure_private_append_file(path: Path) -> None:
        if path.is_symlink():
            raise ValueError("results path must not be a symlink")
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)

    @staticmethod
    def _metrics_for(records: list[Mapping[str, Any]]) -> dict[str, Any]:
        positive = [record for record in records if record.get("ground_truth") in _POSITIVE_LABELS]
        negative = [record for record in records if record.get("ground_truth") not in _POSITIVE_LABELS]
        retained_positive = sum(
            str(record.get("final_outcome", "")).startswith("KEEP_") for record in positive
        )
        filtered_negative = sum(
            str(record.get("final_outcome", "")).startswith("FILTER_") for record in negative
        )
        latencies = [
            float(record["latency_seconds"])
            for record in records
            if isinstance(record.get("latency_seconds"), (int, float))
        ]
        return {
            "event_count": len(records),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "confusion": {
                "retained_positive": retained_positive,
                "filtered_positive": len(positive) - retained_positive,
                "filtered_negative": filtered_negative,
                "retained_negative": len(negative) - filtered_negative,
            },
            "positive_retention": (
                retained_positive / len(positive) if positive else None
            ),
            "false_positive_reduction": (
                filtered_negative / len(negative) if negative else None
            ),
            "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        }


def _atomic_private_json(path: Path, body: object) -> None:
    _atomic_private_bytes(
        path,
        (json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_private_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _load_manifest(path: Path) -> FrozenManifest:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid frozen manifest") from error
    if not isinstance(body, dict) or body.get("schema_version") != 1:
        raise ValueError("invalid frozen manifest schema")
    raw_entries = body.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("invalid frozen manifest entries")
    try:
        entries = tuple(ManifestEntry(**entry) for entry in raw_entries)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid frozen manifest entry") from error
    canonical = json.dumps(
        [asdict(entry) for entry in entries],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if body.get("digest") != digest:
        raise ValueError("frozen manifest digest mismatch")
    return FrozenManifest(schema_version=1, entries=entries, digest=digest)


def _fetch_json(url: str, *, timeout: float = 5.0) -> Mapping[str, Any]:
    request = Request(url, method="GET", headers={"Cache-Control": "no-store"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if int(getattr(response, "status", 200)) != 200:
                raise RuntimeError("foreground status is unavailable")
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except (HTTPError, OSError, TimeoutError, URLError) as error:
        raise RuntimeError("foreground status is unavailable") from error
    if len(body) > _MAX_RESPONSE_BYTES:
        raise RuntimeError("foreground status is too large")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("foreground status is invalid")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jiankong-capture-offline")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-manifest")
    build.add_argument("--review-pool", type=Path, required=True)
    build.add_argument("--runs-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--camera", action="append", required=True)
    build.add_argument("--per-camera-limit", type=int, default=10)

    labels = commands.add_parser("validate-labels")
    labels.add_argument("--manifest", type=Path, required=True)
    labels.add_argument("--labels", type=Path, required=True)

    def add_dataset_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--labels", type=Path, required=True)
        command.add_argument("--visibility-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--baseline-fps", type=float, required=True)
        command.add_argument("--expected-camera-count", type=int, default=8)

    dry_run = commands.add_parser("dry-run")
    add_dataset_arguments(dry_run)

    run = commands.add_parser("run")
    add_dataset_arguments(run)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--shared-secret-file", type=Path, required=True)
    run.add_argument("--expected-model-version", required=True)
    run.add_argument("--foreground-status-url", required=True)
    run.add_argument("--tls-ca-file", type=Path)
    run.add_argument("--max-offline-requests-per-minute", type=int, choices=(1, 2), default=2)

    report = commands.add_parser("report")
    add_dataset_arguments(report)
    return parser


def _loaded_dataset(args: argparse.Namespace) -> tuple[FrozenManifest, tuple[LabeledEvent, ...]]:
    manifest = _load_manifest(args.manifest)
    return manifest, validate_labels(manifest, args.labels)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-manifest":
        manifest = build_manifest(
            args.review_pool,
            args.runs_root,
            args.output,
            frozenset(args.camera),
            args.per_camera_limit,
        )
        print(manifest.digest)
        return 0
    if args.command == "validate-labels":
        manifest = _load_manifest(args.manifest)
        events = validate_labels(manifest, args.labels)
        print(len(events))
        return 0

    manifest, events = _loaded_dataset(args)
    for camera in sorted({event.entry.camera for event in events}):
        load_visibility_sidecar(args.visibility_dir / f"{camera}.json")
    if args.command == "dry-run":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"manifest_digest": manifest.digest, "events": len(events)}))
        return 0

    if args.command == "run":
        client = CaptureReviewClient(
            endpoint=args.endpoint,
            shared_secret_file=args.shared_secret_file,
            expected_model_version=args.expected_model_version,
            tls_ca_file=args.tls_ca_file,
        )
        evaluator = OfflineEvaluator(
            events=events,
            visibility_dir=args.visibility_dir,
            output_dir=args.output_dir,
            manifest_digest=manifest.digest,
            baseline_fps=args.baseline_fps,
            expected_camera_count=args.expected_camera_count,
            status_fetcher=lambda: _fetch_json(args.foreground_status_url),
            mage_health_fetcher=client.health,
            capture_sender=client.send,
            min_request_interval_seconds=60.0 / args.max_offline_requests_per_minute,
        )
        while True:
            result = evaluator.run_once()
            print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
            if result.state in {"done", "paused"}:
                return 0 if result.state == "done" else 75
            if result.state == "rate_limited":
                time.sleep(1.0)

    evaluator = OfflineEvaluator(
        events=events,
        visibility_dir=args.visibility_dir,
        output_dir=args.output_dir,
        manifest_digest=manifest.digest,
        baseline_fps=args.baseline_fps,
        expected_camera_count=args.expected_camera_count,
        status_fetcher=lambda: {},
        mage_health_fetcher=lambda: {},
        capture_sender=lambda _entry, _path: {},
    )
    print(evaluator.write_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
