"""Asynchronous client for the optional Mage-VL event review service.

The DeepStream alarm remains authoritative.  This module only marks an event as
explicitly retained or filtered by the second-stage model; network and model
failures are represented separately so callers can fail open.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import shutil
import ssl
import stat
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_VLM_REVIEW_CONFIG = Path(
    "/media/boshi/Data/JianKong/02_configs/runtime/vlm_review.json"
)
VLM_EVIDENCE_REVISION = (
    "person-roi20-span5s-pre4-native-focus-temporal-early8-jpeg92-v20"
)
VLM_FILTER_RESULTS = frozenset({"pass", "filter", "pending", "uncertain", "error"})
_TERMINAL_SERVICE_RESULTS = frozenset({"pass", "filter", "uncertain"})
VLM_LABEL_BY_RESULT = {
    "pass": "KEEP_NON_CALL_PHONE_USE",
    "filter": "FILTER_FALSE_POSITIVE",
    "uncertain": "UNCERTAIN",
}
_SAFE_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_MODEL_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SAFE_REQUEST_ID = re.compile(r"[A-Fa-f0-9]{64}\Z")
_MAX_RESPONSE_BYTES = 64 * 1024


class _BoundedBytesIO(io.BytesIO):
    """Seekable ZIP target that cannot grow beyond the configured request cap."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = int(limit)
        self._high_water = 0

    def write(self, data: bytes | bytearray | memoryview) -> int:
        end = self.tell() + len(data)
        if max(self._high_water, end) > self._limit:
            raise VLMReviewError("VLM review request exceeds configured size limit")
        written = super().write(data)
        self._high_water = max(self._high_water, self.tell())
        return written


def _regular_file_size(path: Path, *, description: str) -> int:
    """Return the opened file size while rejecting symlinks and special files."""

    try:
        details = path.lstat()
    except OSError as error:
        raise VLMReviewError(f"{description} is unavailable") from error
    if not stat.S_ISREG(details.st_mode):
        raise VLMReviewError(f"{description} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VLMReviewError(f"{description} is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise VLMReviewError(f"{description} must be a regular file")
        return int(opened.st_size)
    finally:
        os.close(descriptor)


class VLMReviewError(RuntimeError):
    """The remote review could not be completed or validated."""

    retryable = True


class VLMReviewPermanentError(VLMReviewError):
    """A request/configuration error that will not heal through retries."""

    retryable = False


def _http_review_error(status: int) -> VLMReviewError:
    message = f"VLM review service returned HTTP {status}"
    if 400 <= status < 500 and status not in {408, 425, 429}:
        return VLMReviewPermanentError(message)
    return VLMReviewError(message)


def _validate_private_file(path: Path, *, description: str) -> None:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{description} must be a regular file")
    if os.name == "posix" and stat.S_IMODE(details.st_mode) != 0o600:
        raise ValueError(f"{description} permissions must be 0600")


def load_shared_secret(path: str | Path) -> bytes:
    """Read a local HMAC key without following a final-component symlink."""

    secret_path = Path(path)
    _validate_private_file(secret_path, description="VLM shared-secret file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(secret_path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("VLM shared-secret file must be a regular file")
        if os.name == "posix" and stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError("VLM shared-secret file permissions must be 0600")
        secret = os.read(descriptor, 4097)
    finally:
        os.close(descriptor)
    secret = secret.strip()
    if len(secret) < 32 or len(secret) > 4096:
        raise ValueError("VLM shared-secret file must contain 32-4096 bytes")
    return secret


@dataclass(frozen=True)
class VLMReviewConfig:
    endpoint: str
    shared_secret_file: Path
    expected_model_version: str
    expected_prompt_revision: str
    expected_evidence_revision: str = VLM_EVIDENCE_REVISION
    review_events_after: datetime | None = None
    tls_ca_file: Path | None = None
    timeout_seconds: float = 120.0
    max_request_bytes: int = 64 * 1024 * 1024
    max_attempts: int = 5
    retry_base_seconds: float = 15.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/v1/review"
        ):
            raise ValueError("VLM endpoint must be an http(s) /v1/review URL without credentials")
        if not self.shared_secret_file.is_absolute():
            raise ValueError("VLM shared-secret path must be absolute")
        if parsed.scheme == "http" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("remote VLM endpoints must use HTTPS")
        if parsed.scheme == "https" and (
            self.tls_ca_file is None or not self.tls_ca_file.is_absolute()
        ):
            raise ValueError("HTTPS VLM endpoint requires an absolute tls_ca_file")
        if _SAFE_MODEL_VERSION.fullmatch(self.expected_model_version) is None:
            raise ValueError("invalid expected VLM model version")
        if _SAFE_MODEL_VERSION.fullmatch(self.expected_prompt_revision) is None:
            raise ValueError("invalid expected VLM prompt revision")
        if _SAFE_MODEL_VERSION.fullmatch(self.expected_evidence_revision) is None:
            raise ValueError("invalid expected VLM evidence revision")
        if self.review_events_after is not None and (
            self.review_events_after.tzinfo is None
            or self.review_events_after.utcoffset() is None
        ):
            raise ValueError("VLM review_events_after must be timezone-aware")
        if not 1.0 <= float(self.timeout_seconds) <= 300.0:
            raise ValueError("VLM timeout_seconds must be between 1 and 300")
        if not 1024 <= int(self.max_request_bytes) <= 256 * 1024 * 1024:
            raise ValueError("VLM max_request_bytes is outside the allowed range")
        if not 1 <= int(self.max_attempts) <= 20:
            raise ValueError("VLM max_attempts must be between 1 and 20")
        if not 1.0 <= float(self.retry_base_seconds) <= 3600.0:
            raise ValueError("VLM retry_base_seconds must be between 1 and 3600")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_VLM_REVIEW_CONFIG) -> "VLMReviewConfig":
        config_path = Path(path)
        _validate_private_file(config_path, description="VLM review configuration")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(config_path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("VLM review configuration must be a regular file")
            if os.name == "posix" and stat.S_IMODE(opened.st_mode) != 0o600:
                raise ValueError("VLM review configuration permissions must be 0600")
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                payload = json.load(handle)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("invalid VLM review configuration schema")
        if payload.get("enabled", True) is not True:
            raise ValueError("VLM review configuration is disabled")
        endpoint = payload.get("endpoint")
        secret_file = payload.get("shared_secret_file")
        model_version = payload.get("expected_model_version")
        prompt_revision = payload.get("expected_prompt_revision")
        evidence_revision = payload.get(
            "expected_evidence_revision", VLM_EVIDENCE_REVISION
        )
        review_events_after_raw = payload.get("review_events_after")
        tls_ca_file_raw = payload.get("tls_ca_file")
        if review_events_after_raw is None:
            review_events_after = None
        elif isinstance(review_events_after_raw, str):
            try:
                review_events_after = datetime.fromisoformat(
                    review_events_after_raw.replace("Z", "+00:00")
                )
            except ValueError as error:
                raise ValueError("invalid VLM review_events_after") from error
        else:
            raise ValueError("invalid VLM review_events_after")
        if tls_ca_file_raw is not None and not isinstance(tls_ca_file_raw, str):
            raise ValueError("invalid VLM tls_ca_file")
        if not all(
            isinstance(value, str)
            for value in (
                endpoint,
                secret_file,
                model_version,
                prompt_revision,
                evidence_revision,
            )
        ):
            raise ValueError(
                "VLM endpoint, shared_secret_file, model, and prompt revisions are required"
            )
        return cls(
            endpoint=endpoint,
            shared_secret_file=Path(secret_file),
            expected_model_version=model_version,
            expected_prompt_revision=prompt_revision,
            expected_evidence_revision=evidence_revision,
            review_events_after=review_events_after,
            tls_ca_file=Path(tls_ca_file_raw) if tls_ca_file_raw is not None else None,
            timeout_seconds=float(payload.get("timeout_seconds", 120.0)),
            max_request_bytes=int(payload.get("max_request_bytes", 64 * 1024 * 1024)),
            max_attempts=int(payload.get("max_attempts", 5)),
            retry_base_seconds=float(payload.get("retry_base_seconds", 15.0)),
        )

    @classmethod
    def load_optional(
        cls, path: str | Path = DEFAULT_VLM_REVIEW_CONFIG
    ) -> "VLMReviewConfig | None":
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None

    def retry_delay_seconds(self, completed_attempts: int) -> float:
        exponent = max(0, min(int(completed_attempts) - 1, 8))
        return min(3600.0, float(self.retry_base_seconds) * (2**exponent))


@dataclass(frozen=True)
class VLMReviewResult:
    event_id: str
    request_id: str
    result: str
    label: str
    model_version: str
    prompt_revision: str
    evidence_revision: str
    reviewed_at: str
    latency_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_id, str)
            or _SAFE_EVENT_ID.fullmatch(self.event_id) is None
        ):
            raise ValueError("invalid VLM event_id")
        if (
            not isinstance(self.request_id, str)
            or _SAFE_REQUEST_ID.fullmatch(self.request_id) is None
        ):
            raise ValueError("invalid VLM request_id")
        if not isinstance(self.result, str) or self.result not in _TERMINAL_SERVICE_RESULTS:
            raise ValueError("invalid VLM review result")
        if not isinstance(self.label, str) or self.label != VLM_LABEL_BY_RESULT[self.result]:
            raise ValueError("VLM result and label disagree")
        if (
            not isinstance(self.model_version, str)
            or _SAFE_MODEL_VERSION.fullmatch(self.model_version) is None
        ):
            raise ValueError("invalid VLM model version")
        if (
            not isinstance(self.prompt_revision, str)
            or _SAFE_MODEL_VERSION.fullmatch(self.prompt_revision) is None
        ):
            raise ValueError("invalid VLM prompt revision")
        if (
            not isinstance(self.evidence_revision, str)
            or _SAFE_MODEL_VERSION.fullmatch(self.evidence_revision) is None
        ):
            raise ValueError("invalid VLM evidence revision")
        if not isinstance(self.reviewed_at, str):
            raise ValueError("invalid VLM reviewed_at")
        try:
            parsed_time = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("invalid VLM reviewed_at") from error
        if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
            raise ValueError("VLM reviewed_at must be timezone-aware")
        if self.latency_seconds is not None:
            if (
                isinstance(self.latency_seconds, bool)
                or not isinstance(self.latency_seconds, (int, float))
                or not math.isfinite(float(self.latency_seconds))
                or not 0 <= float(self.latency_seconds) <= 3600
            ):
                raise ValueError("invalid VLM latency_seconds")


@dataclass(frozen=True)
class VLMPreparedReview:
    """Immutable request body prepared while the previous inference is running."""

    event_id: str
    request_id: str
    body: bytes

    def __post_init__(self) -> None:
        if _SAFE_EVENT_ID.fullmatch(self.event_id) is None:
            raise ValueError("invalid VLM event_id")
        if _SAFE_REQUEST_ID.fullmatch(self.request_id) is None:
            raise ValueError("invalid VLM request_id")
        if not isinstance(self.body, bytes) or not self.body:
            raise ValueError("invalid VLM request body")


def request_signature(secret: bytes, timestamp: str, body: bytes) -> str:
    digest = hashlib.sha256(body).hexdigest()
    message = f"{timestamp}\n{digest}".encode("ascii")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def response_signature(secret: bytes, request_digest: str, body: bytes) -> str:
    response_digest = hashlib.sha256(body).hexdigest()
    message = f"{request_digest}\n{response_digest}".encode("ascii")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def evidence_request_id(
    event_id: str,
    clip_path: str | Path,
    overlay_path: str | Path,
    *,
    model_version: str,
    prompt_revision: str,
    evidence_revision: str,
) -> str:
    """Build a deterministic request fence from immutable event evidence."""

    if _SAFE_EVENT_ID.fullmatch(event_id) is None:
        raise ValueError("invalid VLM event_id")
    if (
        _SAFE_MODEL_VERSION.fullmatch(model_version) is None
        or _SAFE_MODEL_VERSION.fullmatch(prompt_revision) is None
        or _SAFE_MODEL_VERSION.fullmatch(evidence_revision) is None
    ):
        raise ValueError("invalid VLM revision")
    digest = hashlib.sha256()
    for value in (
        "jiankong-vlm-review-v1",
        event_id,
        model_version,
        prompt_revision,
        evidence_revision,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for logical_name, path_value in (
        ("clip.mp4", clip_path),
        ("overlay.json", overlay_path),
    ):
        path = Path(path_value)
        digest.update(logical_name.encode("ascii"))
        digest.update(b"\0")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise VLMReviewError("VLM review input is unavailable") from error
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise VLMReviewError("VLM review input must be a regular file")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return digest.hexdigest()


class VLMReviewClient:
    """Upload one browser clip and its overlay to the review service."""

    def __init__(
        self,
        config: VLMReviewConfig,
        *,
        opener: Callable[..., Any] = urlopen,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._opener = opener
        self._clock = clock
        self._secret = load_shared_secret(config.shared_secret_file)
        self._ssl_context = None
        if urlsplit(config.endpoint).scheme == "https":
            assert config.tls_ca_file is not None
            if config.tls_ca_file.is_symlink() or not config.tls_ca_file.is_file():
                raise ValueError("VLM tls_ca_file must be a regular non-symlink file")
            self._ssl_context = ssl.create_default_context(cafile=str(config.tls_ca_file))

    def check_health(self) -> None:
        """Authenticate the service and verify its complete protocol identity."""

        parsed = urlsplit(self.config.endpoint)
        health_url = urlunsplit(
            (parsed.scheme, parsed.netloc, "/healthz", "", "")
        )
        timestamp = str(int(self._clock()))
        request_digest = request_signature(self._secret, timestamp, b"")
        request = Request(
            health_url,
            method="GET",
            headers={
                "X-Jiankong-Timestamp": timestamp,
                "X-Jiankong-Signature": request_digest,
            },
        )
        try:
            open_kwargs: dict[str, Any] = {
                "timeout": min(3.0, self.config.timeout_seconds)
            }
            if self._ssl_context is not None:
                open_kwargs["context"] = self._ssl_context
            with self._opener(request, **open_kwargs) as response:
                status = int(getattr(response, "status", 200))
                payload_bytes = response.read(_MAX_RESPONSE_BYTES + 1)
                response_headers = getattr(response, "headers", {})
                signed_response = response_headers.get(
                    "X-Jiankong-Response-Signature"
                )
        except HTTPError as error:
            raise _http_review_error(int(error.code)) from error
        except (URLError, OSError, TimeoutError) as error:
            raise VLMReviewError("VLM review service is unavailable") from error
        if status != 200:
            raise _http_review_error(status)
        if len(payload_bytes) > _MAX_RESPONSE_BYTES:
            raise VLMReviewPermanentError("VLM health response is too large")
        expected_signature = response_signature(
            self._secret, request_digest, payload_bytes
        )
        if not isinstance(signed_response, str) or not hmac.compare_digest(
            expected_signature, signed_response
        ):
            raise VLMReviewPermanentError("VLM health response signature is invalid")
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VLMReviewPermanentError("VLM health response is invalid JSON") from error
        if (
            not isinstance(payload, dict)
            or payload.get("ready") is not True
            or payload.get("model_version") != self.config.expected_model_version
            or payload.get("prompt_revision") != self.config.expected_prompt_revision
            or payload.get("evidence_revision") != self.config.expected_evidence_revision
        ):
            raise VLMReviewPermanentError("VLM health response revision mismatch")

    def review(
        self,
        event_id: str,
        request_id: str,
        clip_path: str | Path,
        overlay_path: str | Path,
    ) -> VLMReviewResult:
        return self.review_prepared(
            self.prepare(event_id, request_id, clip_path, overlay_path)
        )

    def prepare(
        self,
        event_id: str,
        request_id: str,
        clip_path: str | Path,
        overlay_path: str | Path,
    ) -> VLMPreparedReview:
        """Read and package immutable evidence without occupying the model slot."""

        if _SAFE_EVENT_ID.fullmatch(event_id) is None:
            raise ValueError("invalid VLM event_id")
        if _SAFE_REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("invalid VLM request_id")
        clip = Path(clip_path)
        overlay = Path(overlay_path)
        clip_size = _regular_file_size(clip, description="VLM review clip")
        overlay_size = _regular_file_size(overlay, description="VLM review overlay")
        # Stored ZIP overhead is small and deterministic; reject before
        # allocating the archive when the raw evidence is already too large.
        if clip_size <= 0 or overlay_size <= 0:
            raise VLMReviewError("VLM review input is empty")
        if clip_size + overlay_size + 4096 > self.config.max_request_bytes:
            raise VLMReviewError("VLM review request exceeds configured size limit")
        body = self._archive(
            event_id,
            request_id,
            self.config.expected_prompt_revision,
            self.config.expected_evidence_revision,
            clip,
            overlay,
            max_bytes=self.config.max_request_bytes,
        )
        if len(body) > self.config.max_request_bytes:
            raise VLMReviewError("VLM review request exceeds configured size limit")
        return VLMPreparedReview(event_id, request_id, body)

    def review_prepared(
        self,
        prepared: VLMPreparedReview,
    ) -> VLMReviewResult:
        event_id = prepared.event_id
        request_id = prepared.request_id
        body = prepared.body
        timestamp = str(int(self._clock()))
        request_digest = request_signature(self._secret, timestamp, body)
        request = Request(
            self.config.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/zip",
                "Content-Length": str(len(body)),
                "X-Jiankong-Timestamp": timestamp,
                "X-Jiankong-Signature": request_digest,
            },
        )
        try:
            open_kwargs: dict[str, Any] = {"timeout": self.config.timeout_seconds}
            if self._ssl_context is not None:
                open_kwargs["context"] = self._ssl_context
            with self._opener(request, **open_kwargs) as response:
                status = int(getattr(response, "status", 200))
                payload_bytes = response.read(_MAX_RESPONSE_BYTES + 1)
                response_headers = getattr(response, "headers", {})
                signed_response = response_headers.get(
                    "X-Jiankong-Response-Signature"
                )
        except HTTPError as error:
            raise _http_review_error(int(error.code)) from error
        except (URLError, OSError, TimeoutError) as error:
            raise VLMReviewError("VLM review service is unavailable") from error
        if status != 200:
            raise _http_review_error(status)
        if len(payload_bytes) > _MAX_RESPONSE_BYTES:
            raise VLMReviewError("VLM review response is too large")
        expected_response_signature = response_signature(
            self._secret, request_digest, payload_bytes
        )
        if not isinstance(signed_response, str) or not hmac.compare_digest(
            expected_response_signature, signed_response
        ):
            raise VLMReviewPermanentError("VLM review response signature is invalid")
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VLMReviewPermanentError("VLM review response is invalid JSON") from error
        return self._result(event_id, request_id, payload)

    @staticmethod
    def _archive(
        event_id: str,
        request_id: str,
        prompt_revision: str,
        evidence_revision: str,
        clip: Path,
        overlay: Path,
        *,
        max_bytes: int,
    ) -> bytes:
        buffer = _BoundedBytesIO(max_bytes)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                "request.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "event_id": event_id,
                        "request_id": request_id,
                        "prompt_revision": prompt_revision,
                        "evidence_revision": evidence_revision,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
            )
            for source_path, member_name in (
                (clip, "clip.mp4"),
                (overlay, "overlay.json"),
            ):
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(source_path, flags)
                except OSError as error:
                    raise VLMReviewError("VLM review input is unavailable") from error
                try:
                    opened = os.fstat(descriptor)
                    if not stat.S_ISREG(opened.st_mode):
                        raise VLMReviewError("VLM review input must be a regular file")
                    with os.fdopen(descriptor, "rb") as source:
                        descriptor = -1
                        with archive.open(member_name, "w") as target:
                            shutil.copyfileobj(source, target, length=1024 * 1024)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
        return buffer.getvalue()

    def _result(self, event_id: str, request_id: str, payload: Any) -> VLMReviewResult:
        if (
            not isinstance(payload, dict)
            or payload.get("event_id") != event_id
            or payload.get("request_id") != request_id
        ):
            raise VLMReviewPermanentError("VLM review response event_id mismatch")
        try:
            latency = payload.get("latency_seconds")
            result = VLMReviewResult(
                event_id=event_id,
                request_id=request_id,
                result=str(payload["result"]),
                label=str(payload["label"]),
                model_version=str(payload["model_version"]),
                prompt_revision=str(payload["prompt_revision"]),
                evidence_revision=str(payload["evidence_revision"]),
                reviewed_at=str(payload["reviewed_at"]),
                latency_seconds=float(latency) if latency is not None else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VLMReviewPermanentError("VLM review response failed validation") from error
        if (
            result.model_version != self.config.expected_model_version
            or result.prompt_revision != self.config.expected_prompt_revision
            or result.evidence_revision != self.config.expected_evidence_revision
        ):
            raise VLMReviewPermanentError("VLM review response revision mismatch")
        return result


class VLMReviewWorker:
    """Serialize remote reviews so event bursts do not overload the model."""

    def __init__(self, client: VLMReviewClient) -> None:
        self.client = client
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jiankong-vlm-review"
        )
        self._prepare_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jiankong-vlm-prepare"
        )

    def submit(
        self,
        event_id: str,
        request_id: str,
        clip_path: str | Path,
        overlay_path: str | Path,
    ) -> Future[VLMReviewResult]:
        return self._executor.submit(
            self.client.review, event_id, request_id, clip_path, overlay_path
        )

    def prepare(
        self,
        event_id: str,
        request_id: str,
        clip_path: str | Path,
        overlay_path: str | Path,
    ) -> Future[VLMPreparedReview]:
        return self._prepare_executor.submit(
            self.client.prepare,
            event_id,
            request_id,
            clip_path,
            overlay_path,
        )

    def submit_prepared(
        self,
        prepared: Future[VLMPreparedReview],
    ) -> Future[VLMReviewResult]:
        return self._executor.submit(self._review_prepared, prepared)

    def _review_prepared(
        self,
        prepared_future: Future[VLMPreparedReview],
    ) -> VLMReviewResult:
        return self.client.review_prepared(prepared_future.result())

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        self._prepare_executor.shutdown(wait=wait, cancel_futures=not wait)
