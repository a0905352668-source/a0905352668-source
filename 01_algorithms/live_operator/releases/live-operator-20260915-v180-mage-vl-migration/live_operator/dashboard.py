"""Read-only live dashboard adapter for one timestamped operator run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import parse_qs, quote, unquote
from zoneinfo import ZoneInfo

from live_operator.camera_management import CameraManagementError, CameraManagementStore
from live_operator.config import redact_text
from live_operator.event_index import (
    EventIndex,
    EventIndexError,
    IndexedRecord,
    IndexSummary,
    RunSignature,
)
from live_operator.false_positive_dataset import archive_person_roi, remove_person_roi
from live_operator.health import (
    DEFAULT_HEALTH_PATH,
    DEFAULT_STATE_PATH,
    build_public_health,
)
from live_operator.range_server import HTTPResponse, file_response
from live_operator.video_download import ExportBusy, VideoDownloads
from live_operator.storage_paths import RunStorage, resolve_run_storage
from live_operator.storage_health import StorageHealth
from live_operator.review_artifacts import (
    FALSE_POSITIVE_REASONS,
    archive_reviewed_event,
    create_fixed_template,
    fixed_template_image,
    list_fixed_templates,
    remove_fixed_template,
)
from live_operator.vlm_review import VLM_EVIDENCE_REVISION
from live_operator.vlm_state import (
    DEFAULT_VLM_STATE_FILENAME,
    VLM_STATE_FIELDS,
    VLMReviewStateStore,
    VLMStateError,
)


_STATIC_DIR = Path(__file__).with_name("static")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
_SAFE_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_PRIVATE_KEYS = {
    "credentials",
    "live_streams",
    "password",
    "rtsp_url",
    "rtsp_urls",
    "source_url",
    "streams",
    "username",
}
_EVENT_SUMMARY_KEYS = {
    "camera",
    "detector_state",
    "event_id",
    "level",
    "occurred_at",
    "risk_peak",
    "status",
    "review_category",
    "vlm_filter_evidence_revision",
    "vlm_filter_result",
}
_REVIEW_RESULTS = {"pending", "confirmed", "false_positive"}
_CONFIRMED_REVIEW_CATEGORIES = {"phone_use", "screen_capture"}
_VLM_FILTER_RESULTS = {"pass", "filter", "pending", "uncertain", "error"}
_ASYNC_EVENT_REFRESH_THRESHOLD = 1000
_RISK_BANDS = ("high", "medium", "low")
_REVIEW_RANGES = {"all", "today", "7d"}
_REVIEW_PATH = re.compile(r"/api/events/([^/]+)/review")
_HISTORICAL_REVIEW_PATH = re.compile(r"/api/runs/([^/]+)/events/([^/]+)/review")
_HISTORICAL_CLIP_PATH = re.compile(r"/runs/([^/]+)/clips/([^/]+)")
_DOWNLOAD_PATH = re.compile(r"/api/(?:runs/([^/]+)/)?events/([^/]+)/download(/file)?")
_FIXED_TEMPLATE_PATH = re.compile(r"/api/fixed-objects/([^/]+)")
_FIXED_TEMPLATE_IMAGE_PATH = re.compile(r"/fixed-objects/([^/]+)/([^/]+)")
_CAMERA_MANAGEMENT_PREVIEW_PATH = re.compile(
    r"/api/camera-management/([A-Za-z][A-Za-z0-9_-]{0,31})/preview"
)
_RUN_ID = re.compile(r"live_[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MONTH = re.compile(r"\d{4}-\d{2}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_HOUR = re.compile(r"(?:[01]\d|2[0-3])")
_POSITIVE_INTEGER = re.compile(r"[1-9]\d*")
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_BEIJING = ZoneInfo("Asia/Shanghai")
_REASONS = {
    400: "Bad Request",
    200: "OK",
    202: "Accepted",
    206: "Partial Content",
    304: "Not Modified",
    404: "Not Found",
    409: "Conflict",
    405: "Method Not Allowed",
    409: "Conflict",
    416: "Range Not Satisfiable",
    500: "Internal Server Error",
}


class DashboardApp:
    """Serve status, events, static UI, and recorded clips for one run."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        redactor: Callable[[str], str] = redact_text,
        lifecycle_state_path: str | Path = DEFAULT_STATE_PATH,
        watchdog_health_path: str | Path = DEFAULT_HEALTH_PATH,
        config_path: str | Path | None = None,
        calibration_dir: str | Path | None = None,
        camera_publish_command: tuple[str, ...] = (
            "/usr/local/bin/jiankong-camera-publish-request",
        ),
        clock: Callable[[], float] = time.time,
        hidden_event_cameras: Iterable[str] = (),
    ) -> None:
        self.run_dir = Path(run_dir)
        storage = resolve_run_storage(self.run_dir)
        self._storage_bindings = {storage.run_dir: storage}
        self.dashboard_dir = storage.metadata_dir
        self.clips_dir = storage.clips_dir
        self.event_reviews_path = self.dashboard_dir / "event_reviews.json"
        self._storage_health = StorageHealth(self.run_dir)
        self.false_positive_dir = self.run_dir / "reviewed_false_positives"
        self.redactor = redactor
        self.lifecycle_state_path = Path(lifecycle_state_path)
        self.watchdog_health_path = Path(watchdog_health_path)
        self.config_path = Path(config_path) if config_path is not None else None
        self.calibration_dir = (
            Path(calibration_dir) if calibration_dir is not None else None
        )
        self.camera_publish_command = tuple(camera_publish_command)
        self.clock = clock
        self.hidden_event_cameras = frozenset(hidden_event_cameras)
        self._review_lock = threading.Lock()
        self._event_index_lock = threading.Lock()
        self._event_index: EventIndex | None = None
        self._event_index_summary = IndexSummary(0, 0, 0, 0)
        self._event_index_last_refresh_ms = 0.0
        self._event_cache_lock = threading.RLock()
        self._indexed_records_cache: list[IndexedRecord] | None = None
        self._indexed_records_fingerprint: tuple[tuple[str, RunSignature], ...] | None = None
        self._event_refresh_thread: threading.Thread | None = None
        self._event_cache_warmup_thread: threading.Thread | None = None
        self._prepared_records_source: list[IndexedRecord] | None = None
        self._prepared_records_vlm_fingerprint: tuple[
            tuple[str, int, int], ...
        ] | None = None
        self._vlm_state_cache_fingerprint: tuple[tuple[str, int, int], ...] | None = None
        self._vlm_state_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._prepared_records_vlm_states: dict[
            str, dict[str, dict[str, Any]]
        ] = {}
        self._prepared_records_cache: tuple[
            list[tuple[datetime, dict[str, Any]]],
            dict[tuple[str, str], IndexedRecord],
            dict[tuple[str, str], str],
            Counter[str],
        ] | None = None
        self._camera_management_lock = threading.Lock()
        # Preview capture opens a short-lived local RTSP reader.  Serializing
        # cache misses prevents the camera-management gallery from starting a
        # burst of ffmpeg HEVC decoders when it first opens.
        self._camera_preview_lock = threading.Lock()
        # Historical clips created before browser-proxy support are converted
        # lazily on first playback. One slot prevents simultaneous range
        # requests from launching duplicate NVENC jobs for the same file.
        self._browser_proxy_lock = threading.Lock()
        self._video_downloads = VideoDownloads()
        self._camera_publish_process: subprocess.Popen[bytes] | None = None

    def start_event_cache_warmup(self) -> threading.Thread:
        """Prepare the historical event view before the first browser request."""

        with self._event_cache_lock:
            if self._event_cache_warmup_thread is not None:
                return self._event_cache_warmup_thread
            warmup = threading.Thread(
                target=self._warm_event_cache,
                daemon=True,
                name="dashboard-event-cache-warmup",
            )
            self._event_cache_warmup_thread = warmup
            warmup.start()
            return warmup

    def _warm_event_cache(self) -> None:
        try:
            self._prepared_event_records()
        except Exception:
            # Warmup is best-effort. A normal API request retains the existing
            # fallback and error handling if storage is temporarily unavailable.
            return

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
        query_string: str = "",
    ) -> HTTPResponse:
        normalized_method = method.upper()
        download_match = _DOWNLOAD_PATH.fullmatch(path)
        if download_match is not None:
            return self._download_video(normalized_method, path, download_match, headers or {})
        if normalized_method == "POST":
            if path == "/api/camera-management/draft":
                return self._save_camera_draft(body)
            if path == "/api/camera-management/publish":
                return self._publish_camera_draft(body)
            match = _REVIEW_PATH.fullmatch(path)
            if match is not None:
                event_id = unquote(match.group(1))
                return self._update_review(event_id, body)
            historical_match = _HISTORICAL_REVIEW_PATH.fullmatch(path)
            if historical_match is not None:
                run_dir = self._historical_run_dir(unquote(historical_match.group(1)))
                if run_dir is None:
                    return self._error_response(404, "event not found")
                return self._update_review(unquote(historical_match.group(2)), body, run_dir=run_dir)
            return HTTPResponse(
                405, {"Allow": "GET, HEAD", "Content-Length": "0"}, b""
            )
        if normalized_method == "DELETE":
            if path == "/api/camera-management/draft":
                return self._discard_camera_draft()
            fixed_match = _FIXED_TEMPLATE_PATH.fullmatch(path)
            if fixed_match is not None:
                template_id = unquote(fixed_match.group(1))
                if not remove_fixed_template(self.run_dir, template_id):
                    return self._error_response(404, "fixed object not found")
                return self._json_response(
                    {"template_id": template_id, "deleted": True},
                    head_only=False,
                )
            return HTTPResponse(
                405, {"Allow": "GET, HEAD", "Content-Length": "0"}, b""
            )
        if normalized_method not in {"GET", "HEAD"}:
            return HTTPResponse(
                405, {"Allow": "GET, HEAD", "Content-Length": "0"}, b""
            )
        head_only = normalized_method == "HEAD"
        request_headers = {key.lower(): value for key, value in (headers or {}).items()}

        if path in _STATIC_FILES:
            filename, content_type = _STATIC_FILES[path]
            source = _STATIC_DIR / filename
            if not source.is_file():
                return HTTPResponse(404, {"Content-Length": "0"}, b"")
            body = source.read_bytes()
            # HTML may be personalized by the authentication wrapper.
            etag = '"' + hashlib.sha256(body).hexdigest() + '"' if path.startswith("/static/") else None
            if etag and request_headers.get("if-none-match") == etag:
                return HTTPResponse(304, {"ETag": etag, "Cache-Control": "private, no-cache"}, b"")
            return HTTPResponse(
                200,
                {
                    "Content-Type": content_type,
                    "Content-Length": str(len(body)),
                    "Cache-Control": "no-cache",
                    **({"ETag": etag} if etag else {}),
                },
                b"" if head_only else body,
            )
        if path == "/api/status":
            storage_health = self._storage_health.snapshot()
            raw_status = {} if storage_health["state"] == "unavailable" else self._read_json(
                self._storage_for_run(self.run_dir).metadata_dir / "status.json", {})
            payload = build_public_health(
                {**raw_status, "storage": storage_health} if isinstance(raw_status, dict) else {"storage": storage_health},
                self._read_json(self.lifecycle_state_path, {}),
                self._read_json(self.watchdog_health_path, {}),
                self._public_event_index_summary(),
                self.clock(),
            )
            return self._json_response(payload, head_only=head_only)
        if path == "/api/events":
            try:
                payload = self._events(query_string)
            except ValueError as error:
                return self._error_response(400, str(error), head_only=head_only)
            return self._json_response(payload, head_only=head_only)
        if path == "/api/fixed-objects":
            return self._json_response(
                {"templates": list_fixed_templates(self.run_dir)},
                head_only=head_only,
            )
        if path == "/api/camera-management":
            try:
                payload = self._camera_management_store().public_state()
            except (CameraManagementError, OSError, ValueError) as error:
                return self._error_response(409, str(error), head_only=head_only)
            return self._json_response(payload, head_only=head_only)
        camera_preview_match = _CAMERA_MANAGEMENT_PREVIEW_PATH.fullmatch(path)
        if camera_preview_match is not None:
            try:
                source = self._camera_preview(camera_preview_match.group(1))
            except CameraManagementError as error:
                return self._error_response(409, str(error), head_only=head_only)
            if source is None:
                return self._error_response(409, "无法获取当前摄像头画面，请确认该路已启用并在线", head_only=head_only)
            return file_response(
                source,
                range_header=request_headers.get("range"),
                head_only=head_only,
            )
        fixed_image_match = _FIXED_TEMPLATE_IMAGE_PATH.fullmatch(path)
        if fixed_image_match is not None:
            source = fixed_template_image(
                self.run_dir,
                unquote(fixed_image_match.group(1)),
                unquote(fixed_image_match.group(2)),
            )
            if source is None:
                return HTTPResponse(404, {"Content-Length": "0"}, b"")
            source = self._browser_compatible_clip(source)
            return file_response(
                source,
                range_header=request_headers.get("range"),
                head_only=head_only,
            )
        historical_clip_match = _HISTORICAL_CLIP_PATH.fullmatch(path)
        if historical_clip_match is not None:
            run_dir = self._historical_run_dir(unquote(historical_clip_match.group(1)))
            source = (
                self._safe_clip_path(unquote(historical_clip_match.group(2)), run_dir=run_dir)
                if run_dir is not None
                else None
            )
            if source is None:
                return HTTPResponse(404, {"Content-Length": "0"}, b"")
            source = self._browser_compatible_clip(source)
            return file_response(
                source,
                range_header=request_headers.get("range"),
                head_only=head_only,
            )
        if path.startswith("/clips/"):
            name = unquote(path.removeprefix("/clips/"))
            source = self._safe_clip_path(name)
            if source is None:
                return HTTPResponse(404, {"Content-Length": "0"}, b"")
            return file_response(
                source,
                range_header=request_headers.get("range"),
                head_only=head_only,
            )
        return HTTPResponse(404, {"Content-Length": "0"}, b"")

    def _download_video(self, method: str, path: str, match: re.Match, headers: Mapping[str, str]) -> HTTPResponse:
        file_request = match.group(3) is not None
        if method not in ({"GET", "HEAD"} if file_request else {"GET", "HEAD", "POST"}):
            return HTTPResponse(405, {"Allow": "GET, HEAD" if file_request else "GET, HEAD, POST", "Content-Length": "0"}, b"")
        head_only = method == "HEAD"
        event_id = unquote(match.group(2))
        run_dir = self._historical_run_dir(unquote(match.group(1))) if match.group(1) else self.run_dir
        if run_dir is None or not _SAFE_EVENT_ID.fullmatch(event_id) or event_id in {".", ".."}:
            return self._error_response(404, "event not found", head_only=head_only)
        try:
            run_dir = self._storage_for_run(run_dir).run_dir
        except (OSError, ValueError, RuntimeError):
            return self._error_response(409, "视频存储路径不可用", head_only=head_only)
        source = self._safe_clip_path(f"{event_id}.browser.mp4", run_dir=run_dir) or self._safe_clip_path(f"{event_id}.mp4", run_dir=run_dir)
        overlay = self._safe_clip_path(f"{event_id}.json", run_dir=run_dir)
        if source is None or overlay is None:
            return self._error_response(404, "event not found", head_only=head_only)
        try:
            state, output = self._video_downloads.status(source, overlay, start=method == "POST")
        except (OSError, ValueError) as error:
            return self._error_response(409, str(error) if isinstance(error, ExportBusy) else "视频导出暂不可用", head_only=head_only)
        if file_request:
            if state != "ready":
                return self._error_response(409, "带框视频尚未就绪", head_only=head_only)
            response = file_response(output, range_header={k.lower(): v for k, v in headers.items()}.get("range"), head_only=head_only)
            return HTTPResponse(response.status, {**response.headers, "Content-Disposition": f'attachment; filename="{event_id}-boxed.mp4"', "Cache-Control": "private, no-store"}, response.body)
        payload = {"state": state}
        if state == "ready":
            payload["download_url"] = f"{path}/file"
        if state == "error":
            payload["error"] = "视频生成失败，请重试"
        return self._json_response(payload, head_only=head_only, status=202 if state in {"queued", "running"} else 200)

    def _camera_management_store(self) -> CameraManagementStore:
        if self.config_path is None or self.calibration_dir is None:
            raise CameraManagementError("当前运行未启用摄像头管理配置")
        return CameraManagementStore(self.config_path, self.calibration_dir)

    def _browser_compatible_clip(self, source: Path) -> Path:
        """Return or lazily build a baseline H.264 proxy for an old MP4 clip."""

        if source.suffix.lower() != ".mp4" or source.name.endswith(".browser.mp4"):
            return source
        try:
            with source.open("rb") as handle:
                if handle.read(8)[4:8] != b"ftyp":
                    return source
        except OSError:
            return source
        destination = source.with_name(f"{source.stem}.browser.mp4")
        existing = self._safe_clip_path(destination.name, run_dir=source.parents[2])
        if existing is not None:
            return existing
        with self._browser_proxy_lock:
            existing = self._safe_clip_path(destination.name, run_dir=source.parents[2])
            if existing is not None:
                return existing
            temporary = source.with_name(f".{source.stem}.ondemand.partial.browser.mp4")
            temporary.unlink(missing_ok=True)
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-an",
                "-vf",
                "scale=1920:1080:flags=lanczos",
                "-c:v",
                "h264_nvenc",
                "-profile:v",
                "baseline",
                "-level:v",
                "4.1",
                "-g",
                "25",
                "-bf",
                "0",
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "23",
                "-b:v",
                "3500k",
                "-maxrate",
                "5M",
                "-bufsize",
                "10M",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(temporary),
            ]
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=90,
                    check=False,
                )
                if completed.returncode != 0 or not self._valid_browser_proxy(temporary):
                    temporary.unlink(missing_ok=True)
                    return source
                os.replace(temporary, destination)
                return destination
            except (OSError, subprocess.SubprocessError, ValueError):
                temporary.unlink(missing_ok=True)
                return source

    @staticmethod
    def _valid_browser_proxy(source: Path) -> bool:
        try:
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,profile,width,height,pix_fmt,level",
                    "-of",
                    "json",
                    str(source),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
            stream = payload.get("streams", [{}])[0]
            return (
                stream.get("codec_name") == "h264"
                and stream.get("profile") == "Constrained Baseline"
                and stream.get("width") == 1920
                and stream.get("height") == 1080
                and stream.get("pix_fmt") == "yuv420p"
                and stream.get("level") == 41
            )
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError, IndexError):
            return False

    def _save_camera_draft(self, body: bytes) -> HTTPResponse:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._error_response(400, "摄像头配置必须是 JSON")
        try:
            with self._camera_management_lock:
                saved = self._camera_management_store().save(payload)
        except (CameraManagementError, OSError, ValueError) as error:
            return self._error_response(400, str(error))
        return self._json_response(saved, head_only=False)

    def _discard_camera_draft(self) -> HTTPResponse:
        try:
            with self._camera_management_lock:
                state = self._camera_management_store().discard_draft()
        except (CameraManagementError, OSError, ValueError) as error:
            return self._error_response(400, str(error))
        return self._json_response(state, head_only=False)

    def _publish_camera_draft(self, body: bytes) -> HTTPResponse:
        """Accept one confirmed, detached publish request.

        The invoked command is fixed at construction time and accepts no
        browser-controlled arguments.  It starts in a new session so stopping
        the current dashboard during the managed restart cannot kill the
        root-owned publisher halfway through the operation.
        """

        try:
            with self._camera_management_lock:
                state = self._camera_management_store().public_state()
                if state.get("pending_activation") is not True:
                    raise CameraManagementError("没有待发布的摄像头配置")
                running = self._camera_publish_process
                if running is not None and running.poll() is None:
                    return self._error_response(409, "已有摄像头配置正在发布")
                if not self.camera_publish_command or any(
                    not isinstance(item, str) or not item for item in self.camera_publish_command
                ):
                    raise CameraManagementError("摄像头发布命令未配置")
                log_path = self.run_dir / "dashboard" / "camera_management_publish.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("ab", buffering=0)
                try:
                    self._camera_publish_process = subprocess.Popen(
                        list(self.camera_publish_command),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        close_fds=True,
                        env={
                            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                            "LANG": "C.UTF-8",
                        },
                    )
                finally:
                    log.close()
        except (CameraManagementError, OSError, ValueError) as error:
            return self._error_response(400, str(error))
        return self._json_response(
            {
                "accepted": True,
                "message": "已开始预检并发布；系统将自动安全重启。",
            },
            status=202,
            head_only=False,
        )

    def _camera_preview(self, relay: str) -> Path | None:
        """Capture one short-lived calibration frame from active or draft config.

        Unchanged active sources use their local MediaMTX relay.  A newly
        saved draft has no local relay yet, so it uses only that draft's
        server-side RTSP credentials for this one manual ffmpeg capture.  The
        browser never receives the source URL or credentials.
        """

        try:
            camera, uses_active_relay = self._camera_management_store().preview_camera(relay)
        except CameraManagementError:
            raise
        except (OSError, ValueError) as error:
            raise CameraManagementError("摄像头管理配置不可用") from error
        source_url = f"rtsp://127.0.0.1:8554/{relay}" if uses_active_relay else camera.rtsp_url()
        preview_dir = self.run_dir / "dashboard" / "calibration_previews"
        destination = preview_dir / f"{relay}.jpg"
        try:
            age = self.clock() - destination.stat().st_mtime
        except OSError:
            age = float("inf")
        if destination.is_file() and 0 <= age < 20.0:
            return destination
        with self._camera_preview_lock:
            # Another gallery request may have finished while this request
            # waited for the capture slot.
            try:
                age = self.clock() - destination.stat().st_mtime
            except OSError:
                age = float("inf")
            if destination.is_file() and 0 <= age < 20.0:
                return destination
            preview_dir.mkdir(parents=True, exist_ok=True)
            # Keep the final suffix as .jpg so ffmpeg selects the image muxer.
            temporary = preview_dir / f".{relay}.tmp.jpg"
            temporary.unlink(missing_ok=True)
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                # A new HEVC RTSP reader can begin on a P/B frame.  Encoding the
                # first decoded frame creates a syntactically valid but grey,
                # reference-corrupted JPEG.  Wait for the next key frame instead.
                "-skip_frame",
                "nokey",
                "-rtsp_transport",
                "tcp",
                "-i",
                source_url,
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                str(temporary),
            ]
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    # Calibration is a manual action.  Allow a full camera GOP to
                    # reach its next I-frame; this does not run on the inference
                    # path and therefore cannot affect live FPS.
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                temporary.unlink(missing_ok=True)
                return None
            if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size < 1024:
                temporary.unlink(missing_ok=True)
                return None
            os.replace(temporary, destination)
            return destination

    def __call__(self, environ: Mapping[str, Any], start_response: Any) -> Iterable[bytes]:
        headers = {}
        if environ.get("HTTP_RANGE"):
            headers["Range"] = str(environ["HTTP_RANGE"])
        if environ.get("HTTP_IF_NONE_MATCH"):
            headers["If-None-Match"] = str(environ["HTTP_IF_NONE_MATCH"])
        body = b""
        if str(environ.get("REQUEST_METHOD", "GET")).upper() == "POST":
            try:
                content_length = int(str(environ.get("CONTENT_LENGTH", "0")))
                if content_length < 0:
                    raise ValueError
                body = environ["wsgi.input"].read(content_length)
            except (KeyError, TypeError, ValueError):
                body = b""
        response = self.handle(
            str(environ.get("REQUEST_METHOD", "GET")),
            str(environ.get("PATH_INFO", "/")),
            headers,
            body,
            str(environ.get("QUERY_STRING", "")),
        )
        reason = _REASONS.get(response.status, "Unknown")
        start_response(
            f"{response.status} {reason}", list(response.headers.items())
        )
        return response.iter_body()

    def _events(self, query_string: str) -> dict[str, Any]:
        query = self._parse_event_query(query_string)
        records, indexed_by_key, display_ids, daily_totals = (
            self._prepared_event_records()
        )
        records = [
            record
            for record in records
            if record[1].get("camera") not in self.hidden_event_cameras
        ]
        daily_totals = Counter(
            occurred_at.strftime("%Y-%m-%d") for occurred_at, _event in records
        )
        if query["screen_capture_only"]:
            matching = [
                record
                for record in records
                if record[1].get("review_result") == "confirmed"
                and record[1].get("review_category") == "screen_capture"
            ]
            return self._screen_capture_events_response(
                matching,
                indexed_by_key,
                display_ids,
                query["page"],
                query["page_size"],
                selected_month=query["month"],
                selected_date=query["date"],
                review_range=query["review_range"],
            )
        records = [
            record
            for record in records
            if self._matches_risk_bands(record[1], query["risk_bands"])
        ]
        if not query["include_false_positives"]:
            records = [
                record
                for record in records
                if record[1].get("review_result") != "false_positive"
            ]
        review_category = query["review_category"]
        if review_category:
            records = [
                record
                for record in records
                if record[1].get("review_result") == "confirmed"
                and record[1].get("review_category") == review_category
            ]

        cameras = query["cameras"]
        camera_set = set(cameras)
        camera_scope_records = [
            record
            for record in records
            if not camera_set or record[1].get("camera") in camera_set
        ]
        selected_date = query["date"]
        if selected_date is None:
            selected_date = (
                camera_scope_records[0][0].strftime("%Y-%m-%d")
                if camera_scope_records
                else ""
            )
        selected_hour = query["hour"]
        if selected_hour is None:
            selected_hour = next(
                (
                    occurred_at.strftime("%H")
                    for occurred_at, _event in camera_scope_records
                    if occurred_at.strftime("%Y-%m-%d") == selected_date
                ),
                "",
            )
        selected_scope_records = [
            record
            for record in camera_scope_records
            if record[0].strftime("%Y-%m-%d") == selected_date
            and record[0].strftime("%H") == selected_hour
        ]
        vlm_review = self._vlm_review_summary(selected_scope_records)
        if query["exclude_vlm_filtered"]:
            camera_records = [
                record
                for record in camera_scope_records
                if record[1].get("review_result") == "confirmed"
                or record[1].get("vlm_filter_result") == "pass"
            ]
        else:
            camera_records = camera_scope_records
        selected_month = query["month"]
        if selected_month is None:
            selected_month = (
                selected_date[:7]
                if selected_date
                else datetime.now(_BEIJING).strftime("%Y-%m")
            )

        calendar_counts = Counter(
            occurred_at.strftime("%Y-%m-%d")
            for occurred_at, _event in camera_records
            if occurred_at.strftime("%Y-%m") == selected_month
        )
        hour_counts = Counter(
            occurred_at.strftime("%Y-%m-%dT%H")
            for occurred_at, _event in camera_records
            if occurred_at.strftime("%Y-%m-%d") == selected_date
        )
        matching = [
            record
            for record in camera_records
            if record[0].strftime("%Y-%m-%d") == selected_date
            and record[0].strftime("%H") == selected_hour
        ]

        total = len(matching)
        page_size = query["page_size"]
        total_pages = (total + page_size - 1) // page_size
        page = min(query["page"], total_pages) if total_pages else 1
        start = (page - 1) * page_size
        page_events = []
        for _occurred_at, event in matching[start : start + page_size]:
            indexed_record = indexed_by_key.get(self._display_key(event))
            if indexed_record is None:
                continue
            decorated = self._decorate_indexed_record(indexed_record)
            if decorated is None:
                continue
            public_event = decorated[1]
            public_event.pop("_source_run_id", None)
            public_event["display_id"] = display_ids[self._display_key(event)]
            page_events.append(public_event)
        risk_values = [
            risk
            for _occurred_at, event in matching
            if self._is_finite_number(risk := event.get("risk_peak"))
        ]
        views = {
            event.get("camera")
            for _occurred_at, event in matching
            if isinstance(event.get("camera"), str) and event.get("camera")
        }
        return {
            "events": page_events,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
            },
            "selection": {
                "camera": cameras[0] if len(cameras) == 1 else "",
                **({"cameras": list(cameras)} if len(cameras) > 1 else {}),
                "month": selected_month,
                "date": selected_date,
                "hour": selected_hour,
                "daily_total": int(daily_totals.get(selected_date, 0)),
            },
            "calendar": dict(sorted(calendar_counts.items())),
            "hours": dict(sorted(hour_counts.items())),
            "vlm_review": vlm_review,
            "summary": {
                **self._human_review_counts(event for _, event in matching),
                "events": total,
                "alarms": sum(
                    event.get("level") == "alarm"
                    for _occurred_at, event in matching
                ),
                "views": len(views),
                "max_risk": max(risk_values, default=0.0),
            },
        }

    @staticmethod
    def _human_review_counts(events: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        counts = {"pending_review": 0, "confirmed_phone_use": 0, "confirmed_screen_capture": 0}
        for event in events:
            result = event.get("review_result", "pending")
            if result == "pending":
                counts["pending_review"] += 1
            elif result == "confirmed":
                category = event.get("review_category")
                if category == "phone_use":
                    counts["confirmed_phone_use"] += 1
                elif category == "screen_capture":
                    counts["confirmed_screen_capture"] += 1
        return counts

    @staticmethod
    def _vlm_review_summary(
        records: Iterable[tuple[datetime, Mapping[str, Any]]],
    ) -> dict[str, int]:
        summary = {"reviewing": 0, "errors": 0, "uncertain": 0}
        for _occurred_at, event in records:
            if event.get("review_result") == "confirmed":
                continue
            result = event.get("vlm_filter_result")
            if result in {None, "", "pending"}:
                summary["reviewing"] += 1
            elif result == "error":
                summary["errors"] += 1
        return summary

    def _screen_capture_events_response(
        self,
        matching: list[tuple[datetime, dict[str, Any]]],
        indexed_by_key: Mapping[tuple[str, str], IndexedRecord],
        display_ids: Mapping[tuple[str, str], str],
        requested_page: int,
        page_size: int,
        *,
        selected_month: str | None,
        selected_date: str | None,
        review_range: str,
    ) -> dict[str, Any]:
        """Return manual screen captures ordered and filtered by review time."""

        review_cache: dict[str, Mapping[str, Any]] = {}
        review_records: list[
            tuple[datetime, datetime | None, datetime, dict[str, Any]]
        ] = []
        for occurred_at, event in matching:
            reviewed_at = self._manual_reviewed_at(event, review_cache)
            review_records.append(
                (reviewed_at or occurred_at, reviewed_at, occurred_at, event)
            )
        review_records.sort(
            key=lambda record: (
                record[0],
                record[2],
                record[3].get("event_id", ""),
                self._source_run_identity(record[3]),
            ),
            reverse=True,
        )

        if selected_month is None:
            selected_month = next(
                (
                    reviewed_at.strftime("%Y-%m")
                    for _sort_time, reviewed_at, _occurred_at, _event in review_records
                    if reviewed_at is not None
                ),
                datetime.now(_BEIJING).strftime("%Y-%m"),
            )
        calendar_counts = Counter(
            reviewed_at.strftime("%Y-%m-%d")
            for _sort_time, reviewed_at, _occurred_at, _event in review_records
            if reviewed_at is not None
            and reviewed_at.strftime("%Y-%m") == selected_month
        )

        today = datetime.now(_BEIJING).date()
        if selected_date:
            filtered = [
                record
                for record in review_records
                if record[1] is not None
                and record[1].strftime("%Y-%m-%d") == selected_date
            ]
        elif review_range == "today":
            filtered = [
                record
                for record in review_records
                if record[1] is not None and record[1].date() == today
            ]
        elif review_range == "7d":
            first_day = today - timedelta(days=6)
            filtered = [
                record
                for record in review_records
                if record[1] is not None and first_day <= record[1].date() <= today
            ]
        else:
            filtered = review_records

        total = len(filtered)
        total_pages = (total + page_size - 1) // page_size
        page = min(requested_page, total_pages) if total_pages else 1
        start = (page - 1) * page_size
        page_events = []
        for _sort_time, reviewed_at, _occurred_at, event in filtered[
            start : start + page_size
        ]:
            indexed_record = indexed_by_key.get(self._display_key(event))
            if indexed_record is None:
                continue
            decorated = self._decorate_indexed_record(indexed_record)
            if decorated is None:
                continue
            public_event = decorated[1]
            public_event.pop("_source_run_id", None)
            public_event["display_id"] = display_ids[self._display_key(event)]
            if reviewed_at is not None:
                public_event["reviewed_at"] = reviewed_at.isoformat()
                public_event["review_time_source"] = "confirmed"
            else:
                public_event["review_time_source"] = "event_fallback"
            page_events.append(public_event)
        risk_values = [
            risk
            for _sort_time, _reviewed_at, _occurred_at, event in filtered
            if self._is_finite_number(risk := event.get("risk_peak"))
        ]
        views = {
            event.get("camera")
            for _sort_time, _reviewed_at, _occurred_at, event in filtered
            if isinstance(event.get("camera"), str) and event.get("camera")
        }
        return {
            "events": page_events,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
            },
            "selection": {
                "camera": "",
                "month": selected_month,
                "date": selected_date or "",
                "hour": "",
                "daily_total": int(calendar_counts.get(selected_date or "", 0)),
                "screen_capture_only": True,
                "review_range": review_range,
            },
            "calendar": dict(sorted(calendar_counts.items())),
            "hours": {},
            "summary": {
                **self._human_review_counts(event for _, _, _, event in filtered),
                "events": total,
                "alarms": sum(
                    event.get("level") == "alarm"
                    for _sort_time, _reviewed_at, _occurred_at, event in filtered
                ),
                "views": len(views),
                "max_risk": max(risk_values, default=0.0),
            },
        }

    def _manual_reviewed_at(
        self,
        event: Mapping[str, Any],
        cache: dict[str, Mapping[str, Any]],
    ) -> datetime | None:
        source_run_id = self._source_run_identity(event)
        if source_run_id not in cache:
            current_run = self._trusted_event_run_dir(self.run_dir)
            if current_run is not None and source_run_id == current_run.name:
                source_run = current_run
            else:
                source_run = self._historical_run_dir(source_run_id)
            reviews: Any = {}
            if source_run is not None:
                reviews = self._read_json(
                    self._storage_for_run(source_run).metadata_dir / "event_reviews.json", {}
                )
            cache[source_run_id] = reviews if isinstance(reviews, dict) else {}
        event_id = event.get("event_id")
        review = cache[source_run_id].get(event_id) if isinstance(event_id, str) else None
        if (
            not isinstance(review, dict)
            or review.get("result") != "confirmed"
            or review.get("category") != "screen_capture"
        ):
            return None
        return self._beijing_time(review.get("updated_at"))

    @staticmethod
    def _display_key(event: Mapping[str, Any]) -> tuple[str, str]:
        return (
            DashboardApp._source_run_identity(event),
            str(event.get("event_id", "")),
        )

    @staticmethod
    def _source_run_identity(event: Mapping[str, Any]) -> str:
        return str(event.get("_source_run_id", ""))

    @classmethod
    def _daily_display_metadata(
        cls,
        records: list[tuple[datetime, dict[str, Any]]],
    ) -> tuple[dict[tuple[str, str], str], Counter[str]]:
        ordinals: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        display_ids: dict[tuple[str, str], str] = {}
        ordered = sorted(
            records,
            key=lambda record: (
                record[0],
                str(record[1].get("event_id", "")),
                cls._source_run_identity(record[1]),
            ),
        )
        for occurred_at, event in ordered:
            date_key = occurred_at.strftime("%Y-%m-%d")
            ordinals[date_key] += 1
            totals[date_key] += 1
            display_ids[cls._display_key(event)] = (
                f"{occurred_at.strftime('%Y%m%d')}{ordinals[date_key]:04d}"
            )
        return display_ids, totals

    def _review_display_id(
        self,
        run_dir: Path,
        event_id: str,
        event: Mapping[str, Any],
    ) -> str:
        """Resolve the same stable daily number displayed by the front end."""

        records: list[tuple[datetime, dict[str, Any]]] = []
        for indexed_record in self._indexed_records():
            summary = self._summarize_indexed_record(indexed_record)
            if summary is not None:
                records.append(summary)
        display_ids, _totals = self._daily_display_metadata(records)
        resolved = display_ids.get((run_dir.name, event_id))
        if resolved is not None:
            return resolved
        occurred_at = self._beijing_time(event.get("occurred_at"))
        date_key = occurred_at.strftime("%Y%m%d") if occurred_at is not None else "unknown"
        return f"{date_key}0000"

    def _event_run_dirs(self) -> list[Path]:
        current = self.run_dir.resolve()
        result = [current]
        try:
            root = current.parent.resolve()
            candidates = sorted(root.glob("live_*"))
        except OSError:
            return result
        for candidate in candidates:
            if candidate.is_symlink() or candidate.name == current.name:
                continue
            resolved = self._historical_run_dir(candidate.name)
            if resolved is not None:
                result.append(resolved)
        return result

    def _indexed_records(self) -> list[IndexedRecord]:
        run_dirs = self._event_run_dirs()
        candidates = [
            {
                "source_run_id": run_dir.name,
                "run_path": str(run_dir),
                "signature": self._event_file_signature(
                    self._storage_for_run(run_dir).metadata_dir,
                    self._storage_for_run(run_dir).clips_dir,
                ),
            }
            for run_dir in run_dirs
        ]
        fingerprint = tuple(
            (str(candidate["source_run_id"]), candidate["signature"])
            for candidate in candidates
        )
        with self._event_cache_lock:
            cached = self._indexed_records_cache
            if cached is not None and fingerprint == self._indexed_records_fingerprint:
                return cached
            if (
                cached is not None
                and self._event_index_summary.total_records
                >= _ASYNC_EVENT_REFRESH_THRESHOLD
            ):
                if (
                    self._event_refresh_thread is None
                    or not self._event_refresh_thread.is_alive()
                ):
                    refresh_thread = threading.Thread(
                        target=self._refresh_index_cache,
                        args=(run_dirs, candidates, fingerprint),
                        daemon=True,
                        name="dashboard-event-index-refresh",
                    )
                    self._event_refresh_thread = refresh_thread
                    refresh_thread.start()
                return cached
            return self._refresh_index_cache(run_dirs, candidates, fingerprint)

    def _refresh_index_cache(
        self,
        run_dirs: list[Path],
        candidates: list[dict[str, Any]],
        fingerprint: tuple[tuple[str, RunSignature], ...],
    ) -> list[IndexedRecord]:
        refresh_started = time.perf_counter()
        try:
            event_index = self._get_event_index()
            summary = event_index.refresh(
                candidates,
                lambda candidate: self._scan_run_records(
                    Path(str(candidate["run_path"]))
                ),
            )
            records = event_index.records()
        except EventIndexError:
            previous_summary = (
                self._event_index.summary
                if self._event_index is not None
                else self._event_index_summary
            )
            summary = IndexSummary(
                0,
                0,
                0,
                previous_summary.total_records,
                fallback_active=True,
                last_error=(
                    previous_summary.last_error
                    if previous_summary.fallback_active
                    and previous_summary.last_error is not None
                    else "index_unavailable"
                ),
            )
            records = []
            for run_dir in run_dirs:
                records.extend(self._scan_run_records(run_dir))
        elapsed_ms = (time.perf_counter() - refresh_started) * 1000
        with self._event_cache_lock:
            self._event_index_summary = summary
            self._event_index_last_refresh_ms = elapsed_ms
            self._indexed_records_cache = records
            self._indexed_records_fingerprint = fingerprint
            if threading.current_thread() is self._event_refresh_thread:
                self._event_refresh_thread = None
            return records

    def _prepared_event_records(
        self,
    ) -> tuple[
        list[tuple[datetime, dict[str, Any]]],
        dict[tuple[str, str], IndexedRecord],
        dict[tuple[str, str], str],
        Counter[str],
    ]:
        indexed_records = self._indexed_records()
        vlm_states, vlm_fingerprint = self._vlm_state_overlays(indexed_records)
        with self._event_cache_lock:
            if (
                self._prepared_records_source is indexed_records
                and self._prepared_records_vlm_fingerprint == vlm_fingerprint
                and self._prepared_records_cache is not None
            ):
                return self._prepared_records_cache

            previous = self._prepared_records_cache
            previous_records = previous[0] if previous is not None else []
            previous_effective_by_key = previous[1] if previous is not None else {}
            previous_source_by_key = {
                (record.source_run_id, record.event_id): record
                for record in (self._prepared_records_source or [])
            }
            records_by_key = {
                self._display_key(event): (occurred_at, event)
                for occurred_at, event in previous_records
            }
            source_by_key: dict[tuple[str, str], IndexedRecord] = {
                (record.source_run_id, record.event_id): record
                for record in indexed_records
            }
            indexed_by_key = dict(previous_effective_by_key)
            changed_keys = {
                key
                for key in set(previous_source_by_key) | set(source_by_key)
                if previous_source_by_key.get(key) != source_by_key.get(key)
            }
            previous_fingerprint = dict(
                (run_id, (modified_ns, size))
                for run_id, modified_ns, size in (
                    self._prepared_records_vlm_fingerprint or ()
                )
            )
            current_fingerprint = {
                run_id: (modified_ns, size)
                for run_id, modified_ns, size in vlm_fingerprint
            }
            changed_runs = {
                run_id
                for run_id in set(previous_fingerprint) | set(current_fingerprint)
                if previous_fingerprint.get(run_id) != current_fingerprint.get(run_id)
            }
            for run_id in changed_runs:
                previous_run_states = self._prepared_records_vlm_states.get(run_id, {})
                current_run_states = vlm_states.get(run_id, {})
                changed_keys.update(
                    (run_id, event_id)
                    for event_id in set(previous_run_states) | set(current_run_states)
                    if previous_run_states.get(event_id) != current_run_states.get(event_id)
                )

            if previous is None:
                changed_keys.update(source_by_key)
            for key in changed_keys:
                indexed_record = source_by_key.get(key)
                if indexed_record is None:
                    records_by_key.pop(key, None)
                    indexed_by_key.pop(key, None)
                    continue
                state = vlm_states.get(indexed_record.source_run_id, {}).get(
                    indexed_record.event_id
                )
                if state is None:
                    effective_record = indexed_record
                else:
                    payload = dict(indexed_record.payload)
                    for field in VLM_STATE_FIELDS:
                        payload.pop(field, None)
                    payload.update(state)
                    effective_record = IndexedRecord(
                        source_run_id=indexed_record.source_run_id,
                        event_id=indexed_record.event_id,
                        occurred_at=indexed_record.occurred_at,
                        payload=payload,
                    )
                indexed_by_key[key] = effective_record
                summary = self._summarize_indexed_record(effective_record)
                if summary is None:
                    records_by_key.pop(key, None)
                else:
                    records_by_key[key] = summary

            records = list(records_by_key.values())
            records.sort(
                key=lambda record: (
                    record[1].get("event_id", ""),
                    self._source_run_identity(record[1]),
                )
            )
            records.sort(key=lambda record: record[0], reverse=True)
            display_ids, daily_totals = self._daily_display_metadata(records)
            prepared = (records, indexed_by_key, display_ids, daily_totals)
            self._prepared_records_source = indexed_records
            self._prepared_records_vlm_fingerprint = vlm_fingerprint
            self._prepared_records_vlm_states = vlm_states
            self._prepared_records_cache = prepared
            return prepared

    def _vlm_state_overlays(
        self, indexed_records: list[IndexedRecord]
    ) -> tuple[dict[str, dict[str, dict[str, Any]]], tuple[tuple[str, int, int], ...]]:
        """Read compact VLM state without invalidating the large event index."""

        current_run_dir = self._trusted_event_run_dir(self.run_dir)
        if current_run_dir is None:
            return {}, ()
        run_dirs: dict[str, Path] = {current_run_dir.name: current_run_dir}
        for run_id in {record.source_run_id for record in indexed_records}:
            if run_id in run_dirs:
                continue
            historical = self._historical_run_dir(run_id)
            if historical is not None:
                run_dirs[run_id] = historical

        fingerprint: list[tuple[str, int, int]] = []
        state_paths: dict[str, Path] = {}
        for run_id, run_dir in sorted(run_dirs.items()):
            state_path = self._storage_for_run(run_dir).metadata_dir / DEFAULT_VLM_STATE_FILENAME
            state_paths[run_id] = state_path
            try:
                details = state_path.stat()
                signature = (run_id, details.st_mtime_ns, details.st_size)
            except OSError:
                signature = (run_id, -1, -1)
            fingerprint.append(signature)
        resolved_fingerprint = tuple(fingerprint)
        with self._event_cache_lock:
            if self._vlm_state_cache_fingerprint == resolved_fingerprint:
                return self._vlm_state_cache, resolved_fingerprint
            previous = {row[0]: row for row in (self._vlm_state_cache_fingerprint or ())}
            states: dict[str, dict[str, dict[str, Any]]] = {}
            for signature in resolved_fingerprint:
                run_id, modified_ns, _size = signature
                if modified_ns < 0:
                    continue
                if previous.get(run_id) == signature and run_id in self._vlm_state_cache:
                    states[run_id] = self._vlm_state_cache[run_id]
                    continue
                try:
                    states[run_id] = VLMReviewStateStore(state_paths[run_id], run_dir=run_dirs[run_id]).read()
                except VLMStateError:
                    continue
            # Replace the snapshot, never mutate a dictionary used by a reader.
            self._vlm_state_cache_fingerprint = resolved_fingerprint
            self._vlm_state_cache = states
        return states, resolved_fingerprint

    def _public_event_index_summary(self) -> dict[str, Any]:
        summary = self._event_index_summary
        return {
            "runs_indexed": summary.loaded_runs + summary.skipped_runs,
            "events_indexed": summary.total_records,
            "last_refresh_ms": self._event_index_last_refresh_ms,
            "changed_runs": summary.loaded_runs + summary.removed_runs,
            "fallback_active": summary.fallback_active,
            "last_error": summary.last_error or "",
        }

    def _get_event_index(self) -> EventIndex:
        if self._event_index is None:
            with self._event_index_lock:
                if self._event_index is None:
                    index_path = self._storage_for_run(self.run_dir).index_path
                    self._event_index = EventIndex(index_path)
        return self._event_index

    def _scan_run_records(self, run_dir: Path) -> list[IndexedRecord]:
        source_run_dir = self._trusted_event_run_dir(run_dir)
        if source_run_dir is None:
            return []
        dashboard_dir = self._storage_for_run(source_run_dir).metadata_dir
        loaded = self._read_json(dashboard_dir / "events.json", [])
        reviews = self._read_json(dashboard_dir / "event_reviews.json", {})
        if not isinstance(loaded, list):
            return []
        try:
            loaded = VLMReviewStateStore(
                dashboard_dir / DEFAULT_VLM_STATE_FILENAME, run_dir=source_run_dir
            ).merge_events(
                [item for item in loaded if isinstance(item, dict)]
            )
        except VLMStateError:
            # The sidecar is optional; retain the legacy events.json view if
            # it cannot be read safely.
            loaded = [item for item in loaded if isinstance(item, dict)]
        if not isinstance(reviews, dict):
            reviews = {}
        records: list[IndexedRecord] = []
        for item in loaded:
            if not isinstance(item, dict) or item.get("status") != "ready":
                continue
            # The frequently polled endpoint is a summary feed. Frame-level boxes
            # and other event details remain available from the overlay endpoint.
            # Select the small summary before recursively redacting it.  Running
            # the redactor over frame-level timelines that are discarded here
            # makes an index refresh take seconds on established deployments.
            event = self._public_value({
                key: item[key]
                for key in _EVENT_SUMMARY_KEYS
                if key in item
            })
            event.pop("review_category", None)
            if (
                event.get("vlm_filter_result") not in _VLM_FILTER_RESULTS
                or event.get("vlm_filter_evidence_revision")
                != VLM_EVIDENCE_REVISION
            ):
                event.pop("vlm_filter_result", None)
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not _SAFE_EVENT_ID.fullmatch(event_id):
                continue
            clip_path = self._safe_clip_path(f"{event_id}.mp4", run_dir=run_dir)
            overlay_path = self._safe_clip_path(f"{event_id}.json", run_dir=run_dir)
            if clip_path is None or overlay_path is None:
                continue
            occurred_at = self._beijing_time(event.get("occurred_at"))
            if occurred_at is None:
                continue
            event["review_result"] = self._review_result(reviews, event_id)
            review_category = self._review_category(reviews, event_id)
            if review_category is not None:
                event["review_category"] = review_category
            review_reason = self._review_reason(reviews, event_id)
            if review_reason is not None:
                event["review_reason"] = review_reason
            records.append(
                IndexedRecord(
                    source_run_id=source_run_dir.name,
                    event_id=event_id,
                    occurred_at=occurred_at,
                    payload=event,
                )
            )
        return records

    def _decorate_indexed_record(
        self, indexed_record: IndexedRecord
    ) -> tuple[datetime, dict[str, Any]] | None:
        summary = self._summarize_indexed_record(indexed_record)
        if summary is None:
            return None
        occurred_at, event = summary
        current_run_dir = self._trusted_event_run_dir(self.run_dir)
        if current_run_dir is None:
            return None
        is_historical = indexed_record.source_run_id != current_run_dir.name
        if is_historical:
            source_run_dir = self._historical_run_dir(indexed_record.source_run_id)
        else:
            source_run_dir = current_run_dir
        if source_run_dir is None:
            return None
        clip_path = self._safe_clip_path(
            f"{indexed_record.event_id}.mp4",
            run_dir=source_run_dir,
        )
        overlay_path = self._safe_clip_path(
            f"{indexed_record.event_id}.json",
            run_dir=source_run_dir,
        )
        if clip_path is None or overlay_path is None:
            return None
        encoded_id = quote(indexed_record.event_id, safe="")
        browser_proxy = self._safe_clip_path(
            f"{indexed_record.event_id}.browser.mp4",
            run_dir=source_run_dir,
        )
        clip_filename = (
            f"{encoded_id}.browser.mp4" if browser_proxy is not None else f"{encoded_id}.mp4"
        )
        if is_historical:
            encoded_run_id = quote(indexed_record.source_run_id, safe="")
            event["run_id"] = indexed_record.source_run_id
            event["is_historical"] = True
            event["clip_url"] = f"/runs/{encoded_run_id}/clips/{clip_filename}"
            event["overlay_url"] = f"/runs/{encoded_run_id}/clips/{encoded_id}.json"
            event["review_url"] = (
                f"/api/runs/{encoded_run_id}/events/{encoded_id}/review"
            )
        else:
            event["clip_url"] = f"/clips/{clip_filename}"
            event["overlay_url"] = f"/clips/{encoded_id}.json"
            event["review_url"] = f"/api/events/{encoded_id}/review"
        return occurred_at, event

    def _summarize_indexed_record(
        self, indexed_record: IndexedRecord
    ) -> tuple[datetime, dict[str, Any]] | None:
        if (
            not isinstance(indexed_record.source_run_id, str)
            or not isinstance(indexed_record.event_id, str)
            or not isinstance(indexed_record.payload, dict)
            or not _SAFE_EVENT_ID.fullmatch(indexed_record.event_id)
            or indexed_record.payload.get("event_id") != indexed_record.event_id
            or indexed_record.occurred_at.tzinfo is None
            or indexed_record.occurred_at.utcoffset() is None
        ):
            return None
        public_payload = self._public_value(indexed_record.payload)
        event = {
            key: public_payload[key]
            for key in _EVENT_SUMMARY_KEYS
            if key in public_payload
        }
        event.pop("review_category", None)
        if (
            event.get("vlm_filter_result") not in _VLM_FILTER_RESULTS
            or event.get("vlm_filter_evidence_revision")
            != VLM_EVIDENCE_REVISION
        ):
            event.pop("vlm_filter_result", None)
        event.pop("vlm_filter_evidence_revision", None)
        event["review_result"] = self._review_result(
            {
                indexed_record.event_id: {
                    "result": public_payload.get("review_result")
                }
            },
            indexed_record.event_id,
        )
        review_category = public_payload.get("review_category")
        if (
            event["review_result"] == "confirmed"
            and review_category in _CONFIRMED_REVIEW_CATEGORIES
        ):
            event["review_category"] = review_category
        review_reason = public_payload.get("review_reason")
        if review_reason in FALSE_POSITIVE_REASONS:
            event["review_reason"] = review_reason
        event["_source_run_id"] = indexed_record.source_run_id
        return indexed_record.occurred_at.astimezone(_BEIJING), event

    def _trusted_event_run_dir(self, run_dir: Path) -> Path | None:
        try:
            current = self.run_dir.resolve(strict=True)
            candidate = run_dir.resolve(strict=True)
            if candidate == current:
                return current if current.is_dir() else None
            if run_dir.is_symlink() or not _RUN_ID.fullmatch(candidate.name):
                return None
            historical = self._historical_run_dir(candidate.name)
            return candidate if historical == candidate else None
        except (OSError, RuntimeError):
            return None

    @staticmethod
    def _event_file_signature(dashboard_dir: Path, clips_dir: Path | None = None) -> RunSignature:
        def signature(path: Path) -> tuple[int, int]:
            try:
                details = path.stat()
                return details.st_mtime_ns, details.st_size
            except OSError:
                return -1, -1

        events_mtime_ns, events_size = signature(dashboard_dir / "events.json")
        reviews_mtime_ns, reviews_size = signature(
            dashboard_dir / "event_reviews.json"
        )
        clips_mtime_ns, _clips_size = signature(clips_dir if clips_dir is not None else dashboard_dir / "clips")
        return RunSignature(
            events_mtime_ns=events_mtime_ns,
            events_size=events_size,
            reviews_mtime_ns=reviews_mtime_ns,
            reviews_size=reviews_size,
            clips_mtime_ns=clips_mtime_ns,
        )

    @staticmethod
    def _parse_event_query(query_string: str) -> dict[str, Any]:
        if not query_string:
            query_string = "page=1"
        if _INVALID_PERCENT_ESCAPE.search(query_string):
            raise ValueError("malformed query string")
        try:
            parsed = parse_qs(
                query_string,
                keep_blank_values=True,
                strict_parsing=True,
                encoding="utf-8",
                errors="strict",
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("malformed query string") from error
        for key in (
            "month",
            "date",
            "hour",
            "page",
            "page_size",
            "risk",
            "include_false_positives",
            "exclude_vlm_filtered",
            "screen_capture_only",
            "review_range",
            "review_category",
        ):
            if len(parsed.get(key, [])) > 1:
                raise ValueError(f"invalid {key}")

        month = parsed.get("month", [None])[0]
        if month is not None:
            if not _MONTH.fullmatch(month):
                raise ValueError("invalid month")
            try:
                datetime.strptime(month, "%Y-%m")
            except ValueError as error:
                raise ValueError("invalid month") from error
        date = parsed.get("date", [None])[0]
        if date is not None:
            if not _DATE.fullmatch(date):
                raise ValueError("invalid date")
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError as error:
                raise ValueError("invalid date") from error
        hour = parsed.get("hour", [None])[0]
        if hour is not None and not _HOUR.fullmatch(hour):
            raise ValueError("invalid hour")

        page = DashboardApp._positive_integer(parsed, "page", 1)
        page_size = DashboardApp._positive_integer(parsed, "page_size", 20)
        if page_size > 100:
            raise ValueError("invalid page_size")
        risk = parsed.get("risk", [None])[0]
        risk_bands: tuple[str, ...] | None = None
        if risk is not None:
            risk_bands = tuple(filter(None, risk.split(",")))
            if (
                len(risk_bands) != len(set(risk_bands))
                or any(band not in _RISK_BANDS for band in risk_bands)
                or (risk and not risk_bands)
            ):
                raise ValueError("invalid risk")
        include_false_positives_value = parsed.get(
            "include_false_positives", [None]
        )[0]
        if include_false_positives_value is None:
            include_false_positives = True
        elif include_false_positives_value in {"0", "1"}:
            include_false_positives = include_false_positives_value == "1"
        else:
            raise ValueError("invalid include_false_positives")
        exclude_vlm_filtered_value = parsed.get(
            "exclude_vlm_filtered", [None]
        )[0]
        if exclude_vlm_filtered_value is None:
            exclude_vlm_filtered = False
        elif exclude_vlm_filtered_value in {"0", "1"}:
            exclude_vlm_filtered = exclude_vlm_filtered_value == "1"
        else:
            raise ValueError("invalid exclude_vlm_filtered")
        screen_capture_only_value = parsed.get("screen_capture_only", [None])[0]
        if screen_capture_only_value is None:
            screen_capture_only = False
        elif screen_capture_only_value in {"0", "1"}:
            screen_capture_only = screen_capture_only_value == "1"
        else:
            raise ValueError("invalid screen_capture_only")
        review_range = parsed.get("review_range", ["all"])[0]
        if review_range not in _REVIEW_RANGES:
            raise ValueError("invalid review_range")
        review_category = parsed.get("review_category", [None])[0]
        if review_category is not None and review_category not in _CONFIRMED_REVIEW_CATEGORIES:
            raise ValueError("invalid review_category")
        camera_values = parsed.get("camera", [])
        if len(camera_values) == 1 and camera_values[0] == "":
            camera_values = []
        if (
            len(camera_values) > 32
            or any(
                not value
                or len(value) > 128
                or any(ord(character) < 32 for character in value)
                for value in camera_values
            )
        ):
            raise ValueError("invalid camera")
        cameras = tuple(dict.fromkeys(camera_values))
        return {
            "month": month,
            "date": date,
            "hour": hour,
            "cameras": cameras,
            "page": page,
            "page_size": page_size,
            "risk_bands": risk_bands,
            "include_false_positives": include_false_positives,
            "exclude_vlm_filtered": exclude_vlm_filtered,
            "screen_capture_only": screen_capture_only,
            "review_range": review_range,
            "review_category": review_category,
        }

    @staticmethod
    def _positive_integer(
        parsed: Mapping[str, list[str]], key: str, default: int
    ) -> int:
        value = parsed.get(key, [None])[0]
        if value is None:
            return default
        if not _POSITIVE_INTEGER.fullmatch(value):
            raise ValueError(f"invalid {key}")
        return int(value)

    @staticmethod
    def _beijing_time(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            occurred_at = datetime.fromisoformat(
                f"{value[:-1]}+00:00" if value.endswith("Z") else value
            )
            if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
                return None
            return occurred_at.astimezone(_BEIJING)
        except (OverflowError, ValueError):
            return None

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    @classmethod
    def _risk_band(cls, event: Mapping[str, Any]) -> str | None:
        risk = event.get("risk_peak")
        if not cls._is_finite_number(risk):
            return None
        if risk >= 0.80:
            return "high"
        if risk >= 0.75:
            return "medium"
        return "low"

    @classmethod
    def _matches_risk_bands(
        cls, event: Mapping[str, Any], risk_bands: tuple[str, ...] | None
    ) -> bool:
        if risk_bands is None:
            return True
        if not risk_bands:
            return False
        risk_band = cls._risk_band(event)
        return risk_band is None or risk_band in risk_bands

    def _update_review(
        self, event_id: str, body: bytes, *, run_dir: Path | None = None
    ) -> HTTPResponse:
        target_run_dir = run_dir or self.run_dir
        dashboard_dir = self._storage_for_run(target_run_dir).metadata_dir
        event_reviews_path = dashboard_dir / "event_reviews.json"
        if not _SAFE_EVENT_ID.fullmatch(event_id):
            return self._error_response(404, "event not found")
        event = self._event(event_id, run_dir=target_run_dir)
        if event is None:
            return self._error_response(404, "event not found")
        clip_path = self._safe_clip_path(f"{event_id}.mp4", run_dir=target_run_dir)
        if (
            event.get("status") != "ready"
            or clip_path is None
            or self._safe_clip_path(f"{event_id}.json", run_dir=target_run_dir) is None
        ):
            return self._error_response(409, "event is not reviewable")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._error_response(400, "malformed JSON")
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, str) or result not in _REVIEW_RESULTS:
            return self._error_response(400, "invalid review result")
        reason: str | None = None
        category: str | None = None
        requested_category = payload.get("category") if isinstance(payload, dict) else None
        if result == "confirmed":
            if requested_category is not None:
                if (
                    not isinstance(requested_category, str)
                    or requested_category not in _CONFIRMED_REVIEW_CATEGORIES
                ):
                    return self._error_response(400, "invalid confirmed review category")
                category = requested_category
        elif requested_category is not None:
            return self._error_response(400, "review category requires confirmed result")
        if result == "false_positive":
            requested_reason = payload.get("reason") if isinstance(payload, dict) else None
            if requested_reason is None:
                reason = "unspecified"
            elif (
                not isinstance(requested_reason, str)
                or requested_reason not in FALSE_POSITIVE_REASONS
            ):
                return self._error_response(400, "invalid false-positive reason")
            else:
                reason = requested_reason

        with self._review_lock:
            reviews = self._read_json(event_reviews_path, {})
            if not isinstance(reviews, dict):
                reviews = {}
            archive_path = self._false_positive_archive_path(event_id, event, run_dir=target_run_dir)
            person_roi_archived = False
            fixed_template_created = False
            overlay_path = self._safe_clip_path(f"{event_id}.json", run_dir=target_run_dir)
            assert overlay_path is not None
            reviewed_at = datetime.now(timezone.utc).isoformat()
            display_event_id = self._review_display_id(target_run_dir, event_id, event)
            reviewed_time = self._beijing_time(reviewed_at)
            reviewed_date = (
                reviewed_time.strftime("%Y%m%d") if reviewed_time is not None else "unknown"
            )
            if result == "false_positive":
                try:
                    self._archive_raw_clip(clip_path, archive_path)
                    person_roi_archived = archive_person_roi(
                        run_dir=target_run_dir,
                        clip_path=clip_path,
                        overlay_path=overlay_path,
                        archive_clip_path=archive_path,
                        event=event,
                        review_reason=reason if reason != "unspecified" else None,
                        display_event_id=display_event_id,
                        reviewed_date=reviewed_date,
                    )
                    if reason == "fixed_phone":
                        fixed_template_created = create_fixed_template(
                            run_dir=target_run_dir,
                            event_id=event_id,
                            event=event,
                            clip_path=clip_path,
                            overlay_path=overlay_path,
                            reviewed_at=reviewed_at,
                        )
                    else:
                        remove_fixed_template(target_run_dir, event_id)
                except OSError:
                    return self._error_response(500, "false-positive archive failed")
            else:
                archive_path.unlink(missing_ok=True)
                remove_person_roi(archive_path, run_dir=target_run_dir)
                remove_fixed_template(target_run_dir, event_id)
            try:
                archive_reviewed_event(
                    run_dir=target_run_dir,
                    event_id=event_id,
                    result=result,
                    reason=reason,
                    category=category,
                    event=event,
                    clip_path=clip_path,
                    overlay_path=overlay_path,
                    reviewed_at=reviewed_at,
                )
            except OSError:
                return self._error_response(500, "review archive failed")
            reviews[event_id] = {
                "result": result,
                "updated_at": reviewed_at,
            }
            if result == "false_positive":
                reviews[event_id]["reason"] = reason
            elif result == "confirmed" and category is not None:
                reviews[event_id]["category"] = category
            try:
                if self._storage_for_run(target_run_dir).metadata_dir != dashboard_dir:
                    raise OSError("review metadata authority changed")
                self._atomic_write_json(event_reviews_path, reviews)
            except OSError:
                return self._error_response(500, "review save failed")
        response: dict[str, Any] = {
            "event_id": event_id,
            "review_result": result,
            "review_reason": reason,
            "review_category": category,
        }
        if result == "false_positive":
            response["archived"] = True
            response["person_roi_archived"] = person_roi_archived
            response["fixed_template_created"] = fixed_template_created
        return self._json_response(response, head_only=False)

    def _false_positive_archive_path(
        self, event_id: str, event: Mapping[str, Any], *, run_dir: Path | None = None
    ) -> Path:
        camera = event.get("camera") or event.get("camera_id") or "unknown"
        safe_camera = re.sub(r"[^A-Za-z0-9_.-]", "_", str(camera)) or "unknown"
        occurred_at = self._beijing_time(event.get("occurred_at"))
        event_date = occurred_at.strftime("%Y-%m-%d") if occurred_at else "unknown"
        return (run_dir or self.run_dir) / "reviewed_false_positives" / safe_camera / event_date / f"{event_id}.mp4"

    @staticmethod
    def _archive_raw_clip(source: Path, destination: Path) -> None:
        """Copy the browser-ready source clip without its box-overlay sidecar."""
        if destination.is_file():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _event(self, event_id: str, *, run_dir: Path | None = None) -> Mapping[str, Any] | None:
        loaded = self._read_json(self._storage_for_run(run_dir or self.run_dir).metadata_dir / "events.json", [])
        if not isinstance(loaded, list):
            return None
        return next(
            (
                item
                for item in reversed(loaded)
                if isinstance(item, dict) and item.get("event_id") == event_id
            ),
            None,
        )

    @staticmethod
    def _review_result(reviews: Mapping[str, Any], event_id: Any) -> str:
        review = reviews.get(event_id) if isinstance(event_id, str) else None
        if isinstance(review, dict) and review.get("result") in _REVIEW_RESULTS:
            return str(review["result"])
        return "pending"

    @staticmethod
    def _review_reason(reviews: Mapping[str, Any], event_id: Any) -> str | None:
        review = reviews.get(event_id) if isinstance(event_id, str) else None
        if (
            isinstance(review, dict)
            and review.get("result") == "false_positive"
            and review.get("reason") in FALSE_POSITIVE_REASONS
        ):
            return str(review["reason"])
        return None

    @staticmethod
    def _review_category(reviews: Mapping[str, Any], event_id: Any) -> str | None:
        review = reviews.get(event_id) if isinstance(event_id, str) else None
        if (
            isinstance(review, dict)
            and review.get("result") == "confirmed"
            and review.get("category") in _CONFIRMED_REVIEW_CATEGORIES
        ):
            return str(review["category"])
        return None

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def _error_response(
        cls, status: int, message: str, *, head_only: bool = False
    ) -> HTTPResponse:
        return cls._json_response(
            {"error": message}, head_only=head_only, status=status
        )

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return default

    def _public_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._public_value(item)
                for key, item in value.items()
                if str(key).lower() not in _PRIVATE_KEYS
            }
        if isinstance(value, list):
            return [self._public_value(item) for item in value]
        if isinstance(value, str):
            return self.redactor(value)
        return value

    def _storage_for_run(self, run_dir: Path) -> RunStorage:
        storage = resolve_run_storage(run_dir)
        known = self._storage_bindings.setdefault(storage.run_dir, storage)
        if known != storage:
            raise ValueError("dashboard metadata authority changed")
        return storage

    def _historical_run_dir(self, run_id: str) -> Path | None:
        if not _RUN_ID.fullmatch(run_id):
            return None
        try:
            root = self.run_dir.resolve().parent
            candidate = root / run_id
            if candidate.is_symlink():
                return None
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_dir():
                return None
            return resolved
        except (OSError, RuntimeError):
            return None

    def _safe_clip_path(self, name: str, *, run_dir: Path | None = None) -> Path | None:
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "%" in name
            or name != Path(name).name
        ):
            return None
        try:
            target_run_dir = run_dir or self.run_dir
            dashboard_root = (target_run_dir / "dashboard").resolve()
            clips_root = (target_run_dir / "dashboard" / "clips").resolve()
            candidate = clips_root / name
            if candidate.is_symlink():
                return None
            resolved = candidate.resolve(strict=True)
            if not clips_root.is_relative_to(dashboard_root):
                return None
            if not resolved.is_relative_to(clips_root) or not resolved.is_file():
                return None
            return resolved
        except (OSError, RuntimeError):
            return None

    @staticmethod
    def _json_response(
        value: Any, *, head_only: bool, status: int = 200
    ) -> HTTPResponse:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        return HTTPResponse(
            status,
            {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
            },
            b"" if head_only else body,
        )
