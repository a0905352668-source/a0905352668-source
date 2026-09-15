from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen


def _camera_list(config_or_cameras: object) -> list[object]:
    value = getattr(config_or_cameras, "cameras", config_or_cameras)
    if isinstance(value, Mapping):
        value = value.values()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("config must provide a cameras iterable")
    return [camera for camera in value if bool(getattr(camera, "enabled", True))]


def _camera_value(camera: object, *names: str) -> Any:
    if isinstance(camera, Mapping):
        for name in names:
            if name in camera:
                return camera[name]
    for name in names:
        if hasattr(camera, name):
            return getattr(camera, name)
    raise ValueError(f"camera is missing one of: {', '.join(names)}")


def _relay(camera: object) -> str:
    return str(_camera_value(camera, "relay", "name", "path"))


def _source(camera: object) -> str:
    value = _camera_value(camera, "rtsp_url", "source", "url")
    return str(value() if callable(value) else value)


def _validate_cameras(config_or_cameras: object) -> list[object]:
    result = _camera_list(config_or_cameras)
    relays = [_relay(camera) for camera in result]
    if not relays:
        raise ValueError("at least one enabled camera is required")
    if len(set(relays)) != len(relays):
        raise ValueError("camera relays must be unique")
    return result


def _credential_redactor(
    cameras: list[object], primary: Callable[[str], str] | None
) -> Callable[[str], str]:
    secrets: set[str] = set()
    for camera in cameras:
        source = _source(camera)
        secrets.add(source)
        parsed = urlsplit(source)
        userinfo, separator, _host = parsed.netloc.rpartition("@")
        if separator:
            secrets.add(userinfo)
            encoded_password = userinfo.partition(":")[2]
            if encoded_password:
                secrets.add(encoded_password)
                secrets.add(unquote(encoded_password))

    ordered = sorted((value for value in secrets if value), key=len, reverse=True)

    def redact(text: str) -> str:
        result = primary(str(text)) if primary is not None else str(text)
        for secret in ordered:
            result = result.replace(secret, "[REDACTED]")
        return result

    return redact


def render_mediamtx_config(config_or_cameras: object, run_dir: str | os.PathLike[str]) -> Path:
    """Write a private MediaMTX relay configuration for one timestamped run."""

    cameras = _validate_cameras(config_or_cameras)
    run_path = Path(run_dir)
    private_dir = run_path / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = run_path / "recordings"

    paths: dict[str, dict[str, object]] = {}
    for camera in cameras:
        relay = _relay(camera)
        paths[relay] = {
            "source": _source(camera),
            "sourceOnDemand": False,
            # Camera main streams are high-bitrate 1440p HEVC.  UDP loss can
            # drop an entire reference picture while leaving the fMP4 file
            # structurally valid, which later appears as grey/mosaic frames.
            # Interleaved RTSP/TCP preserves packet ordering and retransmits
            # loss before MediaMTX records or relays the stream.
            "rtspTransport": "tcp",
            "record": True,
            "recordPath": str(recordings_dir / "%path" / "%Y-%m-%d_%H-%M-%S-%f"),
            "recordFormat": "fmp4",
            "recordPartDuration": "1s",
            "recordSegmentDuration": "2s",
            "recordDeleteAfter": "2h",
        }

    document = {
        "logLevel": "info",
        "api": True,
        "apiAddress": "127.0.0.1:9997",
        "rtspAddress": "127.0.0.1:8554",
        "paths": paths,
    }
    destination = private_dir / "mediamtx.yml"
    temporary = private_dir / ".mediamtx.yml.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(document, handle, indent=2)
            handle.write("\n")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    os.chmod(destination, 0o600)
    return destination


def _default_api_probe(timeout: float) -> bool:
    if timeout <= 0:
        return False
    try:
        with urlopen(
            "http://127.0.0.1:9997/v3/paths/list", timeout=timeout
        ) as response:
            return 200 <= response.status < 300
    except (OSError, TimeoutError):
        return False


def _default_stream_probe(url: str, timeout: float) -> bool:
    if timeout <= 0:
        return False
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


class MediaMTXProcess:
    """Own a MediaMTX child process and verify its local relays before use."""

    def __init__(
        self,
        config_path: str | os.PathLike[str],
        run_dir: str | os.PathLike[str],
        config_or_cameras: object,
        *,
        binary: str = "mediamtx",
        popen_factory: Callable[..., Any] = subprocess.Popen,
        api_probe: Callable[[float], bool] = _default_api_probe,
        stream_probe: Callable[[str, float], bool] = _default_stream_probe,
        redact: Callable[[str], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config_path = Path(config_path)
        self.run_dir = Path(run_dir)
        self.cameras = _validate_cameras(config_or_cameras)
        self.binary = binary
        self._popen_factory = popen_factory
        self._api_probe = api_probe
        self._stream_probe = stream_probe
        configured_redactor = redact or getattr(config_or_cameras, "redact_text", None)
        self._redact = _credential_redactor(self.cameras, configured_redactor)
        self._sleep = sleep
        self._monotonic = monotonic
        self._process: Any | None = None
        self._log_file: TextIO | None = None
        self._log_thread: threading.Thread | None = None

    def start(self) -> "MediaMTXProcess":
        if self._process is not None and self._process.poll() is None:
            return self
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = (log_dir / "mediamtx.log").open("a", encoding="utf-8")
        self._process = self._popen_factory(
            [self.binary, str(self.config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._log_thread = threading.Thread(target=self._copy_redacted_log, daemon=True)
        self._log_thread.start()
        return self

    def _copy_redacted_log(self) -> None:
        process = self._process
        log_file = self._log_file
        if process is None or process.stdout is None or log_file is None:
            return
        for raw_line in process.stdout:
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = str(raw_line)
            log_file.write(self._redact(line))
            log_file.flush()

    def ready(self, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
        deadline = self._monotonic() + max(timeout, 0)
        urls = [f"rtsp://127.0.0.1:8554/{_relay(camera)}" for camera in self.cameras]
        while True:
            if self._process is None or self._process.poll() is not None:
                return False
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return False
            try:
                api_ready = self._api_probe(remaining)
            except Exception:
                api_ready = False

            streams_ready = api_ready
            if api_ready:
                for url in urls:
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        return False
                    try:
                        if not self._stream_probe(url, remaining):
                            streams_ready = False
                            break
                    except Exception:
                        streams_ready = False
                        break

            if streams_ready:
                within_deadline = self._monotonic() <= deadline
                child_alive = self._process.poll() is None
                return within_deadline and child_alive

            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return False
            self._sleep(min(poll_interval, remaining))

    def stop(self, timeout: float = 10.0) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
        if self._log_thread is not None:
            self._log_thread.join(timeout=timeout)
        if self._log_file is not None:
            self._log_file.close()
        self._log_thread = None
        self._log_file = None
