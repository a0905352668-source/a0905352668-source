from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import pytest

from live_operator import mediamtx
from live_operator.mediamtx import MediaMTXProcess, render_mediamtx_config


@dataclass(frozen=True)
class Camera:
    relay: str
    url: str

    def rtsp_url(self) -> str:
        return self.url


def cameras(password: str = "p@ss:word") -> list[Camera]:
    return [
        Camera(
            relay=f"camera{index:02d}",
            url=f"rtsp://operator:{password}@192.0.2.{index}/Streaming/Channels/101",
        )
        for index in range(1, 8)
    ]


def test_render_writes_private_seven_path_relay_and_recording_config(tmp_path: Path) -> None:
    config_path = render_mediamtx_config(cameras(), tmp_path)

    assert config_path == tmp_path / "private" / "mediamtx.yml"
    if os.name != "nt":
        assert os.stat(config_path).st_mode & 0o777 == 0o600

    document = json.loads(config_path.read_text(encoding="utf-8"))
    assert document["api"] is True
    assert document["apiAddress"] == "127.0.0.1:9997"
    assert document["rtspAddress"] == "127.0.0.1:8554"
    assert set(document["paths"]) == {f"camera{index:02d}" for index in range(1, 8)}

    for index, (name, path) in enumerate(document["paths"].items(), start=1):
        assert name == f"camera{index:02d}"
        assert path == {
            "source": cameras()[index - 1].rtsp_url(),
            "sourceOnDemand": False,
            "rtspTransport": "tcp",
            "record": True,
            "recordPath": str(
                tmp_path
                / "recordings"
                / "%path"
                / "%Y-%m-%d_%H-%M-%S-%f"
            ),
            "recordFormat": "fmp4",
            "recordPartDuration": "1s",
            "recordSegmentDuration": "2s",
            "recordDeleteAfter": "2h",
        }


def test_render_accepts_config_object_with_cameras_attribute(tmp_path: Path) -> None:
    class Config:
        def __init__(self) -> None:
            self.cameras = cameras()

    assert render_mediamtx_config(Config(), tmp_path).is_file()


def test_render_accepts_partial_inventory_and_rejects_duplicate_relays(tmp_path: Path) -> None:
    partial = render_mediamtx_config(cameras()[:-1], tmp_path)
    assert set(json.loads(partial.read_text(encoding="utf-8"))["paths"]) == {
        f"camera{index:02d}" for index in range(1, 7)
    }
    duplicated = cameras()
    duplicated[-1] = Camera("camera01", duplicated[-1].url)
    with pytest.raises(ValueError, match="unique"):
        render_mediamtx_config(duplicated, tmp_path)


class FakeProcess:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self.stdout = iter(lines or [])
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_process_start_logs_only_redacted_output_and_stop_terminates(tmp_path: Path) -> None:
    config_path = render_mediamtx_config(cameras(), tmp_path)
    fake = FakeProcess([b"pull rtsp://operator:p@ss:word@192.0.2.1/live\n"])
    calls: list[tuple[list[str], dict[str, object]]] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        calls.append((command, kwargs))
        return fake

    process = MediaMTXProcess(
        config_path,
        tmp_path,
        cameras(),
        popen_factory=popen,
        redact=lambda text: text.replace("p@ss:word", "***"),
    )
    process.start()
    process.stop()

    assert calls[0][0] == ["mediamtx", str(config_path)]
    assert fake.terminated is True
    log_text = (tmp_path / "logs" / "mediamtx.log").read_text(encoding="utf-8")
    assert "p@ss:word" not in log_text
    assert "***" in log_text


def test_process_automatically_redacts_configured_rtsp_urls(tmp_path: Path) -> None:
    configured_cameras = cameras(password="encoded%40secret")
    config_path = render_mediamtx_config(configured_cameras, tmp_path)
    source = configured_cameras[0].rtsp_url()
    userinfo = source.removeprefix("rtsp://").split("@", 1)[0]
    encoded_password = userinfo.split(":", 1)[1]
    raw_password = unquote(encoded_password)
    fake = FakeProcess(
        [
            f"raw={raw_password}\n".encode(),
            f"encoded={encoded_password}\n".encode(),
            f"userinfo={userinfo}\n".encode(),
        ]
    )
    process = MediaMTXProcess(
        config_path,
        tmp_path,
        configured_cameras,
        popen_factory=lambda _command, **_kwargs: fake,
    )

    process.start()
    process.stop()

    log_text = (tmp_path / "logs" / "mediamtx.log").read_text(encoding="utf-8")
    assert raw_password not in log_text
    assert encoded_password not in log_text
    assert userinfo not in log_text
    assert log_text.count("[REDACTED]") == 3


def test_ready_requires_live_process_api_and_all_seven_local_relays(tmp_path: Path) -> None:
    config_path = render_mediamtx_config(cameras(), tmp_path)
    process = MediaMTXProcess(
        config_path,
        tmp_path,
        cameras(),
        api_probe=lambda _timeout: True,
        stream_probe=lambda url, _timeout: not url.endswith("camera07"),
    )
    process._process = FakeProcess()  # Inject a running child without spawning it.

    assert process.ready(timeout=1) is False

    probed: list[str] = []
    process._stream_probe = lambda url, _timeout: probed.append(url) is None or True
    assert process.ready(timeout=1) is True
    assert probed == [f"rtsp://127.0.0.1:8554/camera{index:02d}" for index in range(1, 8)]


def test_ready_times_out_when_api_never_becomes_ready(tmp_path: Path) -> None:
    config_path = render_mediamtx_config(cameras(), tmp_path)
    process = MediaMTXProcess(
        config_path,
        tmp_path,
        cameras(),
        api_probe=lambda _timeout: False,
        stream_probe=lambda _url, _timeout: True,
        sleep=lambda _seconds: None,
    )
    process._process = FakeProcess()

    assert process.ready(timeout=0.01, poll_interval=0.001) is False


@pytest.mark.parametrize(
    "error",
    [subprocess.TimeoutExpired(["ffprobe"], 0.01), OSError("ffprobe unavailable")],
)
def test_default_stream_probe_timeout_and_errors_are_not_ready(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(mediamtx.subprocess, "run", fail)

    assert mediamtx._default_stream_probe("rtsp://127.0.0.1:8554/camera01", 0.01) is False


def test_ready_enforces_one_deadline_across_sequential_probes(tmp_path: Path) -> None:
    class Clock:
        value = 0.0

        def now(self) -> float:
            return self.value

    clock = Clock()
    observed_timeouts: list[float] = []

    def api_probe(timeout: float) -> bool:
        observed_timeouts.append(timeout)
        clock.value += 0.4
        return True

    def stream_probe(_url: str, timeout: float) -> bool:
        observed_timeouts.append(timeout)
        clock.value += 0.4
        return True

    process = MediaMTXProcess(
        render_mediamtx_config(cameras(), tmp_path),
        tmp_path,
        cameras(),
        api_probe=api_probe,
        stream_probe=stream_probe,
        monotonic=clock.now,
        sleep=lambda _seconds: None,
    )
    process._process = FakeProcess()

    assert process.ready(timeout=1.0) is False
    assert len(observed_timeouts) == 3
    assert observed_timeouts == pytest.approx([1.0, 0.6, 0.2])


def test_ready_returns_false_when_child_exits_during_probes(tmp_path: Path) -> None:
    class ExitingProcess(FakeProcess):
        polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls == 1 else 1

    process = MediaMTXProcess(
        render_mediamtx_config(cameras(), tmp_path),
        tmp_path,
        cameras(),
        api_probe=lambda _timeout: True,
        stream_probe=lambda _url, _timeout: True,
    )
    process._process = ExitingProcess()

    assert process.ready(timeout=1) is False


def test_ready_converts_probe_exception_to_false(tmp_path: Path) -> None:
    def failing_probe(_url: str, _timeout: float) -> bool:
        raise subprocess.TimeoutExpired(["ffprobe"], 0.01)

    process = MediaMTXProcess(
        render_mediamtx_config(cameras(), tmp_path),
        tmp_path,
        cameras(),
        api_probe=lambda _timeout: True,
        stream_probe=failing_probe,
        sleep=lambda _seconds: None,
    )
    process._process = FakeProcess()

    assert process.ready(timeout=0.01) is False


def test_installer_is_checksum_verified_idempotent_and_does_not_configure_services() -> None:
    installer = Path(__file__).parents[2] / "scripts" / "install_mediamtx_50p2.sh"
    text = installer.read_text(encoding="utf-8")

    assert "sha256sum -c" in text
    assert "if [[ -x" in text
    assert "mktemp" in text
    assert "mv" in text
    assert "systemctl" not in text
    assert "iptables" not in text
    assert "ufw" not in text
