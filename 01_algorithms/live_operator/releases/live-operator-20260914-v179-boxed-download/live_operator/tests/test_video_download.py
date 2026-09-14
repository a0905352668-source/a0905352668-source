import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from live_operator.dashboard import DashboardApp


def fixture_clip(run: Path) -> Path:
    clips = run / "dashboard" / "clips"
    clips.mkdir(parents=True)
    source = clips / "event-001.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "color=black:s=160x120:r=10:d=2", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(source),
    ], check=True, timeout=15)
    (clips / "event-001.json").write_text(json.dumps({
        "overlay": {"bbox_timeline": [
            {"time_sec": 0, "frame_width": 320, "frame_height": 240,
             "boxes": [
                 {"label": "person", "bbox": [40, 40, 160, 160]},
                 {"label": "screen", "bbox": [0, 0, 20, 20]},
             ], "phone_boxes": [{"box": [200, 100, 240, 140]}]},
            {"time_sec": 0.6, "frame_width": 320, "frame_height": 240,
             "alarm": False, "bbox": [100, 40, 180, 80]},
        ]},
    }))
    return source


def ready(app: DashboardApp, url: str) -> dict:
    response = app.handle("POST", url)
    assert response.status in {200, 202}, response.body
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = app.handle("GET", url)
        payload = json.loads(response.body)
        if payload.get("state") == "ready":
            return payload
        assert payload.get("state") in {"queued", "running"}, payload
        time.sleep(0.02)
    pytest.fail("CPU export did not finish")


def frame(source: Path, seconds: float) -> bytes:
    return subprocess.run([
        "ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(source),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], check=True, capture_output=True, timeout=15).stdout


def pixel(rgb: bytes, x: int, y: int) -> tuple[int, int, int]:
    offset = (y * 160 + x) * 3
    return tuple(rgb[offset:offset + 3])


@pytest.mark.parametrize("historical", [False, True])
def test_download_burns_scaled_person_phone_boxes_without_screen_or_stale_boxes(
    tmp_path: Path, historical: bool,
) -> None:
    current = tmp_path / "live_20260914_070000"
    run = tmp_path / "live_20260913_070000" if historical else current
    current.mkdir(exist_ok=True)
    source = fixture_clip(run)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    app = DashboardApp(current)
    url = (f"/api/runs/{run.name}/events/event-001/download" if historical
           else "/api/events/event-001/download")
    payload = ready(app, url)
    response = app.handle("GET", payload["download_url"])
    assert response.status == 200
    assert response.headers["Content-Type"] == "video/mp4"
    assert response.headers["Content-Disposition"] == 'attachment; filename="event-001-boxed.mp4"'
    exported = response.body.path
    rgb = frame(exported, 0.2)
    r, g, b = pixel(rgb, 20, 40)
    assert r > 130 and r > g * 1.5 and r > b * 1.5
    r, g, b = pixel(rgb, 100, 60)
    assert r > 130 and g > 70 and b < 80
    assert max(pixel(rgb, 0, 5)) < 35  # screen outline must not be burned in
    assert max(pixel(rgb, 30, 15)) < 35  # no person text label above box
    rgb = frame(exported, 0.8)
    assert max(pixel(rgb, 20, 40)) < 35  # newer sample clears previous boxes
    assert pixel(rgb, 50, 30)[0] > 130  # non-alarm person remains a person box
    assert max(frame(exported, 1.6)) < 35  # age >0.75s clears all annotations
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    repeated = ready(app, url)
    assert repeated["download_url"] == payload["download_url"]
    partial = app.handle("GET", payload["download_url"], {"Range": "bytes=0-7"})
    assert partial.status == 206
    assert len(b"".join(partial.iter_body())) == 8
    assert "Content-Disposition" not in app.handle("GET", f"/clips/{source.name}").headers
    assert app.handle("HEAD", payload["download_url"]).body == b""


@pytest.mark.parametrize("path", [
    "/api/events/%2e%2e/download",
    "/api/events/event%2f001/download",
    "/api/runs/%2e%2e/events/event-001/download",
    "/api/events/missing/download",
])
def test_download_rejects_unsafe_or_missing_media(tmp_path: Path, path: str) -> None:
    app = DashboardApp(tmp_path)
    assert app.handle("POST", path).status == 404


def test_get_download_status_does_not_start_cpu_export(tmp_path: Path) -> None:
    fixture_clip(tmp_path)
    app = DashboardApp(tmp_path)
    response = app.handle("GET", "/api/events/event-001/download")
    assert response.status == 200
    assert json.loads(response.body)["state"] == "idle"
    assert not list(tmp_path.rglob("*-boxed.mp4"))


def test_export_cache_rejects_symlink_outside_clips(tmp_path: Path) -> None:
    source = fixture_clip(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (source.parent / "boxed_downloads").symlink_to(outside, target_is_directory=True)
    response = DashboardApp(tmp_path).handle("POST", "/api/events/event-001/download")
    assert response.status == 409
    assert list(outside.iterdir()) == []


def test_export_queue_rejects_fifth_job_and_reuses_existing_job(tmp_path: Path, monkeypatch) -> None:
    from live_operator import video_download
    source = fixture_clip(tmp_path)
    overlay = source.with_suffix(".json")
    renderer = video_download.render_boxed_clip
    entered, release = threading.Event(), threading.Event()

    def held_renderer(*args):
        entered.set()
        assert release.wait(10)
        renderer(*args)

    monkeypatch.setattr(video_download, "render_boxed_clip", held_renderer)
    downloads = video_download.VideoDownloads()
    jobs = []
    try:
        state, first = downloads.status(source, overlay, start=True)
        assert entered.wait(2)
        assert downloads.status(source, overlay, start=True) == ("running", first)
        for index in range(1, 5):
            another = source.with_name(f"event-{index + 1:03d}.mp4")
            another.write_bytes(source.read_bytes())
            if index < 4:
                assert downloads.status(another, overlay, start=True)[0] == "queued"
            else:
                with pytest.raises(video_download.ExportBusy):
                    downloads.status(another, overlay, start=True)
        jobs = list(downloads._jobs.values())
    finally:
        release.set()
        for job in jobs:
            job.result(timeout=15)


def test_bad_overlay_does_not_publish_partial_mp4_and_allows_retry(tmp_path: Path) -> None:
    source = fixture_clip(tmp_path)
    overlay = source.with_suffix(".json")
    original = overlay.read_text()
    overlay.write_text("{invalid")
    app = DashboardApp(tmp_path)
    url = "/api/events/event-001/download"
    assert app.handle("POST", url).status == 202
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        payload = json.loads(app.handle("GET", url).body)
        if payload["state"] == "error":
            break
        time.sleep(0.02)
    assert payload["state"] == "error"
    assert app.handle("GET", url + "/file").status == 409
    assert list(source.parent.glob("boxed_downloads/*.mp4")) == []
    overlay.write_text(original)
    assert ready(app, url)["state"] == "ready"


def test_historical_dashboard_symlink_cannot_export_outside_run(tmp_path: Path) -> None:
    current = tmp_path / "live_20260914_070000"
    current.mkdir()
    historical = tmp_path / "live_20260913_070000"
    historical.mkdir()
    outside = tmp_path / "outside"
    fixture_clip(outside)
    (historical / "dashboard").symlink_to(outside / "dashboard", target_is_directory=True)
    app = DashboardApp(current)
    response = app.handle("POST", f"/api/runs/{historical.name}/events/event-001/download")
    assert response.status == 409
    assert not list(outside.rglob("*-boxed.mp4"))


@pytest.mark.parametrize("change", ["cache_symlink", "disk_full", "source_updated"])
def test_worker_rechecks_queued_source_cache_and_disk(tmp_path: Path, monkeypatch, change: str) -> None:
    from live_operator import video_download
    source = fixture_clip(tmp_path)
    entered, release = threading.Event(), threading.Event()
    renderer = video_download.render_boxed_clip

    def held_renderer(*args):
        entered.set()
        assert release.wait(5)
        renderer(*args)

    monkeypatch.setattr(video_download, "render_boxed_clip", held_renderer)
    downloads = video_download.VideoDownloads()
    state, output = downloads.status(source, source.with_suffix(".json"), start=True)
    assert entered.wait(2)
    outside = tmp_path / "outside"
    outside.mkdir()
    if change == "cache_symlink":
        output.parent.rename(output.parent.with_name("retired-cache"))
        output.parent.symlink_to(outside, target_is_directory=True)
    elif change == "disk_full":
        from collections import namedtuple
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr("shutil.disk_usage", lambda path: usage(100 * 1024**3, 99 * 1024**3, 1024**3))
    else:
        source.write_bytes(source.read_bytes() + b"changed")
    release.set()
    with pytest.raises(ValueError):
        downloads._jobs[output].result(timeout=5)
    assert list(outside.iterdir()) == []


def test_export_declines_low_disk_space_without_starting_job(tmp_path: Path, monkeypatch) -> None:
    from collections import namedtuple
    from live_operator import video_download
    source = fixture_clip(tmp_path)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("shutil.disk_usage", lambda path: usage(100 * 1024**3, 99 * 1024**3, 1024**3))
    response = DashboardApp(tmp_path).handle("POST", "/api/events/event-001/download")
    assert response.status == 409
    assert not list(source.parent.rglob("*-boxed.mp4"))


def test_negative_sample_crossing_zero_still_draws_clipped_person(tmp_path: Path) -> None:
    source = fixture_clip(tmp_path)
    source.with_suffix(".json").write_text(json.dumps({"overlay": {"bbox_timeline": [
        {"time_sec": -0.1, "frame_width": 160, "frame_height": 120,
         "bbox": [-10, 20, 80, 80]},
    ]}}))
    app = DashboardApp(tmp_path)
    payload = ready(app, "/api/events/event-001/download")
    exported = app.handle("GET", payload["download_url"]).body.path
    assert pixel(frame(exported, 0.2), 0, 40)[0] > 130
    assert max(frame(exported, 0.8)) < 35


def test_cache_budget_declines_new_export_without_deleting_existing(tmp_path: Path) -> None:
    source = fixture_clip(tmp_path)
    cache = source.parent / "boxed_downloads"
    cache.mkdir()
    existing = cache / "existing-boxed.mp4"
    with existing.open("wb") as handle:
        handle.truncate(2 * 1024**3)
    response = DashboardApp(tmp_path).handle("POST", "/api/events/event-001/download")
    assert response.status == 409
    assert existing.stat().st_size == 2 * 1024**3


def test_annotation_expansion_limit_fails_without_publishing_output(tmp_path: Path) -> None:
    source = fixture_clip(tmp_path)
    source.with_suffix(".json").write_text(json.dumps({"overlay": {"bbox_timeline": [
        {"time_sec": 0, "boxes": [{"label": "person", "bbox": [20, 20, 80, 80]}] * 65},
    ]}}))
    from live_operator.video_download import render_boxed_clip
    output = source.parent / "out.mp4"
    with pytest.raises(ValueError):
        render_boxed_clip(source, source.with_suffix(".json"), output)
    assert not output.exists()


def test_total_annotation_limit_rejects_large_expansion(tmp_path: Path) -> None:
    source = fixture_clip(tmp_path)
    source.with_suffix(".json").write_text(json.dumps({"overlay": {"bbox_timeline": [
        {"time_sec": index / 1000, "boxes": [{"label": "person", "bbox": [20, 20, 80, 80]}] * 64}
        for index in range(400)
    ]}}))
    from live_operator.video_download import render_boxed_clip
    output = source.parent / "out.mp4"
    with pytest.raises(ValueError):
        render_boxed_clip(source, source.with_suffix(".json"), output)
    assert not output.exists()
