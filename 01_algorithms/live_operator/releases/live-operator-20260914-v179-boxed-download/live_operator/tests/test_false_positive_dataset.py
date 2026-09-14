import json
import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from live_operator import false_positive_dataset
from live_operator.false_positive_dataset import _extract_crop, archive_person_roi


def test_empty_negative_prelabel_is_saved_and_operator_edits_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    clips = run_dir / "dashboard" / "clips"
    clips.mkdir(parents=True)
    event_id = "event-negative"
    clip = clips / f"{event_id}.mp4"
    clip.write_bytes(b"clip")
    overlay = clips / f"{event_id}.json"
    overlay.write_text(
        json.dumps(
            {
                "overlay": {
                    "bbox_timeline": [
                        {
                            "time_sec": 5.0,
                            "frame_index": 40,
                            "bbox": [40, 20, 280, 220],
                            "roi": [32, 16, 288, 224],
                            "frame_width": 320,
                            "frame_height": 240,
                            "phone_boxes": [],
                            "track_id": "7",
                            "risk_score": 0.9,
                            "alarm": True,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    archive_clip = run_dir / "reviewed_false_positives" / "camera01" / "2026-07-20" / f"{event_id}.mp4"

    def fake_extract(_ffmpeg, _clip, destination, **_kwargs):
        destination.write_bytes(b"jpeg")

    monkeypatch.setattr(false_positive_dataset, "_extract_crop", fake_extract)
    monkeypatch.setattr(false_positive_dataset.shutil, "which", lambda _name: "ffmpeg")
    arguments = {
        "run_dir": run_dir,
        "clip_path": clip,
        "overlay_path": overlay,
        "archive_clip_path": archive_clip,
        "event": {
            "event_id": event_id,
            "camera": "camera01",
            "person_id": "7",
        },
    }

    assert archive_person_roi(**arguments) is True
    label_path = run_dir.parent / "reviewed_false_positive_labelme" / f"{event_id}__roi-01.json"
    label = json.loads(label_path.read_text(encoding="utf-8"))
    assert label["shapes"] == []
    label_path.write_text('{"operator_edited": true}\n', encoding="utf-8")

    assert archive_person_roi(**arguments) is True
    assert json.loads(label_path.read_text(encoding="utf-8")) == {"operator_edited": True}


def test_model_misdetect_reason_writes_empty_shapes_instead_of_false_phone_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    clips = run_dir / "dashboard" / "clips"
    clips.mkdir(parents=True)
    clip = clips / "model-misdetect.mp4"
    clip.write_bytes(b"clip")
    overlay = clips / "model-misdetect.json"
    overlay.write_text(
        json.dumps(
            {
                "overlay": {
                    "bbox_timeline": [
                        {
                            "time_sec": 1.0,
                            "frame_index": 8,
                            "bbox": [20, 20, 220, 220],
                            "roi": [20, 20, 220, 220],
                            "frame_width": 320,
                            "frame_height": 240,
                            "phone_boxes": [
                                {"box": [80, 90, 120, 150], "phone_score": 0.91}
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_extract(_ffmpeg, _clip, destination, **_kwargs):
        destination.write_bytes(b"jpeg")

    monkeypatch.setattr(false_positive_dataset, "_extract_crop", fake_extract)
    monkeypatch.setattr(false_positive_dataset.shutil, "which", lambda _name: "ffmpeg")
    archived = archive_person_roi(
        run_dir=run_dir,
        clip_path=clip,
        overlay_path=overlay,
        archive_clip_path=run_dir / "reviewed_false_positives" / "model-misdetect.mp4",
        event={"event_id": "model-misdetect", "camera": "camera01"},
        review_reason="model_misdetect",
    )

    assert archived is True
    label_path = (
        run_dir.parent
        / "reviewed_false_positive_labelme"
        / "model-misdetect__roi-01.json"
    )
    label = json.loads(label_path.read_text(encoding="utf-8"))
    assert label["shapes"] == []
    assert label["flags"]["reason_model_misdetect"] is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_archive_person_roi_with_real_ffmpeg(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    run_dir = tmp_path / "run"
    clips = run_dir / "dashboard" / "clips"
    clips.mkdir(parents=True)
    event_id = "event-real-ffmpeg"
    clip = clips / f"{event_id}.mp4"
    created = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=1",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(clip),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    overlay = clips / f"{event_id}.json"
    overlay.write_text(
        json.dumps(
            {
                "window_start": "2026-07-20T06:00:00+00:00",
                "window_end": "2026-07-20T06:00:01+00:00",
                "overlay": {
                    "bbox_timeline": [
                        {
                            "time_sec": 0.5,
                            "frame_index": 4,
                            "bbox": [40, 20, 280, 220],
                            "roi": [32, 16, 288, 224],
                            "frame_width": 320,
                            "frame_height": 240,
                            "phone_boxes": [
                                {"box": [120, 80, 160, 140], "phone_score": 0.82}
                            ],
                            "track_id": "7",
                            "risk_score": 0.9,
                            "alarm": True,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    archive_clip = run_dir / "reviewed_false_positives" / "camera01" / "2026-07-20" / f"{event_id}.mp4"

    archived = archive_person_roi(
        run_dir=run_dir,
        clip_path=clip,
        overlay_path=overlay,
        archive_clip_path=archive_clip,
        event={
            "event_id": event_id,
            "camera": "camera01",
            "person_id": "7",
            "bbox_timeline": [],
        },
        ffmpeg_bin=ffmpeg,
    )

    assert archived is True
    roi_dir = archive_clip.parent / "person_roi"
    image = roi_dir / f"{event_id}.jpg"
    assert image.read_bytes().startswith(b"\xff\xd8\xff")
    label = json.loads((roi_dir / f"{event_id}.json").read_text(encoding="utf-8"))
    assert (label["imageWidth"], label["imageHeight"]) == (256, 208)
    assert len(label["shapes"]) == 1


def test_extract_crop_clamps_a_timeline_timestamp_after_clip_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "event.mp4"
    destination = tmp_path / "roi.jpg"
    observed_commands: list[list[str]] = []

    monkeypatch.setattr(false_positive_dataset, "_clip_duration_seconds", lambda *_args: 10.0)

    def fake_run(command, **_kwargs):
        observed_commands.append(command)
        Path(command[-1]).write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(false_positive_dataset.subprocess, "run", fake_run)

    _extract_crop(
        "ffmpeg",
        clip,
        destination,
        # Live timelines can outlast the browser clip by more than the old
        # single 0.25 s fallback could cover.
        clip_time_sec=11.52,
        roi=(32, 16, 288, 224),
    )

    assert destination.read_bytes() == b"jpeg"
    assert observed_commands[0][observed_commands[0].index("-ss") + 1] == "9.750000"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_extract_crop_clamps_a_timeline_timestamp_after_clip_end_with_real_ffmpeg(
    tmp_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    clip = tmp_path / "ten-second-event.mp4"
    created = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=1",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(clip),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert created.returncode == 0, created.stderr

    image = tmp_path / "roi.jpg"
    _extract_crop(
        ffmpeg,
        clip,
        image,
        # The live event timeline can extend beyond the browser clip by more
        # than the old single 0.25 s fallback could cover.
        clip_time_sec=1.52,
        roi=(32, 16, 288, 224),
    )

    assert image.read_bytes().startswith(b"\xff\xd8\xff")
