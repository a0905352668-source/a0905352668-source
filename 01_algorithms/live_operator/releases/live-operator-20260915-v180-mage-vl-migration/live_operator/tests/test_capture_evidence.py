from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from live_operator.capture_evidence import (
    CAPTURE_EVIDENCE_REVISION,
    build_capture_panel,
    decode_capture_frames,
)


def _synthetic_video(path: Path, *, frame_count: int = 60, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (320, 240),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 20)
        frame[60:200, 110:210] = (40, 80, 160)
        frame[115:130, 175:190] = (220, min(255, index * 4), 20)
        writer.write(frame)
    writer.release()
    return path


def _overlay(*, frame_count: int = 60, alarm_index: int = 30) -> dict:
    return {
        "bbox_timeline": [
            {
                "track_id": "person-1",
                "time_sec": index / 10.0,
                "frame_index": index,
                "frame_width": 320,
                "frame_height": 240,
                "bbox": [110, 60, 210, 200],
                "alarm": index == alarm_index,
                "phone_boxes": [
                    {
                        "box": [175, 115, 190, 130],
                        "accepted": True,
                        "phone_score": 0.9,
                    }
                ],
            }
            for index in range(frame_count)
        ]
    }


def _visibility() -> dict:
    return {
        "frame_width": 320,
        "frame_height": 240,
        "screens": [
            {
                "screen_id": "screen-1",
                "screen_polygon": [[10, 10], [90, 10], [90, 55], [10, 55]],
            }
        ],
        "occluders": [
            {
                "occluder_id": "partition-1",
                "polygon": [[220, 0], [240, 0], [240, 220], [220, 220]],
            }
        ],
    }


def test_panel_keeps_scene_native_person_and_phone_detail() -> None:
    frame = Image.new("RGB", (320, 240), (10, 20, 30))
    for x in range(110, 210):
        for y in range(60, 200):
            frame.putpixel((x, y), (30, 60, 90))
    frame.putpixel((180, 120), (255, 0, 255))

    panel = build_capture_panel(
        frame,
        person_box=(110, 60, 210, 200),
        phone_box=(175, 115, 190, 130),
        screen_polygons=[[(10, 10), (90, 10), (90, 55), (10, 55)]],
        occluder_polygons=[[(220, 0), (240, 0), (240, 220), (220, 220)]],
    )

    assert panel.size == (896, 448)
    assert panel.getpixel((14, 70)) == (255, 0, 0)
    assert panel.getpixel((308, 56)) == (0, 255, 0)
    assert any(
        panel.getpixel((x, y)) == (255, 0, 255)
        for x in range(448, 896)
        for y in range(448)
    )
    # The native 100x140 person pixels remain present without mandatory upscaling.
    native_person_pixels = sum(
        panel.getpixel((x, y)) == (30, 60, 90)
        for x in range(448, 896)
        for y in range(448)
    )
    assert native_person_pixels >= 100 * 140 - 1


def test_decode_selects_16_distinct_chronological_frames_within_five_seconds(
    tmp_path: Path,
) -> None:
    video = _synthetic_video(tmp_path / "event.avi")

    sequences = decode_capture_frames(video, _overlay(), _visibility())

    assert CAPTURE_EVIDENCE_REVISION == "scene-person-phone-span5s-16f-jpeg92-v1"
    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.track_id == "person-1"
    assert len(sequence.frames) == 16
    assert sequence.source_frame_indices == tuple(sorted(set(sequence.source_frame_indices)))
    assert sequence.times[-1] - sequence.times[0] <= 5.0
    assert all(frame.size == (896, 448) for frame in sequence.frames)


def test_short_sequence_is_not_padded_with_repeated_frames(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / "short.avi", frame_count=3)
    overlay = _overlay(frame_count=3, alarm_index=1)

    sequences = decode_capture_frames(video, overlay, _visibility())

    assert len(sequences) == 1
    assert sequences[0].source_frame_indices == (0, 1, 2)
    assert len(sequences[0].frames) == 3
