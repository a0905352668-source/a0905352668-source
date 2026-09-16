from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from live_operator.capture_evidence import (
    CAPTURE_CROP_MARGIN_RATIO,
    CAPTURE_EVIDENCE_REVISION,
    CAPTURE_FRAME_COUNT,
    build_capture_frame,
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
                "screen_id": "screen-1",
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


def test_frame_is_a_joint_person_screen_crop_without_full_scene_or_phone_inset() -> None:
    frame = Image.new("RGB", (320, 240), (10, 20, 30))
    for x in range(110, 210):
        for y in range(60, 200):
            frame.putpixel((x, y), (30, 60, 90))
    for x in range(176, 185):
        for y in range(116, 125):
            frame.putpixel((x, y), (255, 0, 255))
    for x in range(290, 310):
        for y in range(210, 230):
            frame.putpixel((x, y), (0, 255, 0))

    evidence = build_capture_frame(
        frame,
        crop_box=(0, 0, 245, 225),
    )

    assert evidence.size == (448, 448)
    assert CAPTURE_CROP_MARGIN_RATIO == 0.25
    assert any(
        evidence.getpixel((x, y)) == (255, 0, 255)
        for x in range(448)
        for y in range(448)
    )
    assert not any(
        evidence.getpixel((x, y)) == (0, 255, 0)
        for x in range(448)
        for y in range(448)
    )
    assert not any(
        evidence.getpixel((x, y)) in {(255, 0, 0), (255, 165, 0)}
        for x in range(448)
        for y in range(448)
    )


def test_decode_selects_30_distinct_chronological_joint_crop_frames(
    tmp_path: Path,
) -> None:
    video = _synthetic_video(tmp_path / "event.avi")

    sequences = decode_capture_frames(video, _overlay(), _visibility())

    assert CAPTURE_FRAME_COUNT == 30
    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.track_id == "person-1"
    assert len(sequence.frames) == 30
    assert sequence.source_frame_indices == tuple(sorted(set(sequence.source_frame_indices)))
    assert sequence.times[-1] - sequence.times[0] <= 5.0
    assert all(frame.size == (448, 448) for frame in sequence.frames)


def test_short_sequence_is_not_padded_with_repeated_frames(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / "short.avi", frame_count=3)
    overlay = _overlay(frame_count=3, alarm_index=1)

    sequences = decode_capture_frames(video, overlay, _visibility())

    assert len(sequences) == 1
    assert sequences[0].source_frame_indices == (0, 1, 2)
    assert len(sequences[0].frames) == 3


def test_missing_screen_annotation_returns_no_evidence(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / "missing-screen.avi")
    visibility = _visibility()
    visibility["screens"] = []

    assert decode_capture_frames(video, _overlay(), visibility) == ()


def test_stale_screen_id_does_not_replace_two_nearby_screen_candidates(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / "stale-screen.avi")
    visibility = _visibility()
    visibility['screens'] = [
        {'screen_id': 'screen-1', 'screen_polygon': [[10, 10], [30, 10], [30, 30], [10, 30]]},
        {'screen_id': 'right-a', 'screen_polygon': [[220, 70], [250, 70], [250, 150], [220, 150]]},
        {'screen_id': 'right-b', 'screen_polygon': [[260, 70], [290, 70], [290, 150], [260, 150]]},
    ]

    sequence = decode_capture_frames(video, _overlay(), visibility)[0]

    assert sequence.screen_ids == ('right-a', 'right-b')
    assert sequence.crop_box == (65, 25, 320, 235)


def test_screen_candidates_scale_phone_coordinates_to_video(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / 'scaled.avi')
    overlay = _overlay()
    for entry in overlay['bbox_timeline']:
        entry['frame_width'], entry['frame_height'] = 640, 480
        entry['bbox'] = [v * 2 for v in entry['bbox']]
        entry['phone_boxes'][0]['box'] = [v * 2 for v in entry['phone_boxes'][0]['box']]
    sequence = decode_capture_frames(video, overlay, _visibility())[0]
    assert sequence.screen_ids == ('screen-1',)
    assert sequence.crop_box == (0, 0, 260, 240)


def test_crop_limits_candidates_to_three_nearby_screens(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / 'three-screens.avi')
    visibility = _visibility()
    visibility['screens'] = [
        {'screen_id': f'near-{i}', 'screen_polygon': [[220+i*20, 70], [230+i*20, 70], [230+i*20, 150], [220+i*20, 150]]}
        for i in range(4)
    ]
    sequence = decode_capture_frames(video, _overlay(), visibility)[0]
    assert sequence.screen_ids == ('near-0', 'near-1', 'near-2')


def test_unavailable_phone_boxes_fall_back_to_person_geometry(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / 'no-phone-box.avi')
    overlay = _overlay()
    for entry in overlay['bbox_timeline']:
        entry['phone_boxes'] = None
    sequence = decode_capture_frames(video, overlay, _visibility())[0]
    assert sequence.screen_ids == ('screen-1',)
    assert len(sequence.frames) == 30
