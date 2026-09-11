#!/usr/bin/env python3
"""Validate spatial-static diagnostics and formal outcome in candidate JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_STATIC_FIELDS = frozenset({
    "static_cluster_samples",
    "static_detection_ratio",
    "static_center_spread_px",
    "static_bbox_iou_median",
    "static_pending",
    "static_pending_duration",
    "static_hotspot_score",
    "static_context_reason",
    "static_shadow_hits",
    "static_suppressed",
    "static_exit_reason",
})


def validate_rows(rows: Iterable[dict[str, Any]], stream_index: int, track_id: int) -> dict[str, int]:
    target_frames = 0
    target_entities = 0
    first_pending: int | None = None
    first_suppressed: int | None = None
    s4_frames: list[int] = []
    alarm_frames: list[int] = []

    for row_number, row in enumerate(rows, start=1):
        if row.get("stream_index") != stream_index:
            continue
        frame_index = row.get("frame_index", row_number)
        persons = [
            entity for entity in row.get("persons", [])
            if entity.get("track_id") == track_id
        ]
        phones = [
            entity for entity in row.get("phones", [])
            if entity.get("track_id") == track_id
        ]
        entities = persons + phones
        if not entities:
            continue
        target_frames += 1
        target_entities += len(entities)

        for kind, matches in (("person", persons), ("phone", phones)):
            for entity in matches:
                missing = sorted(REQUIRED_STATIC_FIELDS - entity.keys())
                if missing:
                    raise ValueError(
                        f"frame {frame_index} {kind} missing static diagnostics: "
                        + ", ".join(missing)
                    )
                if entity["static_pending"] and first_pending is None:
                    first_pending = target_frames
                if entity["static_suppressed"] and first_suppressed is None:
                    first_suppressed = target_frames

        for person in persons:
            if person.get("state") == "S4_ALARM":
                s4_frames.append(frame_index)
            if person.get("alarm") is True:
                alarm_frames.append(frame_index)

    if target_frames == 0:
        raise ValueError(
            f"no events found for stream {stream_index}, person track {track_id}"
        )
    if s4_frames:
        raise ValueError(f"S4_ALARM appears in target interval at frames {s4_frames}")
    if alarm_frames:
        raise ValueError(f"formal alarm appears in target interval at frames {alarm_frames}")
    if first_suppressed is None:
        raise ValueError("static_suppressed never appears in target interval")
    if first_pending is None or first_pending >= first_suppressed:
        raise ValueError("static_pending must appear before suppression")
    return {
        "target_frames": target_frames,
        "target_entities": target_entities,
        "first_pending_frame_offset": first_pending,
        "first_suppressed_frame_offset": first_suppressed,
    }


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--stream-index", type=int, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_rows(_load_jsonl(args.jsonl), args.stream_index, args.track_id)
    print(
        "static phone regression passed: "
        f"frames={summary['target_frames']} entities={summary['target_entities']} "
        f"pending_offset={summary['first_pending_frame_offset']} "
        f"suppressed_offset={summary['first_suppressed_frame_offset']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
