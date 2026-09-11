#!/usr/bin/env python3
"""Extract a deterministic, redacted spatial-static regression fixture."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _bbox(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must contain four coordinates")
    box = [_finite_number(item, field) for item in value]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{field} must have positive width and height")
    return box


def _matching_entity(entities: Any, track_id: int, kind: str) -> dict[str, Any]:
    if not isinstance(entities, list):
        raise ValueError(f"{kind} list is missing")
    matches = [entity for entity in entities if entity.get("track_id") == track_id]
    if not matches:
        raise ValueError(f"no {kind} matched person track {track_id}")
    if kind == "phone":
        return max(
            matches,
            key=lambda entity: (
                bool(entity.get("accepted", False)),
                float(entity.get("risk_score", entity.get("final_risk_score", 0.0))),
                float(entity.get("confidence", 0.0)),
            ),
        )
    if len(matches) != 1:
        raise ValueError(f"expected one {kind} for track {track_id}, found {len(matches)}")
    return matches[0]


def _selected_rows(
    lines: Iterable[str], stream_index: int, first_frame: int, last_frame: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if row.get("stream_index") != stream_index:
            continue
        frame_index = row.get("frame_index")
        if isinstance(frame_index, int) and frame_index > last_frame:
            break
        if isinstance(frame_index, int) and first_frame <= frame_index <= last_frame:
            selected.append(row)
    selected.sort(key=lambda row: row["frame_index"])
    expected_frames = list(range(first_frame, last_frame + 1))
    actual_frames = [row["frame_index"] for row in selected]
    if actual_frames != expected_frames:
        raise ValueError(
            f"target interval must contain each frame exactly once; "
            f"expected {len(expected_frames)}, found {len(actual_frames)}"
        )
    return selected


def build_fixture(
    rows: list[dict[str, Any]], stream_index: int, track_id: int
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    source_times: list[float] = []
    for row in rows:
        frame_index = row["frame_index"]
        person = _matching_entity(row.get("persons"), track_id, "person")
        phone = _matching_entity(row.get("phones"), track_id, "phone")
        source_time = _finite_number(row.get("time_sec"), f"frame {frame_index} time_sec")
        source_times.append(source_time)
        risk_value = phone.get("risk_score", phone.get("final_risk_score"))
        samples.append({
            "time_sec": 0.0,
            "phone_bbox": _bbox(phone.get("box"), f"frame {frame_index} phone box"),
            "person_bbox": _bbox(person.get("box"), f"frame {frame_index} person box"),
            "risk_score": _finite_number(risk_value, f"frame {frame_index} risk_score"),
            "accepted": bool(phone.get("accepted", False)),
        })

    start_time = source_times[0]
    for sample, source_time in zip(samples, source_times):
        sample["time_sec"] = round(source_time - start_time, 6)
    intervals = [
        later - earlier for earlier, later in zip(source_times, source_times[1:])
    ]
    if not intervals or any(interval <= 0.0 for interval in intervals):
        raise ValueError("target timestamps must be strictly increasing")
    infer_fps = round(1.0 / statistics.median(intervals), 6)
    if not math.isclose(infer_fps, 8.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(f"target interval is not 8 FPS (derived {infer_fps})")

    return {
        "camera": f"camera{stream_index + 1:02d}",
        "track_id": track_id,
        "infer_fps": 8.0,
        "expected": {"alarm": False, "static_suppressed": True},
        "samples": samples,
    }


def _maximum_phone_displacement(samples: list[dict[str, Any]]) -> float:
    centers = [
        ((sample["phone_bbox"][0] + sample["phone_bbox"][2]) * 0.5,
         (sample["phone_bbox"][1] + sample["phone_bbox"][3]) * 0.5)
        for sample in samples
    ]
    return max(
        math.hypot(x1 - x2, y1 - y2)
        for x1, y1 in centers
        for x2, y2 in centers
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--stream-index", type=int, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--first-frame", type=int, required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stream_index < 0:
        raise ValueError("--stream-index must be non-negative")
    if args.last_frame < args.first_frame:
        raise ValueError("--last-frame must be >= --first-frame")
    with args.jsonl.open("r", encoding="utf-8") as handle:
        rows = _selected_rows(
            handle, args.stream_index, args.first_frame, args.last_frame
        )
    fixture = build_fixture(rows, args.stream_index, args.track_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    displacement = _maximum_phone_displacement(fixture["samples"])
    print(
        f"selected={len(rows)} samples={len(fixture['samples'])} "
        f"max_phone_displacement_px={displacement:.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
