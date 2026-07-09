from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from realtime_sim.clip_builder import build_missing_clips
from realtime_sim.dashboard_live import write_dashboard_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--until-complete", action="store_true")
    return parser.parse_args()


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def all_segments_finished(run_dir: Path) -> bool:
    payload = load_json(run_dir / "events" / "segments.json", {"segments": []})
    segments = payload.get("segments", [])
    return bool(segments) and all(s.get("clip_status") in {"ready", "failed"} for s in segments)


def detector_complete(run_dir: Path) -> bool:
    state = load_json(run_dir / "events" / "state.json", {})
    return state.get("status") == "complete"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    while True:
        changed = build_missing_clips(run_dir)
        write_dashboard_api(run_dir)
        if args.until_complete and detector_complete(run_dir) and all_segments_finished(run_dir):
            print(f"clip_worker_done changed={changed}", flush=True)
            return
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    main()
