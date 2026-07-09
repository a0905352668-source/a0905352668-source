from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def curl_head(url: str, range_header: bool = False) -> str:
    cmd = ["curl", "-sI", "--max-time", "10"]
    if range_header:
        cmd += ["-H", "Range: bytes=0-1023"]
    cmd.append(url)
    return subprocess.check_output(cmd, text=True)


def curl_range_get(url: str) -> str:
    return subprocess.check_output(
        ["curl", "-s", "-D", "-", "-o", "/dev/null", "--max-time", "10", "-H", "Range: bytes=0-1023", url],
        text=True,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    state = json.loads((run_dir / "dashboard/api/run_state.json").read_text(encoding="utf-8"))
    segments = json.loads((run_dir / "dashboard/api/segments.json").read_text(encoding="utf-8"))["segments"]
    index_head = curl_head(f"http://127.0.0.1:{args.port}/")
    ready = [s for s in segments if s.get("clip_status") == "ready" and s.get("clip")]
    clip_head = ""
    if ready:
        clip_head = curl_range_get(f"http://127.0.0.1:{args.port}/{ready[0]['clip']}")
    report = {
        "run_dir": str(run_dir),
        "state": state,
        "segment_count": len(segments),
        "ready_clip_count": len(ready),
        "index_ok": "200" in index_head.splitlines()[0],
        "range_ok": ("206" in clip_head.splitlines()[0]) if clip_head else False,
    }
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs/validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["index_ok"]:
        raise SystemExit("dashboard index did not return 200")
    if ready and not report["range_ok"]:
        raise SystemExit("ready clip did not return 206 for Range request")


if __name__ == "__main__":
    main()
