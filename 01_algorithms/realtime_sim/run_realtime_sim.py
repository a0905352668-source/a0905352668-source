from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

from realtime_sim.aggregator import append_batch_events
from realtime_sim.clip_builder import build_missing_clips
from realtime_sim.config import default_runtime_config, select_videos_by_rank_from_end
from realtime_sim.cpp_worker import run_cpp_batch
from realtime_sim.dashboard_live import write_dashboard_api
from realtime_sim.metrics import append_batch_metrics
from realtime_sim.range_server import serve_dashboard
from realtime_sim.segment_producer import batch_count_for_views, create_run_dir, produce_batches, write_json_atomic, write_source_manifest

DEFAULT_CLIP_PYTHON = Path("/home/boshi/miniconda3/bin/python")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["realtime", "fast"], default="realtime")
    parser.add_argument("--max-batches", type=int, default=6)
    parser.add_argument("--infer-fps", type=float)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--keep-serving", action="store_true")
    parser.add_argument("--no-cpp-video", dest="no_cpp_video", action="store_true")
    parser.add_argument("--with-cpp-video", dest="no_cpp_video", action="store_false")
    parser.set_defaults(no_cpp_video=True)
    parser.add_argument("--clip-build-mode", choices=["during", "after", "background", "none"], default="background")
    parser.add_argument("--cpp-binary")
    parser.add_argument("--source-root")
    parser.add_argument("--pick-from-end", type=int, default=1)
    parser.add_argument("--clip-python", default=str(DEFAULT_CLIP_PYTHON))
    return parser.parse_args(argv)


def build_clip_builder_command(run_dir: Path, python_executable: Path, project_root: Path) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "realtime_sim.clip_builder",
        "--run-dir",
        str(run_dir),
    ]


def build_clip_worker_command(run_dir: Path, python_executable: Path, project_root: Path) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "realtime_sim.clip_worker",
        "--run-dir",
        str(run_dir),
        "--until-complete",
    ]


def start_clip_worker(run_dir: Path, python_executable: Path, project_root: Path, log_path: Path) -> subprocess.Popen | None:
    if not python_executable.exists():
        return None
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        build_clip_worker_command(run_dir, python_executable, project_root),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def run_clip_builder(run_dir: Path, python_executable: Path, project_root: Path) -> int:
    if python_executable.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
        subprocess.check_call(build_clip_builder_command(run_dir, python_executable, project_root), env=env)
        return 0
    changed = build_missing_clips(run_dir)
    write_dashboard_api(run_dir)
    return changed


def update_state(run_dir, extra: dict) -> None:
    state_path = run_dir / "events" / "state.json"
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(extra)
    write_json_atomic(state_path, state)
    write_dashboard_api(run_dir)


def main() -> None:
    args = parse_args()
    cfg0 = default_runtime_config()
    cfg_updates = {"segment_seconds": args.segment_seconds, "release_mode": args.mode, "port": args.port}
    if args.infer_fps is not None:
        cfg_updates["infer_fps"] = args.infer_fps
    if args.cpp_binary:
        cfg_updates["cpp_binary"] = Path(args.cpp_binary)
    if args.source_root:
        cfg_updates["source_root"] = Path(args.source_root)
    cfg = replace(cfg0, **cfg_updates)
    views = select_videos_by_rank_from_end(cfg.source_root, cfg, rank_from_end=args.pick_from_end)
    target_batches = args.max_batches if args.max_batches > 0 else batch_count_for_views(views, cfg.segment_seconds)
    run_dir = create_run_dir(cfg)
    write_source_manifest(run_dir, views, cfg.segment_seconds)
    update_state(run_dir, {
        "status": "starting",
        "run_dir": str(run_dir),
        "event_count": 0,
        "alarm_event_count": 0,
        "infer_fps": cfg.infer_fps,
        "phone_conf": cfg.phone_conf,
        "realtime_path": "metadata_only" if args.no_cpp_video else "boxed_video",
        "clip_build_mode": args.clip_build_mode,
        "pose_plan": str(cfg.pose_plan),
        "phone_plan": str(cfg.phone_plan),
        "phone_model_id": cfg.phone_plan.parent.name,
    })

    server_thread = threading.Thread(target=serve_dashboard, args=(run_dir / "dashboard", cfg.port), daemon=True)
    server_thread.start()
    producer_thread = threading.Thread(
        target=produce_batches,
        args=(run_dir, views, cfg.segment_seconds, target_batches, args.mode),
        daemon=True,
    )
    producer_thread.start()
    clip_worker_proc = None
    if args.clip_build_mode == "background":
        clip_worker_proc = start_clip_worker(
            run_dir,
            Path(args.clip_python),
            cfg.project_root / "01_algorithms",
            run_dir / "logs" / "clip_worker.log",
        )

    processed = set()
    update_state(run_dir, {"status": "running", "url": f"http://192.168.50.2:{cfg.port}/"})
    while len(processed) < target_batches:
        for batch_dir in sorted((run_dir / "incoming").glob("batch_*")):
            if batch_dir.name in processed or not (batch_dir / "batch.json").exists():
                continue
            t0 = time.monotonic()
            processed_dir = run_cpp_batch(cfg, run_dir, batch_dir, no_video=args.no_cpp_video)
            state = append_batch_events(run_dir, batch_dir, processed_dir)
            worker_summary = json.loads((processed_dir / "worker_summary.json").read_text(encoding="utf-8"))
            batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
            batch.setdefault("batch_name", batch_dir.name)
            metric = append_batch_metrics(
                run_dir=run_dir,
                batch=batch,
                elapsed_sec=time.monotonic() - t0,
                worker_summary=worker_summary,
                state=state,
                infer_fps=cfg.infer_fps,
                phone_conf=cfg.phone_conf,
                pose_plan=cfg.pose_plan,
                phone_plan=cfg.phone_plan,
                no_video=args.no_cpp_video,
                clip_build_mode=args.clip_build_mode,
            )
            changed = 0
            if args.clip_build_mode == "during":
                changed = run_clip_builder(run_dir, Path(args.clip_python), cfg.project_root / "01_algorithms")
            write_dashboard_api(run_dir)
            processed.add(batch_dir.name)
            update_state(
                run_dir,
                {
                    "status": "running",
                    "processed_batches": len(processed),
                    "target_batches": target_batches,
                    "last_batch_sec": round(time.monotonic() - t0, 3),
                    "clip_updates": changed,
                    "last_aggregate_fps": metric.get("aggregate_fps", 0),
                    "realtime_path": metric.get("realtime_path"),
                    "clip_build_mode": metric.get("clip_build_mode"),
                    "phone_model_id": metric.get("phone_model_id"),
                    **state,
                },
            )
        time.sleep(0.5)

    producer_thread.join(timeout=1)
    if args.clip_build_mode in {"during", "after"}:
        run_clip_builder(run_dir, Path(args.clip_python), cfg.project_root / "01_algorithms")
    write_dashboard_api(run_dir)
    update_state(run_dir, {"status": "complete", "processed_batches": len(processed), "target_batches": target_batches})
    if clip_worker_proc is not None:
        write_dashboard_api(run_dir)
    print(run_dir, flush=True)
    print(f"http://192.168.50.2:{cfg.port}/", flush=True)

    while args.keep_serving:
        time.sleep(60)


if __name__ == "__main__":
    main()
