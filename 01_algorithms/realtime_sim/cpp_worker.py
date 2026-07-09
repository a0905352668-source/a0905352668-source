from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from realtime_sim.config import RuntimeConfig
from realtime_sim.segment_producer import write_json_atomic

TRT_LIB = "/home/boshi/hqc/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib"
CUDA_LIBS = "/usr/local/cuda-12.1/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib"


def view_folder_name(view_key: str) -> str:
    return f"{view_key}_live"


def expected_stream_file_count(batch: dict) -> int:
    return len(batch.get("files", []))


def load_batch(batch_dir: Path) -> dict:
    return json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def build_mini_root(run_dir: Path, batch_dir: Path) -> Path:
    batch = load_batch(batch_dir)
    batch_index = int(batch["batch_index"])
    mini_root = run_dir / "mini_roots" / f"batch_{batch_index:06d}"
    if mini_root.exists():
        shutil.rmtree(mini_root)
    mini_root.mkdir(parents=True)
    for item in sorted(batch["files"], key=lambda x: int(x["stream_index"])):
        folder = mini_root / view_folder_name(item["view_key"])
        source = Path(item["segment_file"])
        target = folder / f"batch_{batch_index:06d}_{item['view_key']}.mp4"
        link_or_copy(source, target)
    return mini_root


def build_cpp_command(cfg: RuntimeConfig, mini_root: Path, output_dir: Path, batch: dict, no_video: bool = False) -> list[str]:
    max_samples = int(float(batch["duration_sec"]) * cfg.infer_fps) + 2
    cmd = [
        str(cfg.cpp_binary),
        "--root", str(mini_root),
        "--pick-from-end", "1",
        "--max-sampled", str(max_samples),
        "--infer-fps", str(cfg.infer_fps),
        "--pose-plan", str(cfg.pose_plan),
        "--phone-plan", str(cfg.phone_plan),
        "--phone-conf", f"{cfg.phone_conf:g}",
        "--calib-dir", str(cfg.calib_dir),
        "--output-dir", str(output_dir),
        "--gpu-preprocess",
        "--pipelined-read",
        "--queue-size", "16",
        "--cv-threads", "2",
    ]
    if no_video:
        cmd.append("--no-video")
    return cmd


def parse_cpp_summary(text: str) -> dict:
    parsed: dict[str, float | int] = {}
    match = re.search(r"^\[SUMMARY\]\s+(.*)$", text, flags=re.MULTILINE)
    if not match:
        return parsed
    for key, value in re.findall(r"([A-Za-z_]+)=([0-9.]+)", match.group(1)):
        number = float(value) if "." in value else int(value)
        parsed[key] = number
    return parsed


def run_cpp_batch(cfg: RuntimeConfig, run_dir: Path, batch_dir: Path, no_video: bool = False) -> Path:
    batch = load_batch(batch_dir)
    batch_index = int(batch["batch_index"])
    mini_root = build_mini_root(run_dir, batch_dir)
    output_dir = run_dir / "processed" / f"batch_{batch_index:06d}" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"batch_{batch_index:06d}_cpp.log"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TRT_LIB}:{CUDA_LIBS}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = build_cpp_command(cfg, mini_root, output_dir, batch, no_video=no_video)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cfg.project_root / "01_algorithms"),
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    summary = {
        "batch_index": batch_index,
        "returncode": proc.returncode,
        "mini_root": str(mini_root),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "no_video": no_video,
        "cpp_summary": parse_cpp_summary(log_text),
    }
    write_json_atomic(output_dir.parent / "worker_summary.json", summary)
    if proc.returncode != 0:
        raise RuntimeError(f"C++ worker failed for batch {batch_index}; see {log_path}")
    return output_dir.parent
