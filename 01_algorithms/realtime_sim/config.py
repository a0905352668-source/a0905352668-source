from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIEW_ORDER = ["dianqi1", "dianqi2", "jixie1", "jixie2", "ruanjian1", "ruanjian2", "zoulang"]

VIEW_LABELS = {
    "dianqi1": "电气1",
    "dianqi2": "电气2",
    "jixie1": "机械1",
    "jixie2": "机械2",
    "ruanjian1": "软件1",
    "ruanjian2": "软件2",
    "zoulang": "走廊",
}

CALIBRATION_FILES = {
    "dianqi1": "camera_01_screen_calibration_v21.json",
    "dianqi2": "camera_02_screen_calibration_v21.json",
    "jixie1": "camera_mechanical_01_screen_calibration_v21.json",
    "jixie2": "camera_mechanical_02_screen_calibration_v21.json",
    "ruanjian1": "camera_software_01_screen_calibration_v21.json",
    "ruanjian2": "camera_software_02_screen_calibration_v21.json",
    "zoulang": "camera_corridor_screen_calibration_v21.json",
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
DEFAULT_PHONE_MODEL_VERSION = "v20260708_02_yolo11s512_ALLtodayx5_fullNet_lowFP_50p3"
DEFAULT_PHONE_ENGINE_NAME = "best_trt86_50_2_b16_fp16.engine"


@dataclass(frozen=True)
class ViewSpec:
    stream_index: int
    view_key: str
    view_label: str
    source_folder: Path
    source_video: Path
    calibration_file: Path


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    source_root: Path
    run_root: Path
    cpp_binary: Path
    pose_plan: Path
    phone_plan: Path
    calib_dir: Path
    infer_fps: float = 8.0
    phone_conf: float = 0.5
    segment_seconds: float = 10.0
    release_mode: str = "realtime"
    port: int = 8767


def default_runtime_config() -> RuntimeConfig:
    project_root = Path("/media/boshi/Data/JianKong")
    return RuntimeConfig(
        project_root=project_root,
        source_root=project_root / "03_raw_videos_and_frames/2026-07-03",
        run_root=project_root / "06_training_runs",
        cpp_binary=project_root / "01_algorithms/tools/cpp_full_pipeline_bench_gpu",
        pose_plan=project_root / "06_training_runs/raw_trt_plans_20260706_114432/pose960_static_b7.plan",
        phone_plan=project_root / "07_models/phone_models/versions" / DEFAULT_PHONE_MODEL_VERSION / DEFAULT_PHONE_ENGINE_NAME,
        calib_dir=project_root / "02_configs/surveillance/recalibration_20260706/generated_v21_from_labelme_20260706_103319",
    )


def view_key_from_path(path: Path) -> str:
    text = str(path).lower()
    for key in VIEW_ORDER:
        if key in text:
            return key
    aliases = {
        "camera_01": "dianqi1",
        "camera_02": "dianqi2",
        "mechanical1": "jixie1",
        "mechanical2": "jixie2",
        "software1": "ruanjian1",
        "software2": "ruanjian2",
        "corridor": "zoulang",
    }
    for alias, key in aliases.items():
        if alias in text:
            return key
    raise ValueError(f"cannot infer view key from {path}")


def select_videos_by_rank_from_end(source_root: Path, cfg: RuntimeConfig | None = None, rank_from_end: int = 1) -> list[ViewSpec]:
    cfg = cfg or default_runtime_config()
    if rank_from_end < 1:
        raise ValueError("rank_from_end must be >= 1")
    by_key: dict[str, ViewSpec] = {}
    for folder in sorted(p for p in source_root.iterdir() if p.is_dir()):
        key = view_key_from_path(folder)
        videos = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS],
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        if not videos:
            continue
        if len(videos) < rank_from_end:
            raise RuntimeError(f"view {key} has only {len(videos)} videos, cannot pick rank {rank_from_end} from end")
        index = VIEW_ORDER.index(key)
        by_key[key] = ViewSpec(
            stream_index=index,
            view_key=key,
            view_label=VIEW_LABELS[key],
            source_folder=folder,
            source_video=videos[-rank_from_end],
            calibration_file=cfg.calib_dir / CALIBRATION_FILES[key],
        )
    missing = [key for key in VIEW_ORDER if key not in by_key]
    if missing:
        raise RuntimeError(f"missing source videos for views: {missing}")
    return [by_key[key] for key in VIEW_ORDER]


def select_last_videos(source_root: Path, cfg: RuntimeConfig | None = None) -> list[ViewSpec]:
    return select_videos_by_rank_from_end(source_root, cfg, rank_from_end=1)
