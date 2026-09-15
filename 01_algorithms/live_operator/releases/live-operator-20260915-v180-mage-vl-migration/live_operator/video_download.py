"""Bounded, CPU-only exports of recorded clips with outline annotations."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


_OUTPUT_BUDGET = 128 * 1024**2
_CACHE_BUDGET = 2 * 1024**3
_MIN_FREE = 20 * 1024**3


def _guard_paths(*paths: Path) -> None:
    for path in paths:
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("导出路径不可用")
        for component in (*reversed(path.parents), path):
            if component.is_symlink():
                raise ValueError("导出路径不可用")


def _signature(path: Path) -> tuple[int, int, int, int]:
    details = path.stat()
    return details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns


def _check_capacity(root: Path, reserved: int = _OUTPUT_BUDGET) -> None:
    _guard_paths(root)
    used = 0
    if root.exists():
        for path in root.iterdir():
            if path.is_symlink():
                raise ValueError("下载缓存路径不可用")
            if path.is_file():
                used += path.stat().st_size
    if used + reserved > _CACHE_BUDGET:
        raise ExportBusy("下载缓存容量不足，请联系管理员")
    if shutil.disk_usage(root.parent).free - reserved < _MIN_FREE:
        raise ExportBusy("录像盘剩余空间不足，暂停视频导出")


class ExportBusy(ValueError):
    pass


def _number(value: object, fallback: float = 0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else fallback
    except (ValueError, TypeError):
        return fallback


def _ass_time(seconds: float) -> str:
    ticks = max(0, round(seconds * 100))
    return f"{ticks // 360000}:{ticks // 6000 % 60:02d}:{ticks // 100 % 60:02d}.{ticks % 100:02d}"


def _outlines(payload: dict, width: int, height: int, duration: float) -> str:
    timeline = payload.get("overlay", {}).get("bbox_timeline")
    if not isinstance(timeline, list) or len(timeline) > 12000:
        raise ValueError("视频标注数据不可用")
    groups: dict[float, list[dict]] = {}
    for sample in timeline:
        if not isinstance(sample, dict):
            continue
        when = _number(sample.get("time_sec"), -100)
        if -0.75 <= when < duration:
            groups.setdefault(when, []).append(sample)
    times = sorted(groups)
    header = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,10,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    ass_size = len(header)
    colors = {"person": "343FEF", "risk": "1F8AFF", "phone": "20B0FF"}
    for index, when in enumerate(times):
        end = min(duration, when + 0.75, times[index + 1] if index + 1 < len(times) else duration)
        if end <= when:
            continue
        for sample in groups[when]:
            boxes = sample.get("boxes")
            phones = sample.get("phone_boxes")
            if (len(boxes) if isinstance(boxes, list) else 1) + (len(phones) if isinstance(phones, list) else 0) > 64:
                raise ValueError("视频标注目标过多")
            if not isinstance(boxes, list):
                boxes = [{"label": "risk" if sample.get("alarm") is False else "person", "bbox": sample.get("bbox")}]
            boxes = boxes + [{**p, "label": "phone", "bbox": p.get("box") or p.get("bbox")}
                             for p in phones if isinstance(p, dict)] if isinstance(phones, list) else boxes
            for item in boxes:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or item.get("type") or "person").lower()
                box = item.get("bbox") or item.get("box")
                if label not in colors or not isinstance(box, list) or len(box) < 4:
                    continue
                sw = _number(item.get("frame_width", sample.get("frame_width")), width)
                sh = _number(item.get("frame_height", sample.get("frame_height")), height)
                coords = [_number(v, float("nan")) for v in box[:4]]
                if sw <= 0 or sh <= 0 or not all(math.isfinite(v) for v in coords):
                    continue
                x1, y1, x2, y2 = [round(v * (width / sw if i % 2 == 0 else height / sh)) for i, v in enumerate(coords)]
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                stroke = max(2, round(width / 720))
                rects = [(x1, y1, x2, min(y2, y1 + stroke)),
                         (x1, max(y1, y2 - stroke), x2, y2),
                         (x1, y1, min(x2, x1 + stroke), y2),
                         (max(x1, x2 - stroke), y1, x2, y2)]
                drawing = " ".join(f"m {a} {b} l {c} {b} {c} {d} {a} {d} {a} {b}" for a, b, c, d in rects)
                text = f"{{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H{colors[label]}&\\p1}}{drawing}"
                if len(lines) >= 24000:
                    raise ValueError("视频标注目标过多")
                lines.append(f"Dialogue: 0,{_ass_time(max(0, when))},{_ass_time(end)},Default,,0,0,0,,{text}\n")
                ass_size += len(lines[-1])
                if ass_size > 8 * 1024**2:
                    raise ValueError("视频标注展开数据过大")
    return header + "".join(lines)


def render_boxed_clip(source: Path, overlay: Path, destination: Path, expected: tuple | None = None) -> None:
    _guard_paths(source, overlay, destination)
    if expected is not None and expected != (_signature(source), _signature(overlay)):
        raise ValueError("视频或标注已更新，请重试")
    _check_capacity(destination.parent)
    if overlay.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("视频标注数据过大")
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration", "-of", "json", str(source),
    ], check=True, capture_output=True, timeout=15)
    info = json.loads(probe.stdout)
    width, height = int(info["streams"][0]["width"]), int(info["streams"][0]["height"])
    duration = _number(info.get("format", {}).get("duration"))
    if not 0 < duration <= 60 or not 0 < width <= 3840 or not 0 < height <= 2160:
        raise ValueError("该视频超出导出范围")
    with tempfile.TemporaryDirectory(prefix="boxed-", dir=destination.parent) as folder:
        work = Path(folder)
        (work / "boxes.ass").write_text(_outlines(payload, width, height, duration), encoding="utf-8")
        _guard_paths(source, overlay, destination, work)
        subprocess.run([
            "nice", "-n", "10", "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-threads", "2", "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
            "-filter_threads", "1", "-vf", "ass=boxes.ass", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p",
            "-maxrate", "8M", "-bufsize", "16M", "-t", str(duration), "-fs", str(_OUTPUT_BUDGET),
            "-c:a", "aac", "-sn", "-dn", "-movflags", "+faststart", "output.mp4",
        ], cwd=work, check=True, capture_output=True, timeout=180)
        output = work / "output.mp4"
        _guard_paths(source, overlay, destination, output)
        if expected is not None and expected != (_signature(source), _signature(overlay)):
            raise ValueError("视频或标注已更新，请重试")
        if not 0 < output.stat().st_size <= _OUTPUT_BUDGET:
            raise ValueError("视频导出失败")
        exported_probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(output),
        ], check=True, capture_output=True, timeout=15)
        exported_duration = _number(json.loads(exported_probe.stdout).get("format", {}).get("duration"))
        if exported_duration < duration - 0.25:
            raise ValueError("视频导出不完整")
        _guard_paths(source, overlay, destination, output)
        _check_capacity(destination.parent, output.stat().st_size)
        output.replace(destination)


class VideoDownloads:
    """One worker, at most four active requests, source-sensitive disk cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="boxed-video")
        self._jobs: dict[Path, Future] = {}

    def status(self, source: Path, overlay: Path, *, start: bool = False) -> tuple[str, Path]:
        root = source.parent / "boxed_downloads"
        _guard_paths(source, overlay, root)
        expected = (_signature(source), _signature(overlay))
        fingerprint = "boxed-v1|" + "|".join(
            f"{p}:{signature}" for p, signature in zip((source, overlay), expected)
        )
        key = hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
        output = root / f"{source.stem}-{key}-boxed.mp4"
        with self._lock:
            if output.is_symlink():
                raise ValueError("下载缓存路径不可用")
            if output.is_file() and output.stat().st_size > 0:
                return "ready", output
            job = self._jobs.get(output)
            if job is not None and not job.done():
                return ("running" if job.running() else "queued"), output
            if not start:
                return ("error" if job is not None else "idle"), output
            if sum(not future.done() for future in self._jobs.values()) >= 4:
                raise ExportBusy("导出任务较多，请稍后重试")
            reserved = 1 + sum(not future.done() and path.parent == root for path, future in self._jobs.items())
            _check_capacity(root, reserved * _OUTPUT_BUDGET)
            root.mkdir(exist_ok=True)
            self._jobs = {path: future for path, future in self._jobs.items() if not future.done()}
            self._jobs[output] = self._executor.submit(render_boxed_clip, source, overlay, output, expected)
            return "queued", output
