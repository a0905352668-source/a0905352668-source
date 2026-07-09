#!/usr/bin/env python3
import argparse
import datetime as dt
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


VIEW_LABELS = {
    "dianqi1": "电气1",
    "dianqi2": "电气2",
    "jixie1": "机械1",
    "jixie2": "机械2",
    "ruanjian1": "软件1",
    "ruanjian2": "软件2",
    "zoulang": "走廊",
}


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_streams(run_dir):
    path = run_dir / "videos" / "streams.json"
    if not path.exists():
        path = run_dir / "streams.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    videos = sorted((run_dir / "videos").glob("*.mp4"))
    return [
        {
            "stream_index": i,
            "output_video": str(video),
            "input_video": "",
            "width": 0,
            "height": 0,
            "infer_fps": 10.0,
            "sample_count": 0,
            "screen_count": 0,
        }
        for i, video in enumerate(videos)
    ]


def view_key_from_path(path):
    text = str(path).lower()
    for key in VIEW_LABELS:
        if key in text:
            return key
    if "corridor" in text:
        return "zoulang"
    if "software1" in text:
        return "ruanjian1"
    if "software2" in text:
        return "ruanjian2"
    if "mechanical1" in text:
        return "jixie1"
    if "mechanical2" in text:
        return "jixie2"
    return "unknown"


def view_label(stream):
    key = view_key_from_path(stream.get("input_video") or stream.get("output_video") or "")
    return VIEW_LABELS.get(key, "未知视角")


def clip_source_video(stream):
    return stream.get("input_video") or stream.get("output_video") or ""


def safe_name(text):
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = text.strip("._")
    return text or "clip"


def ffprobe_duration(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def build_segments(events, streams, before=3.0, after=5.0, merge_gap=4.0, min_duration=2.0):
    stream_map = {int(s["stream_index"]): s for s in streams}
    duration_cache = {}
    by_stream = {}
    for row in events:
        if int(row.get("accepted_count", 0)) <= 0 and int(row.get("alarm_track_count", 0)) <= 0:
            continue
        by_stream.setdefault(int(row["stream_index"]), []).append(row)

    segments = []
    for stream_index, rows in sorted(by_stream.items()):
        rows.sort(key=lambda r: float(r["time_sec"]))
        stream = stream_map.get(stream_index, {"stream_index": stream_index, "output_video": "", "input_video": ""})
        clip_source_text = clip_source_video(stream)
        clip_source = Path(clip_source_text) if clip_source_text else Path("__missing_video__")
        duration = duration_cache.get(clip_source)
        if duration is None:
            duration = ffprobe_duration(clip_source) if clip_source.is_file() else 0.0
            duration_cache[clip_source] = duration

        current = None
        for row in rows:
            t = float(row["time_sec"])
            if current is None or t - current["raw_end"] > merge_gap:
                if current is not None:
                    segments.append(finalize_segment(current, stream, duration, before, after, min_duration))
                current = {
                    "stream_index": stream_index,
                    "raw_start": t,
                    "raw_end": t,
                    "frames": [],
                    "accepted_total": 0,
                    "alarm_frame_count": 0,
                    "max_risk": 0.0,
                }
            current["raw_end"] = t
            current["frames"].append(row)
            current["accepted_total"] += int(row.get("accepted_count", 0))
            if int(row.get("alarm_track_count", 0)) > 0:
                current["alarm_frame_count"] += 1
            current["max_risk"] = max(current["max_risk"], float(row.get("max_risk", 0.0)))
        if current is not None:
            segments.append(finalize_segment(current, stream, duration, before, after, min_duration))

    segments = merge_overlapping_segments(segments)
    for i, seg in enumerate(segments, start=1):
        seg["id"] = f"event_{i:03d}"
        seg["rank"] = i
    return segments


def merge_overlapping_segments(segments):
    merged = []
    by_stream = {}
    for seg in segments:
        by_stream.setdefault(seg["stream_index"], []).append(seg)
    for stream_index in sorted(by_stream):
        rows = sorted(by_stream[stream_index], key=lambda s: (s["start"], s["end"]))
        current = None
        for seg in rows:
            if current is None:
                current = dict(seg)
                current["frames"] = list(seg["frames"])
                continue
            if seg["start"] <= current["end"]:
                current["end"] = max(current["end"], seg["end"])
                current["duration"] = round(current["end"] - current["start"], 3)
                current["raw_start"] = min(current["raw_start"], seg["raw_start"])
                current["raw_end"] = max(current["raw_end"], seg["raw_end"])
                current["accepted_total"] += seg["accepted_total"]
                current["alarm_frame_count"] += seg["alarm_frame_count"]
                current["max_risk"] = max(current["max_risk"], seg["max_risk"])
                current["level"] = "alarm" if current["level"] == "alarm" or seg["level"] == "alarm" else "risk"
                current["frames"].extend(seg["frames"])
            else:
                merged.append(current)
                current = dict(seg)
                current["frames"] = list(seg["frames"])
        if current is not None:
            merged.append(current)
    return sorted(merged, key=lambda s: (s["stream_index"], s["start"], s["end"]))


def finalize_segment(span, stream, duration, before, after, min_duration):
    start = max(0.0, span["raw_start"] - before)
    end = span["raw_end"] + after
    if duration > 0:
        end = min(duration, end)
    if end - start < min_duration:
        end = min(duration if duration > 0 else start + min_duration, start + min_duration)
    stream_label = view_label(stream)
    level = "alarm" if span["alarm_frame_count"] > 0 else "risk"
    return {
        "stream_index": span["stream_index"],
        "view": stream_label,
        "level": level,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0.0, end - start), 3),
        "raw_start": round(span["raw_start"], 3),
        "raw_end": round(span["raw_end"], 3),
        "accepted_total": span["accepted_total"],
        "alarm_frame_count": span["alarm_frame_count"],
        "max_risk": round(span["max_risk"], 4),
        "source_video": clip_source_video(stream),
        "input_video": stream.get("input_video", ""),
        "boxed_video": stream.get("output_video", ""),
        "frames": span["frames"],
    }


def build_browser_mp4_command(source, start, duration, target):
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]


def cut_clip(source, start, duration, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(build_browser_mp4_command(source, start, duration, target))


def clip_name(seg):
    return safe_name(f"{seg['rank']:03d}_s{seg['stream_index']}_{seg['start']:.1f}_{seg['end']:.1f}.mp4")


def assign_clip_paths(segments):
    for seg in segments:
        seg["clip"] = f"clips/{clip_name(seg)}"


def prepare_clips(dashboard_dir, segments):
    clips_dir = dashboard_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        source = Path(seg["source_video"])
        name = clip_name(seg)
        target = clips_dir / name
        cut_clip(source, float(seg["start"]), float(seg["duration"]), target)
        seg["clip"] = f"clips/{name}"


def summarize(segments, streams):
    views = sorted({seg["view"] for seg in segments})
    alarm_segments = [seg for seg in segments if seg["level"] == "alarm"]
    total_alarm_seconds = sum(seg["duration"] for seg in alarm_segments)
    max_risk = max([seg["max_risk"] for seg in segments] or [0.0])
    return {
        "event_count": len(segments),
        "alarm_event_count": len(alarm_segments),
        "view_count": len(views),
        "views": views,
        "total_alarm_seconds": round(total_alarm_seconds, 1),
        "max_risk": round(max_risk, 3),
        "stream_count": len(streams),
    }


def render_html(data):
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>监控防拍事件看板</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --ink: #151a1f;
  --muted: #637083;
  --line: #d9e0e7;
  --blue: #1269d3;
  --teal: #087f8c;
  --red: #d92d20;
  --amber: #b65f00;
  --green: #168a48;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Inter, "Microsoft YaHei", Arial, sans-serif; background: var(--bg); color: var(--ink); }}
header {{ height: 64px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; background: #111820; color: #fff; }}
header h1 {{ margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }}
header .meta {{ color: #b8c4d2; font-size: 13px; }}
.shell {{ display: grid; grid-template-columns: 280px minmax(420px, 1fr) 520px; gap: 16px; padding: 16px; min-height: calc(100vh - 64px); }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
.sidebar {{ padding: 16px; }}
.sidebar h2, .list h2, .player h2 {{ margin: 0 0 12px; font-size: 15px; }}
.stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 18px; }}
.stat {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfd; }}
.stat b {{ display: block; font-size: 22px; margin-bottom: 4px; }}
.stat span {{ color: var(--muted); font-size: 12px; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 7px 10px; cursor: pointer; font-size: 13px; }}
.chip.active {{ background: #e8f2ff; border-color: var(--blue); color: var(--blue); }}
.list {{ min-height: 0; display: flex; flex-direction: column; }}
.list-head {{ padding: 16px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; align-items: center; }}
.search {{ height: 34px; width: 190px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; }}
.cards {{ padding: 12px; overflow: auto; max-height: calc(100vh - 146px); display: grid; gap: 10px; }}
.event-card {{ border: 1px solid var(--line); border-left: 4px solid var(--amber); border-radius: 8px; background: #fff; padding: 12px; cursor: pointer; }}
.event-card.alarm {{ border-left-color: var(--red); }}
.event-card.active {{ outline: 2px solid var(--blue); }}
.event-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 8px; }}
.event-title {{ font-weight: 700; font-size: 15px; }}
.badge {{ padding: 3px 7px; border-radius: 999px; color: #fff; font-size: 12px; background: var(--amber); }}
.badge.alarm {{ background: var(--red); }}
.event-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; color: var(--muted); font-size: 12px; }}
.event-grid b {{ display: block; color: var(--ink); font-size: 14px; margin-top: 2px; }}
.timeline {{ height: 12px; background: #edf1f5; border-radius: 999px; overflow: hidden; margin-top: 10px; position: relative; }}
.timeline i {{ position: absolute; top: 0; bottom: 0; background: var(--amber); }}
.timeline i.alarm {{ background: var(--red); }}
.player {{ display: flex; flex-direction: column; min-width: 0; }}
.player-head {{ padding: 16px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
.player-title {{ font-size: 16px; font-weight: 700; }}
.player-sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
.toggle {{ display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12px; white-space: nowrap; }}
.player-actions {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }}
.stage {{ position: relative; background: #05080b; aspect-ratio: 16 / 9; overflow: hidden; }}
.stage:fullscreen {{ width: 100vw; height: 100vh; aspect-ratio: auto; background: #05080b; }}
.stage:-webkit-full-screen {{ width: 100vw; height: 100vh; aspect-ratio: auto; background: #05080b; }}
body.stage-fallback-fullscreen {{ overflow: hidden; }}
body.stage-fallback-fullscreen .stage {{ position: fixed; inset: 0; z-index: 10000; width: 100vw; height: 100vh; aspect-ratio: auto; border-radius: 0; }}
video {{ width: 100%; height: 100%; display: block; background: #05080b; object-fit: contain; }}
canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }}
.fs-btn {{ position: absolute; right: 12px; bottom: 12px; z-index: 4; border: 1px solid rgba(255,255,255,.35); background: rgba(8,13,18,.72); color: #fff; border-radius: 6px; height: 34px; padding: 0 12px; cursor: pointer; font-size: 13px; }}
.fs-btn:hover {{ background: rgba(18,105,211,.85); }}
.stage-hud {{ position: absolute; left: 12px; top: 12px; z-index: 3; max-width: min(640px, calc(100% - 24px)); padding: 8px 10px; border-radius: 6px; background: rgba(5,8,11,.72); color: #fff; font-size: 13px; line-height: 1.4; pointer-events: none; }}
.stage-hud b {{ color: #ffd2cc; }}
.detail {{ padding: 14px 16px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; border-top: 1px solid var(--line); }}
.kv {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
.kv span {{ display: block; color: var(--muted); font-size: 12px; }}
.kv b {{ font-size: 16px; }}
.empty {{ padding: 24px; color: var(--muted); }}
@media (max-width: 1200px) {{ .shell {{ grid-template-columns: 240px 1fr; }} .player {{ grid-column: 1 / -1; }} }}
@media (max-width: 760px) {{ header {{ padding: 0 14px; }} .shell {{ grid-template-columns: 1fr; padding: 10px; }} .stats, .detail {{ grid-template-columns: 1fr 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>监控防拍事件看板</h1>
  <div class="meta" id="runMeta"></div>
</header>
<main class="shell">
  <aside class="panel sidebar">
    <h2>事件概览</h2>
    <div class="stats" id="stats"></div>
    <h2>视角筛选</h2>
    <div class="filters" id="filters"></div>
  </aside>
  <section class="panel list">
    <div class="list-head">
      <h2>检测时间段</h2>
      <input class="search" id="search" placeholder="筛选视角/等级">
    </div>
    <div class="cards" id="cards"></div>
  </section>
  <section class="panel player">
    <div class="player-head">
      <div>
        <div class="player-title" id="playerTitle">未选择事件</div>
        <div class="player-sub" id="playerSub"></div>
      </div>
      <div class="player-actions">
        <label class="toggle"><input type="checkbox" id="overlayToggle" checked> 报警框</label>
        <label class="toggle"><input type="checkbox" id="riskOverlayToggle"> 候选框</label>
      </div>
    </div>
    <div class="stage" id="stage">
      <video id="video" controls controlslist="nofullscreen nodownload noremoteplayback" disablepictureinpicture playsinline></video>
      <canvas id="overlay"></canvas>
      <div class="stage-hud" id="stageHud"></div>
      <button class="fs-btn" id="fullscreenBtn" type="button">全屏</button>
    </div>
    <div class="detail" id="detail"></div>
  </section>
</main>
<script>
const DATA = {payload};
let activeView = "ALL";
let activeId = DATA.segments[0]?.id || null;
const video = document.getElementById("video");
const stage = document.getElementById("stage");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const hud = document.getElementById("stageHud");
let overlayLoopStarted = false;

function fmt(t) {{
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60).toString().padStart(2, "0");
  return `${{m}}:${{s}}`;
}}
function stat(label, value) {{ return `<div class="stat"><b>${{value}}</b><span>${{label}}</span></div>`; }}
function renderStats() {{
  const s = DATA.summary;
  document.getElementById("stats").innerHTML =
    stat("事件数", s.event_count) +
    stat("稳定报警", s.alarm_event_count) +
    stat("涉及视角", s.view_count) +
    stat("最高风险", s.max_risk.toFixed(2));
  document.getElementById("runMeta").textContent = `${{DATA.generated_at}} · ${{DATA.run_dir}}`;
}}
function renderFilters() {{
  const views = ["ALL", ...DATA.summary.views];
  document.getElementById("filters").innerHTML = views.map(v => {{
    const label = v === "ALL" ? "全部" : v;
    return `<button class="chip ${{activeView === v ? "active" : ""}}" data-view="${{v}}">${{label}}</button>`;
  }}).join("");
  document.querySelectorAll(".chip").forEach(btn => btn.onclick = () => {{
    activeView = btn.dataset.view;
    render();
  }});
}}
function filteredSegments() {{
  const q = document.getElementById("search").value.trim().toLowerCase();
  return DATA.segments.filter(seg => {{
    const viewOk = activeView === "ALL" || seg.view === activeView;
    const qOk = !q || (seg.view + seg.level).toLowerCase().includes(q);
    return viewOk && qOk;
  }});
}}
function renderCards() {{
  const rows = filteredSegments();
  const maxEnd = Math.max(...DATA.segments.map(s => s.end), 1);
  const box = document.getElementById("cards");
  if (!rows.length) {{
    box.innerHTML = `<div class="empty">没有匹配的事件</div>`;
    return;
  }}
  box.innerHTML = rows.map(seg => {{
    const left = Math.max(0, Math.min(100, seg.start / maxEnd * 100));
    const width = Math.max(1, Math.min(100 - left, seg.duration / maxEnd * 100));
    return `<article class="event-card ${{seg.level}} ${{seg.id === activeId ? "active" : ""}}" data-id="${{seg.id}}">
      <div class="event-top">
        <div class="event-title">${{seg.rank}}. ${{seg.view}} · ${{fmt(seg.start)}} - ${{fmt(seg.end)}}</div>
        <span class="badge ${{seg.level}}">${{seg.level === "alarm" ? "ALARM" : "RISK"}}</span>
      </div>
      <div class="event-grid">
        <span>时长<b>${{seg.duration.toFixed(1)}}s</b></span>
        <span>风险<b>${{seg.max_risk.toFixed(2)}}</b></span>
        <span>命中<b>${{seg.accepted_total}}</b></span>
        <span>报警帧<b>${{seg.alarm_frame_count}}</b></span>
      </div>
      <div class="timeline"><i class="${{seg.level}}" style="left:${{left}}%;width:${{width}}%"></i></div>
    </article>`;
  }}).join("");
  document.querySelectorAll(".event-card").forEach(card => card.onclick = () => {{
    activeId = card.dataset.id;
    render();
  }});
}}
function activeSegment() {{
  return DATA.segments.find(s => s.id === activeId) || DATA.segments[0];
}}
function renderPlayer() {{
  const seg = activeSegment();
  if (!seg) return;
  document.getElementById("playerTitle").textContent = `${{seg.view}} · ${{fmt(seg.start)}} - ${{fmt(seg.end)}}`;
  document.getElementById("playerSub").textContent = `${{seg.level.toUpperCase()}} · 来源 ${{seg.source_video}}`;
  const clipSrc = `${{seg.clip}}?v=${{encodeURIComponent(DATA.generated_at)}}`;
  if (!video.src.includes(seg.clip)) {{
    video.src = clipSrc;
  }}
  document.getElementById("detail").innerHTML = [
    ["片段时长", `${{seg.duration.toFixed(1)}}s`],
    ["最高风险", seg.max_risk.toFixed(2)],
    ["规则命中", seg.accepted_total],
    ["报警帧", seg.alarm_frame_count],
    ["原始开始", fmt(seg.raw_start)],
    ["原始结束", fmt(seg.raw_end)],
  ].map(([k, v]) => `<div class="kv"><span>${{k}}</span><b>${{v}}</b></div>`).join("");
}}
function resizeCanvas() {{
  const rect = stage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}}
function canvasCssSize() {{
  const ratio = window.devicePixelRatio || 1;
  return {{ width: canvas.width / ratio, height: canvas.height / ratio }};
}}
function videoContentBox() {{
  const rect = stage.getBoundingClientRect();
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const videoAspect = vw / vh;
  const stageAspect = rect.width / Math.max(1, rect.height);
  let width = rect.width, height = rect.height, x = 0, y = 0;
  if (stageAspect > videoAspect) {{
    height = rect.height;
    width = height * videoAspect;
    x = (rect.width - width) / 2;
  }} else {{
    width = rect.width;
    height = width / videoAspect;
    y = (rect.height - height) / 2;
  }}
  return {{ x, y, width, height }};
}}
function scaleForFrame(frame) {{
  const box = videoContentBox();
  return {{
    x: box.x,
    y: box.y,
    sx: box.width / Math.max(1, frame.width || video.videoWidth || 1),
    sy: box.height / Math.max(1, frame.height || video.videoHeight || 1),
  }};
}}
function px(point, scale) {{ return scale.x + point[0] * scale.sx; }}
function py(point, scale) {{ return scale.y + point[1] * scale.sy; }}
function drawPoly(points, scale, color, label) {{
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = "rgba(0,0,0,.75)";
  ctx.shadowBlur = 4;
  ctx.beginPath();
  points.forEach((p, i) => {{
    const x = px(p, scale), y = py(p, scale);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.closePath();
  ctx.stroke();
  if (label) {{
    ctx.fillStyle = color;
    ctx.fillRect(px(points[0], scale), py(points[0], scale) - 16, ctx.measureText(label).width + 8, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, px(points[0], scale) + 4, py(points[0], scale) - 4);
  }}
  ctx.restore();
}}
function drawRect(box, scale, color, label, thick = 2, fill = false) {{
  if (!box) return;
  const x = scale.x + box[0] * scale.sx;
  const y = scale.y + box[1] * scale.sy;
  const w = (box[2] - box[0]) * scale.sx;
  const h = (box[3] - box[1]) * scale.sy;
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,.85)";
  ctx.shadowBlur = 5;
  if (fill) {{
    ctx.fillStyle = color === "#d92d20" ? "rgba(217,45,32,.14)" : "rgba(182,95,0,.14)";
    ctx.fillRect(x, y, w, h);
  }}
  ctx.strokeStyle = color;
  ctx.lineWidth = thick;
  ctx.lineJoin = "round";
  ctx.strokeRect(x, y, w, h);
  if (label) {{
    ctx.font = "bold 12px Inter, sans-serif";
    const tw = ctx.measureText(label).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 4, Math.max(12, y - 5));
  }}
  ctx.restore();
}}
function isRiskPhone(phone) {{
  return !!(phone && (phone.accepted || phone.alarm || phone.level === "weak" || phone.level === "strong"));
}}
function showCandidateRisk() {{
  return !!document.getElementById("riskOverlayToggle")?.checked;
}}
function shouldDisplayPhone(phone, includeCandidates) {{
  return !!(phone && (phone.alarm || (includeCandidates && isRiskPhone(phone))));
}}
function shouldDisplayPerson(person, includeCandidates) {{
  return !!(person && (person.alarm || (includeCandidates && person.risk)));
}}
function getSegmentSubjects(seg, includeCandidates = false) {{
  const cacheKey = includeCandidates ? "_subjectsWithCandidates" : "_subjectsAlarmOnly";
  if (seg[cacheKey]) return seg[cacheKey];
  const subjects = new Map();
  for (const frame of seg.frames || []) {{
    for (const phone of frame.phones || []) {{
      if (!shouldDisplayPhone(phone, includeCandidates)) continue;
      const trackId = Number(phone.track_id);
      if (!Number.isFinite(trackId)) continue;
      const item = subjects.get(trackId) || {{ trackId, alarm: false, maxRisk: 0, screenIds: new Set() }};
      item.alarm = item.alarm || !!phone.alarm;
      item.maxRisk = Math.max(item.maxRisk, Number(phone.risk_score || 0));
      if (phone.screen_id) item.screenIds.add(phone.screen_id);
      subjects.set(trackId, item);
    }}
    for (const person of frame.persons || []) {{
      if (!shouldDisplayPerson(person, includeCandidates)) continue;
      const trackId = Number(person.track_id);
      if (!Number.isFinite(trackId)) continue;
      const item = subjects.get(trackId) || {{ trackId, alarm: false, maxRisk: 0, screenIds: new Set() }};
      item.alarm = item.alarm || !!person.alarm;
      item.maxRisk = Math.max(item.maxRisk, Number(person.risk_score || 0));
      if (person.screen_id) item.screenIds.add(person.screen_id);
      subjects.set(trackId, item);
    }}
  }}
  seg[cacheKey] = subjects;
  return subjects;
}}
function nearestFrame(seg, absolute) {{
  let best = null, delta = Infinity;
  for (const frame of seg.frames || []) {{
    const d = Math.abs(frame.time_sec - absolute);
    if (d < delta) {{ delta = d; best = frame; }}
  }}
  return best ? {{ frame: best, delta }} : null;
}}
function nearestPersonForTrack(seg, trackId, absolute) {{
  let bestFrame = null, bestPerson = null, delta = Infinity;
  for (const frame of seg.frames || []) {{
    for (const person of frame.persons || []) {{
      if (Number(person.track_id) !== Number(trackId)) continue;
      const d = Math.abs(frame.time_sec - absolute);
      if (d < delta) {{ delta = d; bestFrame = frame; bestPerson = person; }}
    }}
  }}
  return bestPerson ? {{ frame: bestFrame, person: bestPerson, delta }} : null;
}}
function drawOverlay() {{
  resizeCanvas();
  const canvasSize = canvasCssSize();
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);
  const seg = activeSegment();
  if (!seg) return;
  const absolute = seg.start + (video.currentTime || 0);
  if (!document.getElementById("overlayToggle").checked) {{
    hud.innerHTML = `<b>${{seg.level === "alarm" ? "??" : "??"}}</b> | ${{seg.view}} | ${{fmt(absolute)}}`;
    return;
  }}
  if (!video.videoWidth || !video.videoHeight) return;
  const includeCandidates = showCandidateRisk();
  const subjects = getSegmentSubjects(seg, includeCandidates);
  if (!subjects.size) {{
    hud.innerHTML = `<b>${{seg.level === "alarm" ? "??" : "??"}}</b> | ${{seg.view}} | ${{fmt(absolute)}}${{seg.level === "alarm" ? "" : " | ????"}}`;
    return;
  }}
  const nearest = nearestFrame(seg, absolute);
  const best = nearest ? nearest.frame : (seg.frames || [])[0];
  if (!best) return;
  const scale = scaleForFrame(best);
  ctx.font = "12px Inter, sans-serif";
  const riskyPhones = (best.phones || []).filter(phone =>
    subjects.has(Number(phone.track_id)) && shouldDisplayPhone(phone, includeCandidates)
  );
  const riskyScreenIds = new Set(riskyPhones.map(phone => phone.screen_id).filter(Boolean));
  for (const subject of subjects.values()) {{
    for (const screenId of subject.screenIds) {{
      riskyScreenIds.add(screenId);
    }}
  }}
  for (const screen of best.screens || []) {{
    if (riskyScreenIds.has(screen.screen_id)) {{
      drawPoly(screen.screen_poly, scale, "#1269d3", screen.screen_id || "SCREEN");
    }}
  }}
  const labels = [];
  for (const [trackId, subject] of subjects.entries()) {{
    const hit = nearestPersonForTrack(seg, trackId, absolute);
    if (!hit) continue;
    const person = hit.person;
    const color = subject.alarm ? "#d92d20" : "#b65f00";
    const status = subject.alarm ? "ALARM" : "CAND";
    const stale = hit.delta > 0.8 ? " TRACK" : "";
    const label = `P${{person.track_id > 0 ? person.track_id : person.person_index + 1}} ${{status}} ${{subject.maxRisk.toFixed(2)}}${{stale}}`;
    labels.push(label);
    drawRect(person.roi, scale, color, label, subject.alarm ? 5 : 4, true);
  }}
  for (const phone of riskyPhones) {{
    const color = phone.alarm ? "#d92d20" : "#b65f00";
    const label = `${{phone.alarm ? "ALARM" : "CAND"}} PHONE ${{Number(phone.risk_score || 0).toFixed(2)}}`;
    drawRect(phone.box, scale, color, label, phone.alarm ? 5 : 4, true);
  }}
  hud.innerHTML = labels.length
    ? `<b>${{seg.level === "alarm" ? "??" : "??"}}</b> | ${{seg.view}} | ${{fmt(absolute)}} | ${{labels.slice(0, 3).join(" / ")}}`
    : `<b>${{seg.level === "alarm" ? "??" : "??"}}</b> | ${{seg.view}} | ${{fmt(absolute)}}`;
}}
function startOverlayLoop() {{
  if (overlayLoopStarted) return;
  overlayLoopStarted = true;
  if (video.requestVideoFrameCallback) {{
    const tick = () => {{
      drawOverlay();
      video.requestVideoFrameCallback(tick);
    }};
    video.requestVideoFrameCallback(tick);
  }} else {{
    const tick = () => {{
      drawOverlay();
      window.requestAnimationFrame(tick);
    }};
    window.requestAnimationFrame(tick);
  }}
}}
function isStageFullscreen() {{
  return document.fullscreenElement === stage || document.body.classList.contains("stage-fallback-fullscreen");
}}
function updateFullscreenButton() {{
  document.getElementById("fullscreenBtn").textContent = isStageFullscreen() ? "退出全屏" : "全屏";
}}
function exitStageFullscreen() {{
  document.body.classList.remove("stage-fallback-fullscreen");
  if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen();
  updateFullscreenButton();
  setTimeout(drawOverlay, 80);
}}
function toggleFullscreen() {{
  if (isStageFullscreen()) {{
    exitStageFullscreen();
    return;
  }}
  if (stage.requestFullscreen) {{
    stage.requestFullscreen().catch(() => {{
      document.body.classList.add("stage-fallback-fullscreen");
      updateFullscreenButton();
      setTimeout(drawOverlay, 80);
    }});
  }} else {{
    document.body.classList.add("stage-fallback-fullscreen");
    updateFullscreenButton();
    setTimeout(drawOverlay, 80);
  }}
}}
function render() {{
  renderStats();
  renderFilters();
  renderCards();
  renderPlayer();
  drawOverlay();
}}
document.getElementById("search").addEventListener("input", renderCards);
video.addEventListener("timeupdate", drawOverlay);
video.addEventListener("loadedmetadata", drawOverlay);
video.addEventListener("seeked", drawOverlay);
video.addEventListener("play", startOverlayLoop);
video.addEventListener("dblclick", (event) => {{ event.preventDefault(); toggleFullscreen(); }});
window.addEventListener("resize", drawOverlay);
document.addEventListener("fullscreenchange", () => {{
  if (!document.fullscreenElement) document.body.classList.remove("stage-fallback-fullscreen");
  updateFullscreenButton();
  setTimeout(drawOverlay, 80);
}});
document.addEventListener("keydown", (event) => {{
  if (event.key === "Escape" && document.body.classList.contains("stage-fallback-fullscreen")) exitStageFullscreen();
}});
document.getElementById("overlayToggle").addEventListener("change", drawOverlay);
document.getElementById("riskOverlayToggle").addEventListener("change", drawOverlay);
document.getElementById("fullscreenBtn").addEventListener("click", toggleFullscreen);
render();
startOverlayLoop();
</script>
</body>
</html>
"""


def build_dashboard(run_dir, before, after, merge_gap, min_duration, skip_clips=False):
    run_dir = Path(run_dir)
    videos_dir = run_dir / "videos"
    events_path = videos_dir / "frame_events.jsonl"
    if not events_path.exists():
        events_path = run_dir / "frame_events.jsonl"
    events = load_jsonl(events_path)
    streams = load_streams(run_dir)
    segments = build_segments(events, streams, before, after, merge_gap, min_duration)
    dashboard_dir = run_dir / "dashboard"
    clips_backup = None
    if dashboard_dir.exists():
        clips_dir = dashboard_dir / "clips"
        if skip_clips and clips_dir.exists():
            clips_backup = run_dir / f".dashboard_clips_backup_{os.getpid()}"
            if clips_backup.exists():
                shutil.rmtree(clips_backup)
            shutil.move(str(clips_dir), str(clips_backup))
        shutil.rmtree(dashboard_dir)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    if clips_backup is not None:
        shutil.move(str(clips_backup), str(dashboard_dir / "clips"))
    if skip_clips:
        assign_clip_paths(segments)
        missing = [seg for seg in segments if not (dashboard_dir / seg["clip"]).exists()]
        if missing:
            prepare_clips(dashboard_dir, missing)
    else:
        prepare_clips(dashboard_dir, segments)
    data = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "summary": summarize(segments, streams),
        "streams": streams,
        "segments": segments,
    }
    (dashboard_dir / "events.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (dashboard_dir / "index.html").write_text(render_html(data), encoding="utf-8")
    return dashboard_dir, data


def self_test():
    streams = [{
        "stream_index": 0,
        "output_video": "demo_boxed.mp4",
        "input_video": "demo.mp4",
        "infer_fps": 10.0,
    }]
    events = [
        {"stream_index": 0, "time_sec": 10.0, "accepted_count": 1, "alarm_track_count": 0, "max_risk": 0.71},
        {"stream_index": 0, "time_sec": 12.5, "accepted_count": 1, "alarm_track_count": 1, "max_risk": 0.83},
        {"stream_index": 0, "time_sec": 22.0, "accepted_count": 1, "alarm_track_count": 0, "max_risk": 0.66},
    ]
    segs = build_segments(events, streams, before=3.0, after=5.0, merge_gap=4.0, min_duration=2.0)
    assert len(segs) == 2, segs
    assert segs[0]["start"] == 7.0 and segs[0]["end"] == 17.5, segs[0]
    assert segs[0]["level"] == "alarm", segs[0]
    assert segs[0]["source_video"] == "demo.mp4", segs[0]
    assert segs[0]["boxed_video"] == "demo_boxed.mp4", segs[0]
    assert segs[1]["start"] == 19.0 and segs[1]["end"] == 27.0, segs[1]
    assert summarize(segs, streams)["alarm_event_count"] == 1

    overlapping = [
        {"stream_index": 0, "time_sec": 10.0, "accepted_count": 1, "alarm_track_count": 0, "max_risk": 0.71},
        {"stream_index": 0, "time_sec": 18.0, "accepted_count": 1, "alarm_track_count": 0, "max_risk": 0.72},
    ]
    merged = build_segments(overlapping, streams, before=3.0, after=5.0, merge_gap=4.0, min_duration=2.0)
    assert len(merged) == 1, merged
    assert merged[0]["start"] == 7.0 and merged[0]["end"] == 23.0, merged[0]

    cmd = build_browser_mp4_command(Path("source.mp4"), 1.0, 2.0, Path("out.mp4"))
    assert "-c:v" in cmd and "libx264" in cmd, cmd
    assert "-pix_fmt" in cmd and "yuv420p" in cmd, cmd
    assert "-movflags" in cmd and "+faststart" in cmd, cmd
    assert clip_name({"rank": 3, "stream_index": 2, "start": 1.24, "end": 4.56}) == "003_s2_1.2_4.6.mp4"

    html_text = render_html({"generated_at": "test", "summary": {}, "filters": {}, "streams": [], "segments": []})
    assert "function getSegmentSubjects" in html_text
    assert "nearestPersonForTrack" in html_text
    assert "delta > 0.25" not in html_text
    assert 'id="stage"' in html_text
    assert 'controlslist="nofullscreen nodownload noremoteplayback"' in html_text
    assert "requestVideoFrameCallback" in html_text
    assert "scaleForFrame" in html_text
    assert "stage-fallback-fullscreen" in html_text
    assert "exitStageFullscreen" in html_text
    assert "riskOverlayToggle" in html_text
    assert "WARN" not in html_text
    print("[SELF_TEST] ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--before", type=float, default=3.0)
    parser.add_argument("--after", type=float, default=5.0)
    parser.add_argument("--merge-gap", type=float, default=4.0)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--skip-clips", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.run_dir:
        parser.error("--run-dir is required unless --self-test is used")
    dashboard_dir, data = build_dashboard(
        args.run_dir,
        before=args.before,
        after=args.after,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
        skip_clips=args.skip_clips,
    )
    print(f"[DASHBOARD] {dashboard_dir / 'index.html'}")
    print(f"[EVENTS] {data['summary']['event_count']} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
