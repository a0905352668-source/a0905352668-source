from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from realtime_sim.segment_producer import write_json_atomic

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>监控防拍实时复盘</title>
<link rel="icon" href="data:,">
<style>
:root{color-scheme:dark;--bg:#08111f;--panel:#111b2c;--panel2:#0d1626;--line:#26364f;--text:#e5edf8;--muted:#93a4ba;--blue:#38bdf8;--red:#ef4444;--amber:#f59e0b;--green:#22c55e}
*{box-sizing:border-box}body{margin:0;font-family:Arial,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text)}
header{height:58px;display:flex;align-items:center;justify-content:space-between;padding:0 18px;background:#0d1728;border-bottom:1px solid var(--line)}
h1,h2,h3{margin:0}h1{font-size:17px}.muted{color:var(--muted)}
.top{display:flex;align-items:center;gap:12px}.dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 16px rgba(34,197,94,.5)}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(118px,1fr));gap:10px;padding:12px 14px 0}.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}.kpi span{display:block;color:var(--muted);font-size:12px}.kpi b{font-size:22px}
.wrap{display:grid;grid-template-columns:360px minmax(560px,1fr) 320px;gap:14px;padding:12px 14px 14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;min-width:0}
.side,.ops{padding:12px;display:flex;flex-direction:column;gap:12px}.filters{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.filter{border:1px solid #334155;background:var(--panel2);color:#cbd5e1;border-radius:6px;padding:8px;cursor:pointer}.filter.active{border-color:var(--blue);color:white;background:#12304a}
.events{display:flex;flex-direction:column;gap:8px;max-height:calc(100vh - 232px);overflow:auto}.card{padding:10px;border-radius:8px;background:#0e1728;border:1px solid #32445f;cursor:pointer}.card.active{outline:2px solid var(--blue)}.card.alarm{border-color:rgba(239,68,68,.9)}.card.risk{border-color:rgba(245,158,11,.8)}
.line{display:flex;align-items:center;justify-content:space-between;gap:8px}.badge{font-weight:bold}.alarm .badge{color:#fca5a5}.risk .badge{color:#fcd34d}.meta{color:var(--muted);font-size:12px;margin-top:5px;line-height:1.5}
.player{padding:12px}.videoBox{position:relative;background:#020617;border-radius:8px;overflow:hidden;border:1px solid #1f2a3d;min-height:260px;display:flex;align-items:center;justify-content:center}.stage{position:relative;width:100%;line-height:0}video{width:100%;max-height:68vh;background:#000;display:block}#overlayCanvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}.empty{padding:36px;color:var(--muted);text-align:center}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:12px}.toolbtn{border:1px solid #334155;background:var(--panel2);color:#dbeafe;border-radius:6px;padding:7px 10px;cursor:pointer}
.detail{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}.detail div,.opbox{background:var(--panel2);border:1px solid #334155;border-radius:6px;padding:9px}.detail span,.opbox span{display:block;color:var(--muted);font-size:12px}.detail b,.opbox b{font-size:15px}.warn{color:#fcd34d}.ok{color:#86efac}.bad{color:#fca5a5}
@media(max-width:1180px){.kpis{grid-template-columns:repeat(3,1fr)}.wrap{grid-template-columns:1fr}.events{max-height:360px}.detail{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<header><div class="top"><span class="dot"></span><h1>监控防拍实时复盘</h1></div><span id="status" class="muted">加载中</span></header>
<section id="kpis" class="kpis"></section>
<div class="wrap">
<aside class="panel side">
  <h3>事件筛选</h3>
  <div class="filters">
    <button class="filter active" data-filter="all">全部</button>
    <button class="filter" data-filter="alarm">报警</button>
    <button class="filter" data-filter="risk">疑似</button>
    <button class="filter" data-filter="pending">待生成</button>
  </div>
  <h3>异常片段</h3>
  <section id="cards" class="events"></section>
</aside>
<main class="panel">
  <section class="player">
    <div class="videoBox"><div id="stage" class="stage"><video id="video" controls playsinline></video><canvas id="overlayCanvas"></canvas></div><div id="empty" class="empty">请选择左侧事件</div></div>
    <div class="toolbar"><h3 id="title">请选择左侧事件</h3><button id="fullscreenBtn" class="toolbtn">全屏</button></div>
    <div id="detail" class="detail"></div>
  </section>
</main>
<aside class="panel ops"><h3>运行诊断</h3><div id="ops"></div></aside>
</div>
<script>
async function loadJson(path){const r=await fetch(path+'?t='+Date.now());if(!r.ok)throw new Error(path);return await r.json();}
function fmt(t){const n=Number(t||0);const m=Math.floor(n/60);const s=Math.floor(n%60).toString().padStart(2,'0');return `${m}:${s}`;}
let currentFilter='all';let currentSegments=[];let selectedSegmentId='';
let currentOverlay={frames:[],width:0,height:0};let currentParts=[];let currentPartIndex=0;const OVERLAY_HOLD_SECONDS=2.0;
const 状态文本={starting:'启动中',running:'运行中',complete:'已完成',failed:'失败'};
const 等级文本={alarm:'报警',risk:'疑似'};
const 片段状态={ready:'可播放',pending:'待生成',failed:'生成失败'};
const 视角名称=['电气1','电气2','机械1','机械2','软件1','软件2','走廊'];
function eventName(id){return '事件 '+String(id||'').replace('event_','');}
function queueStats(segs){return {ready:segs.filter(s=>s.clip_status==='ready').length,pending:segs.filter(s=>s.clip_status!=='ready').length,alarm:segs.filter(s=>s.level==='alarm').length,risk:segs.filter(s=>s.level==='risk').length};}
function fpsText(v){const n=Number(v);return Number.isFinite(n)&&n>0?`${n.toFixed(1)} FPS`:'--';}
function renderSummary(state,segs,metrics={}){const latest=metrics.latest||{};const q=queueStats(segs);const fps=fpsText(state.last_aggregate_fps||latest.aggregate_fps);const path=state.realtime_path||latest.realtime_path||'metadata_only';const clipMode=state.clip_build_mode||latest.clip_build_mode||'background';const kpis=[['状态',状态文本[state.status]||state.status||'等待'],['事件总数',state.event_count||0],['异常片段',segs.length],['报警片段',q.alarm],['队列状态',`${q.ready}/${segs.length} 可播放`],['检测吞吐',fps]];document.getElementById('kpis').innerHTML=kpis.map(([k,v])=>`<div class="kpi"><span>${k}</span><b>${v}</b></div>`).join('');document.getElementById('ops').innerHTML=`<div class="opbox"><span>推理进度</span><b>${state.processed_batches||0}/${state.target_batches||0} 批</b></div><div class="opbox"><span>上批耗时</span><b>${Number(state.last_batch_sec||0).toFixed(1)} 秒</b></div><div class="opbox"><span>实时链路</span><b>${path==='metadata_only'?'仅元数据':'带框视频'}</b></div><div class="opbox"><span>复盘生成</span><b>${clipMode==='background'?'后台队列':clipMode}</b></div><div class="opbox"><span>坐标叠加</span><b>${q.ready} 可播放 / ${q.pending} 待生成</b></div><div class="opbox"><span>最近吞吐</span><b>${fps}</b></div>`;document.getElementById('status').textContent='最后更新 '+new Date().toLocaleTimeString();}
async function loadOverlay(path){if(!path){currentOverlay={frames:[],width:0,height:0};return;}try{currentOverlay=await loadJson(path);}catch(e){currentOverlay={frames:[],width:0,height:0};}}
function resizeCanvas(){const video=document.getElementById('video');const canvas=document.getElementById('overlayCanvas');canvas.width=Math.max(1,video.clientWidth);canvas.height=Math.max(1,video.clientHeight);}
function nearestFrame(t){const frames=currentOverlay.frames||[];let best=null;let bestDiff=OVERLAY_HOLD_SECONDS;for(const frame of frames){const d=Math.abs(Number(frame.t||0)-t);if(d<bestDiff){best=frame;bestDiff=d;}}return best;}
function drawBox(ctx,item,color,label,scaleX,scaleY){const box=item.box||item;let x,y,w,h;if(item.box_norm&&item.box_norm.length===4){x=item.box_norm[0]*ctx.canvas.width;y=item.box_norm[1]*ctx.canvas.height;w=(item.box_norm[2]-item.box_norm[0])*ctx.canvas.width;h=(item.box_norm[3]-item.box_norm[1])*ctx.canvas.height;}else{x=box[0]*scaleX;y=box[1]*scaleY;w=(box[2]-box[0])*scaleX;h=(box[3]-box[1])*scaleY;}ctx.strokeStyle=color;ctx.lineWidth=3;ctx.strokeRect(x,y,w,h);ctx.font='14px Arial';const tw=ctx.measureText(label).width+8;ctx.fillStyle=color;ctx.fillRect(x,Math.max(0,y-22),tw,20);ctx.fillStyle='#fff';ctx.fillText(label,x+4,Math.max(14,y-7));}
function drawOverlay(){const video=document.getElementById('video');const canvas=document.getElementById('overlayCanvas');resizeCanvas();const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(video.paused&&video.readyState<2){return;}const frame=nearestFrame(video.currentTime);if(!frame){return;}const srcW=currentOverlay.width||video.videoWidth||canvas.width;const srcH=currentOverlay.height||video.videoHeight||canvas.height;const sx=canvas.width/srcW,sy=canvas.height/srcH;for(const p of frame.persons||[]){const color=p.status==='alarm'?'#ef4444':'#f59e0b';const state=p.status==='alarm'?'报警':'主角';const label=`Person ${p.track_id??''} ${state}`;drawBox(ctx,p,color,label,sx,sy);}for(const ph of frame.phones||[]){drawBox(ctx,ph,'#ef4444',`手机 ${Number(ph.confidence||0).toFixed(2)}`,sx,sy);}}
function overlayLoop(){drawOverlay();requestAnimationFrame(overlayLoop);}
function enforcePlaybackWindow(){const video=document.getElementById('video');const part=currentParts[currentPartIndex];if(!part||video.readyState<1){return;}const start=Number(part.start||0);const end=Number(part.end||video.duration||0);if(video.ended||video.currentTime<start-0.25||video.currentTime>end+0.25){video.currentTime=start;drawOverlay();}}
function playPart(index,autoplay,forceSeek=true){const video=document.getElementById('video');const empty=document.getElementById('empty');const part=currentParts[index];if(!part){return;}currentPartIndex=index;video.style.display='block';empty.style.display='none';const targetVideo=part.video;const sameSource=video.dataset.source===targetVideo&&video.readyState>0;video.onloadedmetadata=()=>{if(forceSeek){video.currentTime=Number(part.start||0);}drawOverlay();if(autoplay){video.play().catch(()=>{});}};video.onended=()=>{if(currentPartIndex+1<currentParts.length){playPart(currentPartIndex+1,true,true);}else{enforcePlaybackWindow();}};loadOverlay(part.overlay);if(sameSource){if(forceSeek){video.currentTime=Number(part.start||0);}drawOverlay();return;}video.dataset.source=targetVideo;video.src=targetVideo;video.load();}
function select(seg,userInitiated=false){const sameSelection=selectedSegmentId===seg.id;selectedSegmentId=seg.id;document.querySelectorAll('.card').forEach(el=>el.classList.toggle('active',el.dataset.id===seg.id));document.getElementById('title').textContent=`${eventName(seg.id)} | ${视角名称[seg.stream_index]||('视角'+seg.stream_index)} | ${fmt(seg.start)}-${fmt(seg.end)}`;document.getElementById('detail').innerHTML=`<div><span>事件等级</span><b>${等级文本[seg.level]||seg.level}</b></div><div><span>风险分</span><b>${Number(seg.max_risk||0).toFixed(2)}</b></div><div><span>事件帧数</span><b>${seg.event_count||0}</b></div><div><span>复盘状态</span><b class="${seg.clip_status==='ready'?'ok':'warn'}">${seg.media_status==='batch_ready'?'坐标叠加':(片段状态[seg.clip_status]||'等待中')}</b></div>`;const video=document.getElementById('video');const empty=document.getElementById('empty');const parts=[];if(seg.clip){parts.push({video:seg.clip,start:0,end:Math.max(0,Number(seg.end||0)-Number(seg.start||0)),overlay:''});}if(!parts.length){for(const p of seg.playback_parts||[]){parts.push({video:p.video,start:p.start||0,end:p.end||0,overlay:p.overlay||seg.overlay});}}if(!parts.length&&seg.playback_video){parts.push({video:seg.playback_video,start:seg.playback_start||0,end:seg.boxed_end||0,overlay:seg.overlay});}currentParts=parts;if(parts.length){if(!sameSelection||userInitiated){playPart(0,false,true);}return;}video.pause();video.removeAttribute('src');delete video.dataset.source;video.load();video.style.display='none';empty.style.display='block';empty.textContent='该事件暂无可播放视频';currentOverlay={frames:[],width:0,height:0};drawOverlay();}
function filtered(segs){let rows=[...segs].sort((a,b)=>(a.level==='alarm'?0:1)-(b.level==='alarm'?0:1)||Number(a.start||0)-Number(b.start||0));if(currentFilter==='alarm')rows=rows.filter(s=>s.level==='alarm');if(currentFilter==='risk')rows=rows.filter(s=>s.level==='risk');if(currentFilter==='pending')rows=rows.filter(s=>s.clip_status!=='ready');return rows;}
function renderCards(segs){currentSegments=segs;const rows=filtered(segs);const selectedVisible=rows.some(seg=>seg.id===selectedSegmentId);document.getElementById('cards').innerHTML=rows.map(seg=>`<article class="card ${seg.level} ${seg.id===selectedSegmentId?'active':''}" data-id="${seg.id}"><div class="line"><b>${eventName(seg.id)}</b><span class="badge">${等级文本[seg.level]||seg.level}</span></div><div class="meta">${视角名称[seg.stream_index]||('视角'+seg.stream_index)} | ${fmt(seg.start)}-${fmt(seg.end)} | ${片段状态[seg.clip_status]||'等待中'} | 风险分 ${Number(seg.max_risk||0).toFixed(2)}</div></article>`).join('')||'<div class="muted">当前筛选下暂无事件</div>';document.querySelectorAll('.card').forEach((el,i)=>el.onclick=()=>select(rows[i],true));if(rows.length&&!selectedVisible){select(rows[0]);}}
document.querySelectorAll('.filter').forEach(btn=>btn.onclick=()=>{currentFilter=btn.dataset.filter;document.querySelectorAll('.filter').forEach(x=>x.classList.toggle('active',x===btn));renderCards(currentSegments);});
document.getElementById('fullscreenBtn').onclick=()=>{const stage=document.getElementById('stage');if(stage.requestFullscreen){stage.requestFullscreen();}};
window.addEventListener('resize',resizeCanvas);document.getElementById('video').addEventListener('timeupdate',drawOverlay);document.getElementById('video').addEventListener('play',enforcePlaybackWindow);document.getElementById('video').addEventListener('loadedmetadata',resizeCanvas);overlayLoop();
async function tick(){try{const state=await loadJson('api/run_state.json');const segPayload=await loadJson('api/segments.json');let metrics={latest:{}};try{metrics=await loadJson('api/metrics.json');}catch(e){}const segs=segPayload.segments||[];renderSummary(state,segs,metrics);renderCards(segs);}catch(e){document.getElementById('status').textContent='等待数据';}}
setInterval(tick,3000);tick();
</script>
</body>
</html>
"""


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _segments_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        segments = value.get("segments", [])
        return {**value, "segments": segments if isinstance(segments, list) else []}
    if isinstance(value, list):
        return {"segments": value}
    return {"segments": []}


def _publish_media(source: Path, target: Path) -> bool:
    if not source.exists() or source.stat().st_size <= 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return True
    try:
        os.link(source, target)
        return True
    except OSError:
        pass
    try:
        target.symlink_to(source)
        return True
    except OSError:
        pass
    shutil.copy2(source, target)
    return True


def _stream_index_from_name(path: Path) -> int | None:
    parts = path.stem.split("_")
    if len(parts) >= 2 and parts[0] == "stream":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _load_batches(run_dir: Path) -> list[dict[str, Any]]:
    batches = []
    for batch_dir in sorted((run_dir / "incoming").glob("batch_*")):
        batch = _load_json(batch_dir / "batch.json", {})
        try:
            batch_index = int(batch.get("batch_index", batch_dir.name.split("_")[-1]))
        except ValueError:
            continue
        files: dict[int, Path] = {}
        for item in batch.get("files", []):
            try:
                stream_index = int(item.get("stream_index"))
            except (TypeError, ValueError):
                continue
            segment_file = Path(str(item.get("segment_file", "")))
            if segment_file.exists():
                files[stream_index] = segment_file
        for mp4 in sorted(batch_dir.glob("stream_*.mp4")):
            stream_index = _stream_index_from_name(mp4)
            if stream_index is not None:
                files.setdefault(stream_index, mp4)
        batches.append({
            "batch_index": batch_index,
            "name": batch_dir.name,
            "start": float(batch.get("global_start_sec", batch_index * _segment_seconds(run_dir))),
            "duration": float(batch.get("duration_sec", _segment_seconds(run_dir))),
            "files": files,
        })
    return batches


def _segment_seconds(run_dir: Path) -> float:
    manifest = _load_json(run_dir / "source_manifest.json", {})
    return float(manifest.get("segment_seconds", 10.0) or 10.0)


def _status_for_person(person: dict[str, Any]) -> str | None:
    score = float(person.get("risk_score", 0.0) or 0.0)
    if person.get("alarm"):
        return "alarm"
    if person.get("risk") or score >= 0.55:
        return "risk"
    if person.get("suspect"):
        return "suspect"
    return None


def _box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [round(float(x), 2) for x in value]
    except (TypeError, ValueError):
        return None


def _box_norm(box: list[float], width: int, height: int) -> list[float] | None:
    if width <= 0 or height <= 0:
        return None
    return [
        round(max(0.0, min(1.0, box[0] / width)), 6),
        round(max(0.0, min(1.0, box[1] / height)), 6),
        round(max(0.0, min(1.0, box[2] / width)), 6),
        round(max(0.0, min(1.0, box[3] / height)), 6),
    ]


def _attach_norm(payload: dict[str, Any], box: list[float], width: int, height: int) -> dict[str, Any]:
    norm = _box_norm(box, width, height)
    if norm is not None:
        payload["box_norm"] = norm
    return payload


def _overlay_frame(item: dict[str, Any], target_tracks: set[int] | None = None, width: int = 0, height: int = 0) -> dict[str, Any] | None:
    target_tracks = target_tracks or set()
    persons = []
    phones = []
    for person in item.get("persons", []):
        status = _status_for_person(person)
        track_id = int(person.get("track_id", -1) or -1)
        if not status and track_id in target_tracks:
            status = "track"
        box = _box(person.get("box"))
        if status and box:
            persons.append(_attach_norm({
                "track_id": track_id,
                "status": status,
                "risk_score": round(float(person.get("risk_score", 0.0) or 0.0), 4),
                "box": box,
            }, box, width, height))
    for phone in item.get("phones", []):
        box = _box(phone.get("box"))
        if box and (phone.get("accepted") or phone.get("alarm")):
            phones.append(_attach_norm({
                "track_id": int(phone.get("track_id", -1) or -1),
                "confidence": round(float(phone.get("confidence", 0.0) or 0.0), 4),
                "box": box,
            }, box, width, height))
    if not persons and not phones:
        return None
    frame_index = int(item.get("frame_index", item.get("frame_id", 0)) or 0)
    infer_fps = float(item.get("infer_fps", 10.0) or 10.0)
    local_time = float(item.get("time_sec", frame_index / max(infer_fps, 1e-6)) or 0.0)
    return {
        "t": round(local_time, 3),
        "frame_index": frame_index,
        "persons": persons,
        "phones": phones,
    }


def _iter_frame_event_items(run_dir: Path, batch: dict[str, Any]):
    processed = run_dir / "processed" / batch["name"] / "videos"
    for jsonl_path in sorted(processed.glob("frame_events.jsonl")):
        with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)


def _item_local_time(item: dict[str, Any]) -> float:
    frame_index = int(item.get("frame_index", item.get("frame_id", 0)) or 0)
    infer_fps = float(item.get("infer_fps", 10.0) or 10.0)
    return float(item.get("time_sec", frame_index / max(infer_fps, 1e-6)) or 0.0)


def _candidate_track_ids(item: dict[str, Any]) -> list[int]:
    track_ids = []
    for phone in item.get("phones", []):
        if phone.get("accepted") or phone.get("alarm"):
            try:
                track_id = int(phone.get("track_id", -1) or -1)
            except (TypeError, ValueError):
                continue
            if track_id >= 0:
                track_ids.append(track_id)
    for person in item.get("persons", []):
        status = _status_for_person(person)
        try:
            track_id = int(person.get("track_id", -1) or -1)
        except (TypeError, ValueError):
            continue
        if status and track_id >= 0:
            track_ids.append(track_id)
    return track_ids


def _assign_segment_target_tracks(run_dir: Path, batches: list[dict[str, Any]], payload: dict[str, Any]) -> dict[tuple[int, int], set[int]]:
    target_tracks_by_key: dict[tuple[int, int], set[int]] = defaultdict(set)
    segments = [dict(seg) for seg in payload.get("segments", [])]
    for seg in segments:
        try:
            stream_index = int(seg.get("stream_index"))
        except (TypeError, ValueError):
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        counts: Counter[int] = Counter()
        overlapping_batches = []
        for batch in batches:
            batch_start = float(batch["start"])
            batch_end = batch_start + float(batch["duration"])
            if batch_end < start or batch_start > end:
                continue
            overlapping_batches.append(batch)
            for item in _iter_frame_event_items(run_dir, batch):
                if int(item.get("stream_index", -1) or -1) != stream_index:
                    continue
                global_t = batch_start + _item_local_time(item)
                if start <= global_t <= end:
                    counts.update(_candidate_track_ids(item))
        if counts:
            target_track_id = counts.most_common(1)[0][0]
            seg["target_track_id"] = target_track_id
            for batch in overlapping_batches:
                target_tracks_by_key[(int(batch["batch_index"]), stream_index)].add(target_track_id)
    payload["segments"] = segments
    return target_tracks_by_key


def _write_overlays(run_dir: Path, dashboard: Path, batches: list[dict[str, Any]], target_tracks_by_key: dict[tuple[int, int], set[int]]) -> dict[tuple[int, int], str]:
    overlay_root = dashboard / "api" / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    frames_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    size_by_key: dict[tuple[int, int], tuple[int, int]] = {}
    overlay_map: dict[tuple[int, int], str] = {}
    for batch in batches:
        batch_index = int(batch["batch_index"])
        processed = run_dir / "processed" / batch["name"] / "videos"
        streams = _load_json(processed / "streams.json", [])
        for stream in streams if isinstance(streams, list) else []:
            try:
                key = (batch_index, int(stream.get("stream_index")))
            except (TypeError, ValueError):
                continue
            size_by_key[key] = (int(stream.get("width", 0) or 0), int(stream.get("height", 0) or 0))
        for item in _iter_frame_event_items(run_dir, batch):
            try:
                stream_index = int(item.get("stream_index"))
            except (TypeError, ValueError):
                continue
            key = (batch_index, stream_index)
            width, height = size_by_key.get(key, (int(item.get("width", 0) or 0), int(item.get("height", 0) or 0)))
            if width <= 0 or height <= 0:
                width, height = int(item.get("width", 0) or 0), int(item.get("height", 0) or 0)
            frame = _overlay_frame(item, target_tracks_by_key.get(key, set()), width, height)
            if frame:
                frames_by_key[key].append(frame)
                size_by_key.setdefault(key, (width, height))
        for stream_index in batch["files"]:
            key = (batch_index, int(stream_index))
            width, height = size_by_key.get(key, (0, 0))
            rel = f"api/overlays/{batch['name']}_stream_{stream_index}.json"
            overlay_map[key] = rel
            write_json_atomic(dashboard / rel, {
                "schema_version": "jk-overlay-v1",
                "coordinate_space": "pixel_with_normalized_copy",
                "batch_index": batch_index,
                "stream_index": int(stream_index),
                "width": width,
                "height": height,
                "frames": sorted(frames_by_key.get(key, []), key=lambda item: float(item.get("t", 0.0))),
            })
    return overlay_map


def _publish_batch_media(run_dir: Path, dashboard: Path, batches: list[dict[str, Any]]) -> dict[tuple[int, int], str]:
    media_map: dict[tuple[int, int], str] = {}
    for batch in batches:
        for stream_index, source in batch["files"].items():
            target = dashboard / "media" / batch["name"] / source.name
            if _publish_media(source, target):
                rel = target.relative_to(dashboard).as_posix()
                media_map[(int(batch["batch_index"]), int(stream_index))] = rel
    return media_map


def _overlapping_parts(seg: dict[str, Any], batches: list[dict[str, Any]], media_map: dict[tuple[int, int], str], overlay_map: dict[tuple[int, int], str]) -> list[dict[str, Any]]:
    try:
        stream_index = int(seg.get("stream_index"))
    except (TypeError, ValueError):
        return []
    start = float(seg.get("start", 0.0) or 0.0)
    end = max(start, float(seg.get("end", start) or start))
    parts = []
    for batch in batches:
        batch_start = float(batch["start"])
        batch_end = batch_start + float(batch["duration"])
        if batch_end < start or batch_start > end:
            continue
        key = (int(batch["batch_index"]), stream_index)
        video = media_map.get(key)
        if not video:
            continue
        parts.append({
            "video": video,
            "overlay": overlay_map.get(key, ""),
            "batch_index": int(batch["batch_index"]),
            "start": round(max(0.0, start - batch_start), 3),
            "end": round(min(float(batch["duration"]), end - batch_start), 3),
        })
    return parts


def _sync_existing_clips(dashboard: Path, payload: dict[str, Any]) -> dict[str, Any]:
    clips_dir = dashboard / "clips"
    segments = []
    for item in payload.get("segments", []):
        seg = dict(item)
        seg_id = str(seg.get("id", ""))
        clip = clips_dir / f"{seg_id}.mp4"
        if seg_id and clip.exists() and clip.stat().st_size > 0:
            seg["clip"] = f"clips/{clip.name}"
            seg["boxed_video"] = f"clips/{clip.name}"
            seg["clip_status"] = "ready"
        else:
            seg.setdefault("clip", "")
            seg.setdefault("boxed_video", "")
            seg["clip_status"] = seg.get("clip_status") or "pending"
        segments.append(seg)
    return {**payload, "segments": segments}


def _sync_batch_playback(run_dir: Path, dashboard: Path, payload: dict[str, Any]) -> dict[str, Any]:
    batches = _load_batches(run_dir)
    target_tracks_by_key = _assign_segment_target_tracks(run_dir, batches, payload)
    media_map = _publish_batch_media(run_dir, dashboard, batches)
    overlay_map = _write_overlays(run_dir, dashboard, batches, target_tracks_by_key)
    segments = []
    for item in payload.get("segments", []):
        seg = dict(item)
        parts = _overlapping_parts(seg, batches, media_map, overlay_map)
        if parts:
            first = parts[0]
            seg["playback_parts"] = parts
            seg["playback_video"] = first["video"]
            seg["playback_start"] = first["start"]
            seg["overlay"] = first.get("overlay", "")
            seg["media_status"] = "batch_ready"
            seg["clip_status"] = "ready"
        else:
            seg.setdefault("playback_parts", [])
            seg.setdefault("playback_video", "")
            seg.setdefault("playback_start", 0.0)
            seg.setdefault("overlay", "")
            seg.setdefault("media_status", "missing")
        segments.append(seg)
    return {**payload, "segments": segments}


def _load_metrics_payload(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "events" / "metrics.jsonl"
    history: list[dict[str, Any]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                history.append(item)
    history = history[-100:]
    return {
        "schema_version": "jk-runtime-metrics-v1",
        "latest": history[-1] if history else {},
        "history": history,
    }


def write_dashboard_api(run_dir: Path) -> None:
    dashboard = run_dir / "dashboard"
    api = dashboard / "api"
    api.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "events" / "state.json"
    segments_path = run_dir / "events" / "segments.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"event_count": 0}
    raw_segments = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else {"segments": []}
    segments = _sync_batch_playback(run_dir, dashboard, _sync_existing_clips(dashboard, _segments_payload(raw_segments)))
    write_json_atomic(api / "run_state.json", state)
    write_json_atomic(api / "segments.json", segments)
    write_json_atomic(api / "metrics.json", _load_metrics_payload(run_dir))
    (dashboard / "index.html").write_text(INDEX_HTML, encoding="utf-8")
