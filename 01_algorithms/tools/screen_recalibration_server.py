#!/usr/bin/env python3
"""Small browser-based screen recalibration tool for the JianKong views."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from PIL import Image, ImageDraw


PROJECT = Path("/media/boshi/Data/JianKong")
VIDEO_ROOT = PROJECT / "03_raw_videos_and_frames/2026-07-03"
CONFIG_DIR = PROJECT / "02_configs/surveillance"
WORK_DIR = CONFIG_DIR / "recalibration_20260706"
DRAFT_PATH = WORK_DIR / "calibration_draft.json"

VIEWS = [
    ("dianqi1", "电气1", "camera_01", "camera_01_screen_calibration_v21.json"),
    ("dianqi2", "电气2", "camera_02", "camera_02_screen_calibration_v21.json"),
    ("jixie1", "机械1", "camera_mechanical_01", "camera_mechanical_01_screen_calibration_v21.json"),
    ("jixie2", "机械2", "camera_mechanical_02", "camera_mechanical_02_screen_calibration_v21.json"),
    ("ruanjian1", "软件1", "camera_software_01", "camera_software_01_screen_calibration_v21.json"),
    ("ruanjian2", "软件2", "camera_software_02", "camera_software_02_screen_calibration_v21.json"),
    ("zoulang", "走廊", "camera_corridor", "camera_corridor_screen_calibration_v21.json"),
]

DEFAULT_PARAMS = {
    "enable_dynamic_person_zone": True,
    "person_expand_x": 0.35,
    "person_expand_y": 0.20,
    "near_zone_detect_interval": 10,
    "phone_valid_conf_person_roi": 0.35,
    "phone_valid_conf_near_zone": 0.50,
    "phone_strong_conf": 0.55,
    "ignore_center_inside": True,
    "angle_thresh": 90.0,
    "angle_relaxed_thresh": 110.0,
    "hand_radius_ratio": 0.18,
    "hand_radius_min": 40.0,
    "corridor_width_ratio": 0.35,
    "corridor_width_min": 80.0,
    "alert_counter_threshold": 10.0,
    "person_state_window": 30,
    "person_state_min_hits": 16,
    "person_state_risk_threshold": 0.65,
    "person_state_alarm_risk": 0.90,
    "occluded_enable": False,
    "near_screen_override": False,
}


def find_latest_video(key: str) -> Path | None:
    videos: list[Path] = []
    for subdir in VIDEO_ROOT.iterdir():
        if subdir.is_dir() and key in subdir.name.lower():
            videos.extend(p for p in subdir.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"})
    return max(videos, key=lambda p: (p.stat().st_mtime, p.name)) if videos else None


def run_json(cmd: list[str]) -> dict:
    return json.loads(subprocess.check_output(cmd, text=True))


def extract_frames() -> list[dict]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for key, label, camera_id, filename in VIEWS:
        image_path = WORK_DIR / f"{key}_middle_frame.jpg"
        video = find_latest_video(key)
        if video is None:
            manifest.append({"view": key, "label": label, "camera_id": camera_id, "filename": filename, "error": "video_not_found"})
            continue
        if not image_path.exists():
            duration_text = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
                text=True,
            ).strip()
            try:
                ss = max(0.0, float(duration_text.splitlines()[0]) / 2.0)
            except Exception:
                ss = 10.0
            subprocess.check_call(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{ss:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(image_path)]
            )
        stream = run_json(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(video)]
        )["streams"][0]
        manifest.append(
            {
                "view": key,
                "label": label,
                "camera_id": camera_id,
                "filename": filename,
                "video": str(video),
                "image": image_path.name,
                "width": int(stream["width"]),
                "height": int(stream["height"]),
            }
        )
    (WORK_DIR / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def old_points(filename: str) -> list[list[float]]:
    path = CONFIG_DIR / filename
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    screens = data.get("screens", [])
    if not screens:
        return []
    return [[float(x), float(y)] for x, y in screens[0].get("screen_poly", [])]


def load_draft(manifest: list[dict]) -> dict:
    if DRAFT_PATH.exists():
        draft = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))
        changed = False
        for item in manifest:
            view = item["view"]
            record = draft.setdefault("views", {}).setdefault(view, {})
            if normalize_record(record):
                changed = True
        if changed:
            save_draft(draft)
        return draft
    views = {}
    for item in manifest:
        pts = old_points(item["filename"])
        views[item["view"]] = {
            "screens": [{"screen_id": "screen_01", "points": pts}] if pts else [],
            "active_screen": 0,
            "no_screen": not bool(pts),
            "saved": False,
        }
    draft = {"updated_at": time.strftime("%F %T"), "views": views}
    save_draft(draft)
    return draft


def normalize_record(record: dict) -> bool:
    """Migrate the first-version one-screen draft into a multi-screen draft."""
    if "screens" in record:
        record.setdefault("active_screen", 0)
        record.setdefault("no_screen", not bool(record.get("screens")))
        return False
    points = record.pop("points", [])
    no_screen = bool(record.get("no_screen")) or not bool(points)
    record["screens"] = [] if no_screen else [{"screen_id": "screen_01", "points": points}]
    record["active_screen"] = 0
    record["no_screen"] = no_screen
    return True


def save_draft(draft: dict) -> None:
    draft["updated_at"] = time.strftime("%F %T")
    DRAFT_PATH.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def rect_poly(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)], [round(x1, 2), round(y2, 2)]]


def expand_box(box: tuple[float, float, float, float], xr: float, yr: float, width: int, height: int) -> list[list[float]]:
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * xr
    dy = (y2 - y1) * yr
    return rect_poly(clamp(x1 - dx, 0, width - 1), clamp(y1 - dy, 0, height - 1), clamp(x2 + dx, 0, width - 1), clamp(y2 + dy, 0, height - 1))


def params_from_old(filename: str) -> dict:
    params = dict(DEFAULT_PARAMS)
    path = CONFIG_DIR / filename
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        screens = data.get("screens", [])
        if screens:
            params.update(screens[0].get("params", {}))
    return params


def build_config(item: dict, record: dict) -> dict:
    width = int(item.get("width") or 2560)
    height = int(item.get("height") or 1440)
    screens = []
    if not record.get("no_screen"):
        for idx, screen_record in enumerate(record.get("screens", []), start=1):
            points = [[round(float(x), 2), round(float(y), 2)] for x, y in screen_record.get("points", [])]
            if len(points) != 4:
                continue
            b = bbox(points)
            front = expand_box(b, 0.65, 0.55, width, height)
            fx1, fy1, fx2, fy2 = bbox(front)
            left = rect_poly(fx1, fy1 + (fy2 - fy1) * 0.15, fx1 + (fx2 - fx1) * 0.55, fy2 - (fy2 - fy1) * 0.05)
            right = rect_poly(fx1 + (fx2 - fx1) * 0.45, fy1 + (fy2 - fy1) * 0.15, fx2, fy2 - (fy2 - fy1) * 0.05)
            screens.append(
                {
                    "screen_id": screen_record.get("screen_id") or f"screen_{idx:02d}",
                    "screen_poly": points,
                    "near_zone": expand_box(b, 1.20, 0.90, width, height),
                    "danger_zones": [
                        {"name": "front", "polygon": front, "weight": 1.0},
                        {"name": "left_side", "polygon": left, "weight": 0.8},
                        {"name": "right_side", "polygon": right, "weight": 0.8},
                    ],
                    "ignore_zones": [],
                    "params": params_from_old(item["filename"]),
                }
            )
    return {
        "version": "2.1",
        "camera_id": item["camera_id"],
        "frame_size": [width, height],
        "description": f"Recalibrated from {item['view']} middle frame on 2026-07-06. Empty screens means this view has no fixed monitor target.",
        "screens": screens,
    }


def draw_preview(item: dict, config: dict) -> None:
    src = WORK_DIR / item["image"]
    out = WORK_DIR / f"{item['view']}_preview.jpg"
    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for screen in config.get("screens", []):
        screen_poly = [tuple(p) for p in screen["screen_poly"]]
        near_poly = [tuple(p) for p in screen["near_zone"]]
        draw.polygon(near_poly, outline=(255, 180, 0, 255), fill=(255, 180, 0, 35))
        for zone in screen.get("danger_zones", []):
            draw.polygon([tuple(p) for p in zone["polygon"]], outline=(255, 0, 255, 230), fill=(255, 0, 255, 25))
        draw.polygon(screen_poly, outline=(255, 40, 40, 255), fill=(255, 40, 40, 45))
    image.save(out, quality=92)


def write_config_files(view: str, apply_official: bool) -> dict:
    manifest = extract_frames()
    by_view = {item["view"]: item for item in manifest}
    draft = load_draft(manifest)
    item = by_view[view]
    record = draft["views"][view]
    config = build_config(item, record)
    generated = WORK_DIR / item["filename"]
    generated.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_preview(item, config)
    if apply_official:
        official = CONFIG_DIR / item["filename"]
        if official.exists() and not (CONFIG_DIR / f"{item['filename']}.bak_20260706").exists():
            shutil.copy2(official, CONFIG_DIR / f"{item['filename']}.bak_20260706")
        shutil.copy2(generated, official)
    return {"generated": str(generated), "official": str(CONFIG_DIR / item["filename"]) if apply_official else ""}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JianKong Screen Recalibration</title>
<style>
body{margin:0;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#101418;color:#e8edf2}
.app{display:grid;grid-template-columns:220px 1fr;min-height:100vh}.side{background:#18202a;padding:14px;border-right:1px solid #2c3947}.view{display:block;width:100%;margin:0 0 8px;padding:10px;border:1px solid #344457;background:#202b36;color:#e8edf2;text-align:left;border-radius:6px;cursor:pointer}.view.active{background:#31506d;border-color:#6aa6d9}.view.done{border-color:#4eb66a}.main{padding:14px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.btn{padding:9px 12px;border:1px solid #50687e;background:#253241;color:#fff;border-radius:6px;cursor:pointer}.btn.primary{background:#1d6f42;border-color:#31a764}.btn.warn{background:#6f491d;border-color:#c48639}.btn.danger{background:#712a2a;border-color:#c85858}.status{color:#b8c7d6}.canvasWrap{position:relative;display:inline-block;max-width:100%;background:#050709}.canvasWrap img{display:block;max-width:calc(100vw - 270px);max-height:calc(100vh - 115px);width:auto;height:auto}.canvasWrap canvas{position:absolute;left:0;top:0}.small{font-size:13px;color:#9caec0}.pill{padding:2px 6px;border:1px solid #60758a;border-radius:999px;color:#cfe1ef}
</style>
</head>
<body><div class="app"><aside class="side"><h3>7 个视角</h3><div id="views"></div><p class="small">点当前屏幕四角：左上、右上、右下、左下。一个视角有多个屏幕时，点“新增屏幕”。没有屏幕就点“无屏幕”。</p></aside><main class="main"><div class="toolbar"><strong id="title"></strong><span id="meta" class="pill"></span><span id="screenLabel" class="pill"></span><button class="btn" id="prevScreen">上一屏</button><button class="btn" id="nextScreen">下一屏</button><button class="btn primary" id="addScreen">新增屏幕</button><button class="btn" id="undo">撤销一点</button><button class="btn danger" id="clear">清空当前屏</button><button class="btn warn" id="noscreen">无屏幕</button><button class="btn primary" id="save">保存当前视角并应用</button><button class="btn primary" id="saveAll">全部应用到正式 JSON</button><span id="status" class="status"></span></div><div class="canvasWrap"><img id="img"><canvas id="cv"></canvas></div></main></div>
<script>
let state=null, cur=0, localPoints=[];
const $=id=>document.getElementById(id);
async function api(path, body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});return await r.json();}
function view(){return state.views[cur]}
function normalizeRec(r){
  if(!r.screens){r.screens = r.points && r.points.length ? [{screen_id:'screen_01', points:r.points}] : []; delete r.points;}
  if(typeof r.active_screen!=='number') r.active_screen=0;
  if(typeof r.no_screen!=='boolean') r.no_screen=!r.screens.length;
  return r;
}
function rec(v){return normalizeRec(state.draft.views[v.view] || (state.draft.views[v.view]={screens:[],active_screen:0,no_screen:false,saved:false}))}
function activeIndex(r){if(!r.screens.length)return 0; r.active_screen=Math.max(0,Math.min(r.active_screen,r.screens.length-1)); return r.active_screen}
function activeScreen(r){return r.screens[activeIndex(r)] || null}
function ensureScreen(r){r.no_screen=false; if(!r.screens.length)r.screens=[{screen_id:'screen_01',points:[]}]; return activeScreen(r)}
function renumberScreens(r){r.screens.forEach((s,i)=>s.screen_id=`screen_${String(i+1).padStart(2,'0')}`)}
function draw(){
  const img=$('img'), c=$('cv'); c.width=img.clientWidth; c.height=img.clientHeight; const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
  const sx=c.width/view().width, sy=c.height/view().height; const r=rec(view()); const a=activeIndex(r);
  r.screens.forEach((s,si)=>{
    const pts = (si===a && localPoints.length) ? localPoints : (s.points||[]);
    ctx.lineWidth=si===a?4:2; ctx.strokeStyle=si===a?'#ff3333':'#34d399'; ctx.fillStyle=si===a?'rgba(255,40,40,.18)':'rgba(52,211,153,.12)';
    if(pts.length){ctx.beginPath(); pts.forEach((p,i)=>{const x=p[0]*sx,y=p[1]*sy; if(i)ctx.lineTo(x,y); else ctx.moveTo(x,y)}); if(pts.length===4)ctx.closePath(); ctx.stroke(); if(pts.length===4)ctx.fill();}
    ctx.fillStyle='#fff'; ctx.strokeStyle='#111'; ctx.font='16px sans-serif'; pts.forEach((p,i)=>{const x=p[0]*sx,y=p[1]*sy; ctx.beginPath(); ctx.arc(x,y,6,0,Math.PI*2); ctx.fill(); ctx.stroke(); ctx.fillText(`${s.screen_id}:${i+1}`,x+9,y-9);});
  });
  const s=activeScreen(r); const count=s?(s.points||[]).length:0;
  $('screenLabel').textContent = r.no_screen ? '无屏幕' : (s ? `${s.screen_id} (${a+1}/${r.screens.length})` : '未建屏幕');
  $('status').textContent = r.no_screen ? '当前标记：无屏幕' : `当前屏幕已选 ${count}/4 点；绿色是其它已标屏幕`;
}
function render(){
  $('views').innerHTML=''; state.views.forEach((v,i)=>{const r=rec(v); const b=document.createElement('button'); b.className='view'+(i===cur?' active':'')+(r.saved?' done':''); b.textContent=`${i+1}. ${v.label} ${r.no_screen?'(无屏幕)':`(${r.screens.length}屏)`}`; b.onclick=()=>{cur=i; localPoints=[]; render()}; $('views').appendChild(b);});
  const v=view(); $('title').textContent=v.label+' / '+v.view; $('meta').textContent=`${v.width}x${v.height}`; $('img').src='/image/'+v.image+'?t='+Date.now(); $('img').onload=draw;
}
$('cv').onclick=e=>{const v=view(), img=$('img'), rect=img.getBoundingClientRect(); const x=(e.clientX-rect.left)*v.width/img.clientWidth, y=(e.clientY-rect.top)*v.height/img.clientHeight; const r=rec(v); const s=ensureScreen(r); if(!localPoints.length)localPoints=[...(s.points||[])]; if(localPoints.length<4)localPoints.push([Math.round(x*100)/100,Math.round(y*100)/100]); s.points=localPoints; draw();};
$('prevScreen').onclick=()=>{const r=rec(view()); if(r.screens.length){r.active_screen=(activeIndex(r)+r.screens.length-1)%r.screens.length; localPoints=[]; draw();}};
$('nextScreen').onclick=()=>{const r=rec(view()); if(r.screens.length){r.active_screen=(activeIndex(r)+1)%r.screens.length; localPoints=[]; draw();}};
$('addScreen').onclick=()=>{const r=rec(view()); r.no_screen=false; r.screens.push({screen_id:`screen_${String(r.screens.length+1).padStart(2,'0')}`,points:[]}); r.active_screen=r.screens.length-1; localPoints=[]; draw(); render();};
$('undo').onclick=()=>{const r=rec(view()); const s=ensureScreen(r); if(!localPoints.length)localPoints=[...(s.points||[])]; localPoints.pop(); s.points=localPoints; draw();};
$('clear').onclick=()=>{const r=rec(view()); const s=ensureScreen(r); localPoints=[]; s.points=[]; draw();};
$('noscreen').onclick=()=>{const r=rec(view()); localPoints=[]; r.screens=[]; r.active_screen=0; r.no_screen=true; draw(); render();};
$('save').onclick=async()=>{const v=view(), r=rec(v); renumberScreens(r); if(!r.no_screen && (!r.screens.length || r.screens.some(s=>(s.points||[]).length!==4))){alert('每个屏幕都需要 4 个点，或者选择无屏幕'); return;} const out=await api('/api/save',{view:v.view,screens:r.screens,no_screen:!!r.no_screen,active_screen:r.active_screen,apply_official:true}); r.saved=true; localPoints=[]; $('status').textContent='已保存并应用: '+out.generated; render();};
$('saveAll').onclick=async()=>{const out=await api('/api/save_all',{apply_official:true}); $('status').textContent='已全部应用'; console.log(out);};
(async()=>{state=await api('/api/state'); render();})();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: dict | list) -> None:
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        manifest = extract_frames()
        draft = load_draft(manifest)
        if self.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self.send_json({"views": manifest, "draft": draft})
        elif self.path.startswith("/image/"):
            name = unquote(self.path.split("/image/", 1)[1].split("?", 1)[0])
            path = WORK_DIR / name
            if not path.exists() or path.parent != WORK_DIR:
                self.send_error(404)
            else:
                self.send_bytes(path.read_bytes(), "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        manifest = extract_frames()
        draft = load_draft(manifest)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/save":
            view = body["view"]
            draft["views"][view] = {
                "screens": body.get("screens", []),
                "active_screen": int(body.get("active_screen", 0)),
                "no_screen": bool(body.get("no_screen")),
                "saved": True,
            }
            normalize_record(draft["views"][view])
            save_draft(draft)
            self.send_json(write_config_files(view, bool(body.get("apply_official"))))
        elif self.path == "/api/save_all":
            out = {}
            for item in manifest:
                view = item["view"]
                if view in draft["views"]:
                    out[view] = write_config_files(view, bool(body.get("apply_official")))
            self.send_json(out)
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    extract_frames()
    host = "0.0.0.0"
    port = 8765
    print(f"Open http://127.0.0.1:{port} on Ubuntu, or http://192.168.50.2:{port} from Windows.")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
