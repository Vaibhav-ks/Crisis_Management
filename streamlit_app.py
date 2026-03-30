"""
streamlit_app.py  —  AEGIS Crisis Management AI  (Flood Edition)
=================================================================
Run:  streamlit run streamlit_app.py

Architecture:
  AUTOMATED WATCHER (Stage 1)
    └─ Polls satellite_images/ every 3 s
    └─ New image found → calls classify_flood() (vision agent)
         ├─ NO FLOOD  → move image to scanned_images/  → keep watching
         └─ FLOOD     → trigger full pipeline (Stages 2-10)
                         → after pipeline: move image to scanned_images/
                         → auto-restart watcher (20 s countdown)

  HUMAN-IN-THE-LOOP:  Stage 7 (Resource Approval) · Stage 9 (Route Approval)
  All other stages (2-6, 8, 10) auto-advance via countdown.
  Stepper: click any completed stage to jump back; forward is auto-only.
"""

import streamlit as st

st.set_page_config(
    page_title="AEGIS — Crisis Management AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import io, os, sys, json, shutil, sqlite3, contextlib, traceback, time, random
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import folium
from PIL import Image, ImageDraw, ImageFont
from streamlit_folium import st_folium

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Folder paths ──────────────────────────────────────────────────────────────
SATELLITE_DIR = Path(_ROOT) / "satellite_images"
SCANNED_DIR   = Path(_ROOT) / "scanned_images"
SATELLITE_DIR.mkdir(exist_ok=True)
SCANNED_DIR.mkdir(exist_ok=True)

DB_PATH         = os.path.join(_ROOT, "crisis.db")
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
PIXEL_THRESHOLD = 0.45
FLOOD_FRACTION  = 0.10
WATCH_INTERVAL  = 3.0   # seconds between watcher polls

# ============================================================================
#  THEME
# ============================================================================
THEME = {
    "bg": "#080d14", "bg2": "#0d1520",
    "cyan": "#00d4ff", "red": "#ff2d55",
    "orange": "#ff9500", "green": "#30d158",
    "yellow": "#ffd60a", "text": "#e5e5e7",
    "mono": "#00ff88",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&family=Exo+2:wght@400;600&display=swap');
  body,.stApp{{background:{THEME['bg']};color:{THEME['text']};}}
  h1,h2,h3{{font-family:'Rajdhani',sans-serif;}}
  .stButton>button{{background:{THEME['bg2']};color:{THEME['cyan']};border:2px solid {THEME['cyan']};
    border-radius:6px;padding:8px 18px;font-family:'Share Tech Mono',monospace;transition:.25s;}}
  .stButton>button:hover{{background:{THEME['cyan']};color:{THEME['bg']};}}
  .stMetric{{background:{THEME['bg2']};padding:16px;border-radius:8px;border-left:4px solid {THEME['cyan']};}}
  .terminal-log{{background:#000;color:{THEME['mono']};font-family:'Share Tech Mono',monospace;
    font-size:11px;line-height:1.6;padding:12px;border-radius:4px;border:1px solid {THEME['green']};
    max-height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;}}
  .card{{background:{THEME['bg2']};border-left:4px solid {THEME['cyan']};border-radius:6px;padding:12px;margin:6px 0;}}
  .card-warn{{background:#1a0f00;border-left:4px solid {THEME['orange']};border-radius:6px;padding:12px;margin:6px 0;}}
  .card-flood{{background:#1a0005;border-left:4px solid {THEME['red']};border-radius:6px;padding:12px;margin:6px 0;}}
  .card-ok{{background:#001a08;border-left:4px solid {THEME['green']};border-radius:6px;padding:12px;margin:6px 0;}}
  .card-eq{{background:#1a0a00;border-left:4px solid {THEME['orange']};border-radius:6px;padding:12px;margin:6px 0;}}
  .scan-row{{font-family:'Share Tech Mono',monospace;font-size:12px;padding:4px 0;}}
  /* Stepper button overrides */
  .step-done .stButton>button{{background:#001a08 !important;color:{THEME['green']} !important;
    border:1px solid {THEME['green']} !important;font-size:8px !important;
    padding:4px 2px !important;border-radius:4px !important;width:100%;}}
  .step-done .stButton>button:hover{{background:{THEME['green']} !important;color:#000 !important;}}
  .step-skip .stButton>button{{background:{THEME['bg2']} !important;color:{THEME['yellow']} !important;
    border:1px solid {THEME['yellow']} !important;font-size:10px !important;padding:4px 10px !important;}}
  .step-skip .stButton>button:hover{{background:{THEME['yellow']} !important;color:#000 !important;}}
</style>
""", unsafe_allow_html=True)

# ============================================================================
#  CONSTANTS
# ============================================================================
_DEFAULT_META = {"center_lat": 19.062061, "center_lon": 72.863542, "coverage_km": 1.6, "width_px": 1024, "height_px": 522}
ROUTE_COLORS  = {"ambulance": "#e74c3c", "rescue_team": "#2980b9", "boat": "#16a085", "helicopter": "#8e44ad", "fire_truck": "#e67e22", "truck": "#7f8c8d"}
RESOURCE_EMOJI= {"ambulance": "🚑", "rescue_team": "🚒", "boat": "🚤", "helicopter": "🚁", "fire_truck": "🚒", "truck": "🚛"}
BASE_ICON     = {"ambulance": ("red", "plus-sign"), "rescue_team": ("blue", "home"), "boat": ("darkblue", "tint"), "helicopter": ("purple", "plane")}
_DEFAULT_BASES= {
    "ambulance":   {"name": "Hospital",      "lat": 19.06546856543151,  "lon": 72.86100899070198},
    "rescue_team": {"name": "Rescue Center", "lat": 19.06847079812735,  "lon": 72.85793995490616},
    "boat":        {"name": "Boat Depot",    "lat": 19.063380373548366, "lon": 72.85538649195271},
}
STAGES = ["1️⃣ Scan", "2️⃣ Zone Map", "3️⃣ Drones", "4️⃣ Imagery", "5️⃣ Analysis",
          "6️⃣ Resources", "7️⃣ ✅Approve I", "8️⃣ Routes", "9️⃣ ✅Approve II", "🔟 Comms"]
_PHASE_INFO = {
    "idle":              ("#555",         "⚫ Idle"),
    "watching":          (THEME["cyan"],  "🔵 Watching satellite_images/…"),
    "classifying":       (THEME["yellow"],"🟡 Classifying image…"),
    "running_phase1":    (THEME["yellow"],"🟡 Phase 1 Running"),
    "awaiting_resource": (THEME["orange"],"🟠 Awaiting Resource Approval"),
    "running_phase2":    (THEME["yellow"],"🟡 Phase 2 Running"),
    "awaiting_route":    (THEME["orange"],"🟠 Awaiting Route Approval"),
    "running_phase3":    (THEME["yellow"],"🟡 Phase 3 Running"),
    "complete":          (THEME["green"], "🟢 Pipeline Complete"),
}

# ============================================================================
#  MODEL LOADERS
# ============================================================================
@st.cache_resource
def _load_vision():
    from agents.vision_agent.preprocess         import load_image
    from agents.vision_agent.flood_segmentation import detect_flood
    return load_image, detect_flood

@st.cache_resource
def _load_flood_classifier():
    from agents.vision_agent.preprocess       import load_image
    from agents.vision_agent.flood_classifier import classify_flood
    return load_image, classify_flood

def _quick_classify(img_path: Path) -> dict:
    """Fast flood/no-flood classification — pixel-level UNet thresholding."""
    load_image, classify_flood = _load_flood_classifier()
    image = load_image(str(img_path))
    return classify_flood(image, PIXEL_THRESHOLD, FLOOD_FRACTION)

def _collect_images(folder: Path) -> list:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

# ============================================================================
#  HUD IMAGE OVERLAYS
# ============================================================================
def _font(size=10):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/freefont/FreeMono.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _sat_overlay(pil: Image.Image, is_flood: bool = False,
                 img_name: str = "", prob_map=None) -> Image.Image:
    """Add satellite HUD to image: top/bottom bars, grid, crosshair, brackets, flood tint."""
    img = pil.convert("RGBA").copy()
    W, H = img.size
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d  = ImageDraw.Draw(ov)

    # Flood tint
    if is_flood:
        ov.paste(Image.new("RGBA", (W, H), (0, 70, 200, 38)), (0, 0),
                 Image.new("RGBA", (W, H), (0, 70, 200, 38)))

    # Flood pixel heat-map
    if prob_map is not None and is_flood:
        try:
            pm = np.clip(prob_map, 0, 1)
            pm_r = np.array(Image.fromarray((pm * 255).astype(np.uint8)).resize((W, H)))
            r = np.zeros((H, W, 4), dtype=np.uint8)
            r[pm_r > int(PIXEL_THRESHOLD * 255)] = [0, 100, 255, 90]
            ov.paste(Image.fromarray(r, "RGBA"), (0, 0), Image.fromarray(r, "RGBA"))
        except Exception:
            pass

    # Scan lines
    for y in range(0, H, 6):
        d.line([(0, y), (W, y)], fill=(0, 0, 0, 16))

    # Grid 10x10
    gc = (0, 212, 255, 42)
    for i in range(1, 10):
        d.line([(W * i // 10, 0), (W * i // 10, H)], fill=gc)
        d.line([(0, H * i // 10), (W, H * i // 10)], fill=gc)

    # Crosshair
    cx, cy, g = W // 2, H // 2, 18
    cc = (0, 255, 136, 200)
    d.line([(cx-55, cy), (cx-g, cy)], fill=cc, width=2)
    d.line([(cx+g, cy), (cx+55, cy)], fill=cc, width=2)
    d.line([(cx, cy-55), (cx, cy-g)], fill=cc, width=2)
    d.line([(cx, cy+g), (cx, cy+55)], fill=cc, width=2)
    d.ellipse([(cx-4, cy-4), (cx+4, cy+4)], outline=cc, width=1)

    # Corner brackets
    bc, brk, pad = (255, 45, 85, 220), 26, 7
    for x0, y0, dx, dy in [(pad,pad,1,1),(W-pad,pad,-1,1),(pad,H-pad,1,-1),(W-pad,H-pad,-1,-1)]:
        d.line([(x0, y0), (x0+dx*brk, y0)], fill=bc, width=3)
        d.line([(x0, y0), (x0, y0+dy*brk)], fill=bc, width=3)

    # Top bar
    BAR = 30
    d.rectangle([(0, 0), (W, BAR)], fill=(0, 0, 0, 210))
    d.line([(0, BAR), (W, BAR)], fill=(0, 212, 255, 170), width=1)

    sat_id  = random.choice(["SENTINEL-2A","SENTINEL-2B","LANDSAT-9","GeoEye-1","IKONOS-2"])
    orbit   = random.randint(14000, 16999)
    mission = f"AEGIS-{random.randint(100,999)}"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    f10, f12 = _font(10), _font(12)

    d.text((8,  9), f"🛰  {sat_id}  |  ORBIT {orbit:05d}", font=f12, fill=(0, 212, 255, 255))
    d.text((W//2-80, 9), now_utc,                            font=f10, fill=(200, 200, 200, 240))
    d.text((W-130, 9), f"MISSION {mission}",                font=f10, fill=(0, 255, 136, 240))

    # Bottom bar
    d.rectangle([(0, H-26), (W, H)], fill=(0, 0, 0, 210))
    d.line([(0, H-26), (W, H-26)], fill=(0, 212, 255, 170), width=1)
    meta = st.session_state.get("image_meta", _DEFAULT_META)
    lat, lon, cov = meta.get("center_lat",19.062), meta.get("center_lon",72.863), meta.get("coverage_km",1.6)
    res = f"{cov*1000/1024:.2f}m/px"
    d.text((8, H-18), f"LAT {lat:.5f}°  LON {lon:.5f}°  |  COV {cov:.1f}km  |  {res}", font=f10, fill=(170,170,170,240))

    if is_flood:
        d.text((W-170, H-18), "⚠ FLOOD ZONE", font=f10, fill=(255,45,85,255))

    img = Image.alpha_composite(img, ov)
    return img.convert("RGB")


def _drone_overlay(pil: Image.Image, zone_id: str, drone_id: str,
                   people_count: int = 0, is_flood: bool = False) -> Image.Image:
    """Drone feed HUD: top/bottom bars, corner brackets, scan-lines, flood label."""
    img = pil.convert("RGBA").copy()
    W, H = img.size
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d  = ImageDraw.Draw(ov)

    # Scan lines
    for y in range(0, H, 4):
        d.line([(0, y), (W, y)], fill=(0, 0, 0, 20))

    # Edge vignette (night-vision)
    for e in range(min(18, W//8)):
        a = int(55 * (1 - e/(W//8)))
        d.line([(e,0),(e,H)], fill=(0,40,0,a))
        d.line([(W-e-1,0),(W-e-1,H)], fill=(0,40,0,a))

    # Top bar
    d.rectangle([(0,0),(W,30)], fill=(0,0,0,200))
    d.line([(0,30),(W,30)], fill=(255,45,85,150), width=1)
    alt = random.randint(45, 120)
    sig = "▌▌▌▌" if random.random()>0.2 else "▌▌▌░"
    now = datetime.now().strftime("%H:%M:%S")
    f10, f12 = _font(10), _font(12)
    d.text((8,9),  f"DRONE/{drone_id.upper()}  →  ZONE {zone_id}", font=f12, fill=(255,45,85,255))
    d.text((W-115,9), f"SIG {sig}  {now}", font=f10, fill=(0,212,255,220))

    # Bottom bar
    d.rectangle([(0,H-28),(W,H)], fill=(0,0,0,200))
    d.line([(0,H-28),(W,H-28)], fill=(255,45,85,150), width=1)
    meta = st.session_state.get("image_meta", _DEFAULT_META)
    lat_v = meta.get("center_lat",19.06) + random.uniform(-0.015,0.015)
    lon_v = meta.get("center_lon",72.86) + random.uniform(-0.015,0.015)
    pstr  = f"👤 {people_count} VICTIMS" if people_count else "SCANNING..."
    pc    = (255,50,50,255) if people_count else (180,180,180,200)
    d.text((8,H-19), f"ALT {alt}m  |  {lat_v:.5f}°N  {lon_v:.5f}°E", font=f10, fill=(170,170,170,240))
    d.text((W-145,H-19), pstr, font=f10, fill=pc)

    # Corner brackets
    bc, brk, pad = (255,45,85,220), 20, 6
    for x0,y0,dx,dy in [(pad,pad+30,1,1),(W-pad,pad+30,-1,1),(pad,H-pad-28,1,-1),(W-pad,H-pad-28,-1,-1)]:
        d.line([(x0,y0),(x0+dx*brk,y0)], fill=bc, width=2)
        d.line([(x0,y0),(x0,y0+dy*brk)], fill=bc, width=2)

    # Disaster label
    if is_flood:
        lbl = "FLOOD ZONE"
        lc  = (0,150,255,230)
        tw  = len(lbl)*7
        bx0,bx1 = W//2-tw//2-8, W//2+tw//2+8
        d.rectangle([(bx0,H//2-12),(bx1,H//2+12)], fill=(0,0,0,160), outline=lc, width=1)
        d.text((bx0+8,H//2-7), lbl, font=f10, fill=lc)

    img = Image.alpha_composite(img, ov)
    return img.convert("RGB")

# ============================================================================
#  LANGGRAPH PIPELINE
# ============================================================================
@st.cache_resource
def _get_graph():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, END
    from master_agent.master_state import MasterState
    from master_agent.master_nodes import (
        vision_node, store_zone_node, drone_analysis_node,
        drone_decision_node, drone_dispatch_node, drone_vision_node,
        update_people_node, rescue_decision_node,
        admin_resource_node, resource_approval_router,
        route_planner_node, admin_route_node, route_approval_router,
        communication_node,
    )
    b = StateGraph(MasterState)
    for name, fn in [
        ("vision", vision_node), ("store_zone", store_zone_node),
        ("drone_analysis", drone_analysis_node), ("drone_decision", drone_decision_node),
        ("drone_dispatch", drone_dispatch_node), ("drone_vision", drone_vision_node),
        ("update_people", update_people_node), ("rescue_decision", rescue_decision_node),
        ("admin_resource", admin_resource_node), ("route_planner", route_planner_node),
        ("admin_route", admin_route_node), ("communication", communication_node),
    ]:
        b.add_node(name, fn)
    b.set_entry_point("vision")
    for src, dst in [
        ("vision","store_zone"),("store_zone","drone_analysis"),("drone_analysis","drone_decision"),
        ("drone_decision","drone_dispatch"),("drone_dispatch","drone_vision"),
        ("drone_vision","update_people"),("update_people","rescue_decision"),
        ("rescue_decision","admin_resource"),("route_planner","admin_route"),("communication",END),
    ]:
        b.add_edge(src, dst)
    b.add_conditional_edges("admin_resource", resource_approval_router,
                            {"approved":"route_planner","rejected":"rescue_decision"})
    b.add_conditional_edges("admin_route", route_approval_router,
                            {"approved":"communication","rejected":"route_planner"})
    return b.compile(checkpointer=MemorySaver(), interrupt_before=["admin_resource","admin_route"])

def _cfg(): return {"configurable": {"thread_id": st.session_state.get("thread_id","aegis_main")}}
def _graph_state() -> dict:
    try:
        snap = _get_graph().get_state(_cfg())
        return dict(snap.values) if snap and snap.values else {}
    except Exception:
        return {}
def _next_nodes() -> list:
    try:
        snap = _get_graph().get_state(_cfg())
        return list(snap.next) if snap and snap.next else []
    except Exception:
        return []

def _invoke(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    if buf.getvalue().strip():
        _log(buf.getvalue().strip())
    return result

def _run_phase1(img_path: str, meta: dict):
    _log("LangGraph Phase 1 starting …")
    from dotenv import load_dotenv; load_dotenv()
    _invoke(_get_graph().invoke, {
        "satellite_image": img_path,
        "image_meta":      meta,
        "base_locations":  _DEFAULT_BASES,
        "field_reports":   [],
        "dispatch_config": {
            "send_sms":       bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("YOUR_PHONE_NUMBER")),
            "generate_audio": True,
            "language":       st.session_state.get("comm_language","English"),
            "to_number":      os.getenv("YOUR_PHONE_NUMBER"),
        },
    }, config=_cfg())
    st.session_state["pipeline_phase"] = "awaiting_resource"
    _log("Phase 1 complete — interrupted before admin_resource")

def _run_phase2(approved: bool):
    graph = _get_graph(); config = _cfg()
    graph.update_state(config, {"resource_approved": approved}, as_node="admin_resource")
    _invoke(graph.invoke, None, config=config)
    st.session_state["pipeline_phase"] = "awaiting_route" if approved else "awaiting_resource"

def _run_phase3(approved: bool):
    graph = _get_graph(); config = _cfg()
    graph.update_state(config, {"route_approved": approved}, as_node="admin_route")
    _invoke(graph.invoke, None, config=config)
    st.session_state["pipeline_phase"] = "complete" if approved else "awaiting_route"

# ============================================================================
#  HELPERS
# ============================================================================
def _ts(): return datetime.now().strftime("%H:%M:%S")
def _log(text):
    st.session_state.setdefault("log","")
    for line in (text or "").strip().splitlines():
        if line.strip():
            st.session_state["log"] += f"[{_ts()}] {line}\n"

def _terminal():
    log = st.session_state.get("log","(no output yet)")
    st.markdown(f'<div class="terminal-log" id="tlog">{log}</div>'
                '<script>var t=document.getElementById("tlog");if(t)t.scrollTop=t.scrollHeight;</script>',
                unsafe_allow_html=True)

def _sev_label(s):
    if s>=0.8: return "🔴 CRITICAL"
    if s>=0.6: return "🟠 HIGH"
    if s>=0.4: return "🟡 MODERATE"
    return "🟢 LOW"

def _rcolor(rt):
    k = rt.lower().rstrip("s")
    return ROUTE_COLORS.get(k, ROUTE_COLORS.get(rt.lower(),"#888"))

def _remoji(rt):
    k = rt.lower().rstrip("s")
    return RESOURCE_EMOJI.get(k, RESOURCE_EMOJI.get(rt.lower(),"🚗"))

def _phase_badge():
    phase = st.session_state.get("pipeline_phase","idle")
    color, label = _PHASE_INFO.get(phase, ("#555", phase))
    st.markdown(f'<span style="background:{color};color:#000;padding:4px 14px;border-radius:12px;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:11px;font-weight:bold;">'
                f'{label}</span><br><br>', unsafe_allow_html=True)

def _prob_bar(frac, width=16):
    filled=int(frac*width); ti=int(FLOOD_FRACTION*width); bar=""
    for i in range(width):
        bar += ("█" if i<ti else "▓") if i<filled else "░"
    return bar

def _move_to_scanned(img_path) -> Path:
    """Move a processed image into scanned_images/, avoiding name collisions."""
    src = Path(img_path)
    if not src.exists():
        return src  # already moved or never existed
    SCANNED_DIR.mkdir(exist_ok=True)
    dest = SCANNED_DIR / src.name
    if dest.exists():
        dest = SCANNED_DIR / f"{src.stem}_{int(time.time())}{src.suffix}"
    shutil.move(str(src), str(dest))
    return dest

def _auto_advance(next_stage: int, delay: float = 3.0):
    """
    Auto-advance to next_stage after `delay` seconds.
    SUPPRESSED when the user has navigated back to review a past stage.
    Never call on HITL stages (indices 6 and 8).
    """
    current = st.session_state.get("stage", 0)
    max_reached = st.session_state.get("max_stage", 0)

    # User navigated back to review — do NOT auto-advance
    if current < max_reached:
        return

    key = f"_adv_{current}"
    if not st.session_state.get(key):
        st.session_state[key] = time.time()
    elapsed   = time.time() - st.session_state[key]
    remaining = max(0.0, delay - elapsed)

    if remaining <= 0:
        st.session_state.pop(key, None)
        st.session_state["stage"]     = next_stage
        st.session_state["max_stage"] = max(max_reached, next_stage)
        st.rerun()
        return

    bar_col, btn_col = st.columns([5, 1])
    with bar_col:
        st.progress(min(elapsed/delay, 1.0),
                    text=f"⏩ Auto-advancing to Stage {next_stage+1} in {remaining:.0f}s …")
    with btn_col:
        st.markdown('<div class="step-skip">', unsafe_allow_html=True)
        if st.button("Skip →", key=f"_skip_{next_stage}"):
            st.session_state.pop(key, None)
            st.session_state["stage"]     = next_stage
            st.session_state["max_stage"] = max(max_reached, next_stage)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    time.sleep(min(remaining, 2.0))
    st.rerun()

# ============================================================================
#  SIDEBAR
# ============================================================================
def _sidebar():
    with st.sidebar:
        st.markdown("### 🛰️ AEGIS · LangGraph Pipeline")
        st.divider()
        phase = st.session_state.get("pipeline_phase","idle")
        color, label = _PHASE_INFO.get(phase,("#555",phase))
        st.markdown(f'<div class="card">🔗 <b>Master Graph</b><br>'
                    f'<span style="color:{color};font-size:12px;">{label}</span></div>',
                    unsafe_allow_html=True)
        nxt = _next_nodes()
        if nxt:
            st.markdown(f'<div class="card">⏸️ <b>Interrupted Before</b><br>'
                        f'<span style="color:{THEME["cyan"]};font-size:12px;">{", ".join(nxt)}</span></div>',
                        unsafe_allow_html=True)

        sat_imgs    = _collect_images(SATELLITE_DIR)
        scanned_cnt = len(list(SCANNED_DIR.glob("*"))) if SCANNED_DIR.exists() else 0
        st.markdown(
            f'<div class="card" style="font-size:11px;">'
            f'📂 <b>satellite_images/</b>  <span style="color:{THEME["cyan"]};">{len(sat_imgs)} waiting</span><br>'
            f'📁 <b>scanned_images/</b>  <span style="color:{THEME["green"]};">{scanned_cnt} processed</span>'
            f'</div>', unsafe_allow_html=True)

        flood_log = st.session_state.get("flood_log", [])
        if flood_log:
            st.divider()
            st.markdown("**🌊 Flood Events**")
            for entry in flood_log[-5:]:
                st.markdown(
                    f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                    f'color:{THEME["red"]};padding:2px 0;">🌊 {entry}</div>',
                    unsafe_allow_html=True)

        st.divider()
        st.markdown(f'<div class="card" style="font-family:\'Share Tech Mono\',monospace;font-size:10px;">'
                    f'<b style="color:{THEME["cyan"]};">Pipeline Nodes</b><br><br>'
                    f'vision → store_zone<br>→ drone_analysis<br>→ drone_decision<br>'
                    f'→ drone_dispatch → drone_vision<br>→ update_people<br>→ rescue_decision<br>'
                    f'→ <span style="color:{THEME["orange"]};">[⏸ admin_resource]</span><br>'
                    f'→ route_planner<br>'
                    f'→ <span style="color:{THEME["orange"]};">[⏸ admin_route]</span><br>'
                    f'→ communication → END</div>', unsafe_allow_html=True)

        st.divider()
        gs = _graph_state()
        for name,icon,key in [("Vision Agent","👁️","zone_map"),("Drone Agent","🚁","drone_allocation"),
                               ("Resource Agent","📦","rescue_plan"),("Route Agent","🗺️","route_plan"),
                               ("Comm Agent","📡","dispatch_result")]:
            val = gs.get(key)
            s = (f'<span style="color:{THEME["green"]};">🟢 Done</span>' if val
                 else f'<span style="color:#888;">⚪ Idle</span>')
            st.markdown(f'<div class="card">{icon} <b>{name}</b><br>{s}</div>', unsafe_allow_html=True)

        st.divider()
        st.metric("Stage", f"{st.session_state.get('stage',0)+1} / {len(STAGES)}")
        st.divider()
        if st.button("🔄 Full Reset", use_container_width=True):
            _do_full_reset()

# ============================================================================
#  STEPPER  — past stages are CLICKABLE (back-nav); future stages locked
# ============================================================================
def _stepper():
    s           = st.session_state.get("stage", 0)
    max_reached = st.session_state.get("max_stage", s)
    # Keep max_stage in sync when advancing forward naturally
    if s > max_reached:
        st.session_state["max_stage"] = s
        max_reached = s

    reviewing = s < max_reached   # user has navigated back

    # ── "Resume" banner when reviewing (suppress on HITL gates 7 & 9) ────
    if reviewing and s not in (6, 8):   # 6 = stage_7 index, 8 = stage_9 index
        rc1, rc2 = st.columns([4, 1])
        with rc1:
            st.markdown(
                f'<div style="background:#1a0f00;border-left:4px solid {THEME["yellow"]};'
                f'border-radius:4px;padding:6px 12px;font-family:\'Share Tech Mono\',monospace;'
                f'font-size:11px;color:{THEME["yellow"]};">'
                f'👁️ Reviewing Stage {s+1} — pipeline reached Stage {max_reached+1}</div>',
                unsafe_allow_html=True)
        with rc2:
            if st.button(f"▶ Resume (S{max_reached+1})", key="_resume_btn",
                         use_container_width=True):
                st.session_state["stage"] = max_reached
                st.rerun()

    # ── Stage pills ───────────────────────────────────────────────────────
    cols = st.columns(len(STAGES))
    for i, label in enumerate(STAGES):
        with cols[i]:
            if i == s:
                # Currently viewing — cyan highlight
                st.markdown(
                    f'<div style="background:{THEME["cyan"]};color:{THEME["bg"]};'
                    f'padding:6px 2px;text-align:center;border-radius:4px;'
                    f'font-size:9px;font-weight:bold;border:2px solid {THEME["cyan"]};">'
                    f'{label}</div>', unsafe_allow_html=True)

            elif i <= max_reached:
                # Already visited (before or after current) — green clickable
                st.markdown('<div class="step-done">', unsafe_allow_html=True)
                if st.button(f"✅ {label}", key=f"_step_{i}", use_container_width=True,
                             help=f"Jump to Stage {i+1}"):
                    st.session_state.pop(f"_adv_{i}", None)
                    st.session_state["stage"] = i
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                # Never visited — locked grey
                st.markdown(
                    f'<div style="background:{THEME["bg2"]};color:#444;'
                    f'padding:6px 2px;text-align:center;border-radius:4px;'
                    f'font-size:9px;font-weight:bold;border:1px solid #333;">'
                    f'{label}</div>', unsafe_allow_html=True)

# ============================================================================
#  FOLIUM MAP
# ============================================================================
def _folium_map():
    gs=_graph_state(); routes=gs.get("route_plan",[]); meta=gs.get("image_meta") or _DEFAULT_META
    fmap=folium.Map(location=[meta.get("center_lat",19.06),meta.get("center_lon",72.86)],zoom_start=15,tiles="CartoDB positron")
    seen_b,seen_z=set(),set()
    for r in routes:
        rk=r.get("resource_type","").lower().rstrip("s"); base=_DEFAULT_BASES.get(rk)
        if base and base["name"] not in seen_b:
            ic_c,ic_i=BASE_ICON.get(rk,("gray","info-sign"))
            folium.Marker([base["lat"],base["lon"]],tooltip=f"📍 {base['name']}",icon=folium.Icon(color=ic_c,icon=ic_i)).add_to(fmap)
            seen_b.add(base["name"])
        dest=r.get("destination_latlon")
        if dest and r.get("zone") not in seen_z:
            folium.Marker(list(dest),tooltip=f"🚨 Zone {r['zone']}",icon=folium.Icon(color="orange",icon="exclamation-sign")).add_to(fmap)
            seen_z.add(r["zone"])
    for r in routes:
        if not r.get("success"): continue
        color=_rcolor(r["resource_type"]); emoji=_remoji(r["resource_type"])
        wpts=r.get("waypoints",[])
        if len(wpts)<2:
            dest=r.get("destination_latlon"); rk=r["resource_type"].lower().rstrip("s"); base=_DEFAULT_BASES.get(rk)
            if base and dest: wpts=[(base["lat"],base["lon"]),dest]
            else: continue
        folium.PolyLine(wpts,color=color,weight=5,opacity=0.9,
                        tooltip=f"{emoji} {r.get('unit_count',1)}× {r['resource_type']}\nZone {r['zone']} · {r.get('distance_km',0)}km ETA {r.get('eta_minutes',0)}min").add_to(fmap)
    return fmap

def _do_full_reset():
    """Full session reset — preserves flood log and location prefs only."""
    import uuid
    flood_log = st.session_state.get("flood_log", [])
    meta_lat  = st.session_state.get("meta_lat", 19.062061)
    meta_lon  = st.session_state.get("meta_lon", 72.863542)
    meta_cov  = st.session_state.get("meta_cov", 1.6)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state["thread_id"]      = f"aegis_{uuid.uuid4().hex[:8]}"
    st.session_state["pipeline_phase"] = "watching"
    st.session_state["stage"]          = 0
    st.session_state["max_stage"]      = 0
    st.session_state["log"]            = ""
    st.session_state["flood_log"]      = flood_log
    st.session_state["meta_lat"]       = meta_lat
    st.session_state["meta_lon"]       = meta_lon
    st.session_state["meta_cov"]       = meta_cov
    st.balloons()
    st.rerun()

# ============================================================================
#  STAGE 1 — AUTOMATED SATELLITE IMAGE WATCHER
# ============================================================================
def stage_1():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🛰️ Stage 1: Satellite Image Watcher</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    max_reached = st.session_state.get("max_stage", 0)
    phase       = st.session_state.get("pipeline_phase", "watching")

    # ── User navigated back to review Stage 1 — show cached result only, no polling ──
    if max_reached > 0:
        flood_path = st.session_state.get("flood_image_path")
        flood_pm   = st.session_state.get("flood_prob_map")
        st.markdown(
            f'<div class="card-ok">\u2705 <b>Flood image classified and pipeline triggered.</b><br>'
            f'Image: <code>{Path(flood_path).name if flood_path else "\u2014"}</code></div>',
            unsafe_allow_html=True)
        if flood_path:
            try:
                p = Path(flood_path)
                pil = Image.open(p) if p.exists() else None
                if pil:
                    st.image(_sat_overlay(pil, is_flood=True, img_name=p.name, prob_map=flood_pm),
                             caption=f"Flood image: {p.name}", use_container_width=True)
            except Exception:
                pass
        _terminal()
        return

    # ── If pipeline already triggered, jump to Stage 2 ────────────────────
    if phase == "running_phase1":
        st.session_state["stage"]     = 1
        st.session_state["max_stage"] = max(max_reached, 1)
        st.rerun()

    # ── Auto-fire pipeline after flood detected ────────────────────────────
    flood_path = st.session_state.get("flood_image_path")
    if flood_path and not st.session_state.get("phase1_fired"):
        st.session_state["phase1_fired"]   = True
        st.session_state["pipeline_phase"] = "running_phase1"
        st.session_state["stage"]          = 1
        st.session_state["max_stage"]      = max(max_reached, 1)
        _log(f"FLOOD CONFIRMED \u2014 firing pipeline on {Path(flood_path).name}")
        st.rerun()

    # ── Watcher info card ─────────────────────────────────────────────────
    st.markdown(
        f'<div class="card" style="font-size:13px;">'
        f'<span style="color:{THEME["cyan"]};">\u25cf</span> '
        f'<b>Watching:</b> <code>{SATELLITE_DIR}</code><br>'
        f'Drop satellite images into this folder \u2014 flood classification runs automatically.<br>'
        f'<span style="font-size:11px;color:#888;">'
        f'No-flood images \u2192 moved to <code>scanned_images/</code> immediately.<br>'
        f'Flood images \u2192 pipeline fires \u2192 image moved after completion.</span>'
        f'</div>', unsafe_allow_html=True)

    with st.expander("📍 Location Metadata", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            lat = st.number_input("Centre Latitude",  value=float(st.session_state.get("meta_lat", 19.062061)), format="%.6f", key="mlat")
        with c2:
            lon = st.number_input("Centre Longitude", value=float(st.session_state.get("meta_lon", 72.863542)), format="%.6f", key="mlon")
        with c3:
            cov = st.number_input("Coverage (km)",    value=float(st.session_state.get("meta_cov", 1.6)), min_value=0.1, key="mcov")
        st.session_state["meta_lat"] = lat
        st.session_state["meta_lon"] = lon
        st.session_state["meta_cov"] = cov
        st.markdown(
            f'<div class="card" style="font-size:11px;"><b>Flood thresholds:</b><br>'
            f'Per-pixel UNet cutoff : \u2265 {PIXEL_THRESHOLD}<br>'
            f'Image flood trigger   : \u2265 {FLOOD_FRACTION*100:.0f}% pixels flooded</div>',
            unsafe_allow_html=True)

    st.divider()
    images    = _collect_images(SATELLITE_DIR)
    poll_slot = st.empty(); preview_slot = st.empty(); result_slot = st.empty()

    if not images:
        poll_slot.info(f"📭 No images in `satellite_images/` yet. Checking again in {WATCH_INTERVAL:.0f}s\u2026")
        st.session_state["pipeline_phase"] = "watching"
        time.sleep(WATCH_INTERVAL)
        st.rerun()
        return

    img_path = images[0]
    poll_slot.markdown(
        f'<div class="card">🔬 <b>Image detected:</b> <code>{img_path.name}</code> \u2014 classifying\u2026</div>',
        unsafe_allow_html=True)
    st.session_state["pipeline_phase"] = "classifying"
    try:
        preview_slot.image(_sat_overlay(Image.open(img_path), img_name=img_path.name),
                           caption=f"Classifying: {img_path.name}", use_container_width=True)
    except Exception:
        pass

    _log(f"Classifying {img_path.name} \u2026")
    t0 = time.time()
    try:
        result = _quick_classify(img_path)
    except Exception as exc:
        _log(f"[ERROR] classify_flood failed on {img_path.name}: {exc}")
        result_slot.error(f"Classification error: {exc}")
        dest = _move_to_scanned(img_path)
        _log(f"Moved (error) to scanned_images/{dest.name}")
        time.sleep(2); st.rerun(); return

    elapsed  = time.time() - t0
    is_flood = result["is_flood"]
    frac     = result["flooded_fraction"]

    try:
        preview_slot.image(
            _sat_overlay(Image.open(img_path), is_flood=is_flood, img_name=img_path.name,
                         prob_map=result.get("flood_prob_map")),
            caption=img_path.name, use_container_width=True)
    except Exception:
        pass

    if is_flood:
        result_slot.markdown(
            f'<div class="card-flood">'
            f'🌊 <b>FLOOD DETECTED</b> \u2014 <code>{img_path.name}</code><br>'
            f'Flooded pixels: <b>{frac*100:.1f}%</b>  |  Max prob: <b>{result["max_prob"]:.3f}</b>  |  Time: {elapsed:.1f}s<br>'
            f'<b>\u2192 Sending to main pipeline\u2026</b></div>', unsafe_allow_html=True)
        _log(f"FLOOD: {img_path.name} | {frac*100:.1f}% flooded | max={result['max_prob']:.3f}")
        st.session_state.setdefault("flood_log", []).append(f"{img_path.name} ({frac*100:.1f}%)")
        meta = {"center_lat": lat, "center_lon": lon, "coverage_km": cov, "width_px": 1024, "height_px": 1024}
        try:
            pil_tmp = Image.open(img_path); meta["width_px"] = pil_tmp.width; meta["height_px"] = pil_tmp.height
        except Exception:
            pass
        st.session_state.update({
            "flood_image_path": str(img_path), "flood_prob_map": result.get("flood_prob_map"),
            "image_meta": meta, "pipeline_phase": "running_phase1", "phase1_fired": False,
        })
        time.sleep(1.5); st.rerun()
    else:
        result_slot.markdown(
            f'<div class="card-ok">'
            f'\u2705 <b>No Flood</b> \u2014 <code>{img_path.name}</code><br>'
            f'Flooded pixels: {frac*100:.1f}%  |  Max prob: {result["max_prob"]:.3f}  |  Time: {elapsed:.1f}s<br>'
            f'\u2192 Image moved to <code>scanned_images/</code> \u2014 continuing watch\u2026</div>',
            unsafe_allow_html=True)
        _log(f"NO FLOOD: {img_path.name} | {frac*100:.1f}%")
        dest = _move_to_scanned(img_path)
        _log(f"Moved to scanned_images/{dest.name}")
        time.sleep(2); st.rerun()

def stage_2():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🗺️ Stage 2: Zone Map Analysis</h2>', unsafe_allow_html=True)
    _phase_badge()

    max_reached = st.session_state.get("max_stage", 0)
    phase       = st.session_state.get("pipeline_phase", "watching")

    # Only run Phase 1 when first arriving here fresh — not when reviewing
    if phase == "running_phase1" and max_reached <= 1:
        img_path = st.session_state.get("flood_image_path") or "Images_for_testing/image.png"
        meta     = st.session_state.get("image_meta", _DEFAULT_META.copy())
        st.info(f"🚀 Running LangGraph Phase 1 on `{Path(img_path).name}` …")
        with st.spinner("vision → drones → LLM rescue plan  (~60-120 s)"):
            try:
                _run_phase1(img_path, meta)
                st.success("✅ Phase 1 complete")
                st.session_state["max_stage"] = max(max_reached, 1)
            except Exception as e:
                _log(f"[ERROR] Phase 1:\n{traceback.format_exc()}")
                st.error(f"Phase 1 failed: {e}"); st.code(traceback.format_exc())
                if st.button("◀ Back", key="b2e"): st.session_state["stage"]=0; st.rerun()
                return
        st.rerun()

    gs = _graph_state(); zone_map = gs.get("zone_map",{})
    if not zone_map:
        st.warning("Zone map not in graph state yet.")
        _terminal()
        if st.button("◀ Back to Scan", key="b2"): st.session_state["stage"]=0; st.rerun()
        return

    c1,c2 = st.columns([3,2])
    with c1:
        grid = os.path.join(_ROOT,"zone_results","grid_output.jpg")
        if os.path.exists(grid):
            try:
                g = _sat_overlay(Image.open(grid), is_flood=True, img_name="grid_output.jpg")
                st.image(g, caption="Zone Severity Grid (10×10) — Satellite View", use_container_width=True)
            except Exception:
                st.image(grid, use_container_width=True)
        flood_path = st.session_state.get("flood_image_path")
        flood_pm   = st.session_state.get("flood_prob_map")
        if flood_path and os.path.exists(flood_path):
            try:
                pil = _sat_overlay(Image.open(flood_path), is_flood=True,
                                   img_name=Path(flood_path).name, prob_map=flood_pm)
                st.image(pil, caption=f"Flood trigger: {Path(flood_path).name}", use_container_width=True)
            except Exception:
                pass
    with c2:
        st.markdown("**Top Affected Zones**")
        top = sorted(zone_map.items(),key=lambda x:x[1].get("severity",0),reverse=True)[:15]
        st.dataframe(pd.DataFrame([{"Zone":zid,"Sev":f'{d.get("severity",0):.3f}',
                                    "Flood":f'{d.get("flood_score",0):.3f}',"Dmg":f'{d.get("damage_score",0):.3f}',
                                    "Level":_sev_label(d.get("severity",0))} for zid,d in top]),
                     use_container_width=True, hide_index=True)
        st.success(f"✅ {len(zone_map)} zones analysed")
    _terminal(); st.divider()
    _auto_advance(next_stage=2, delay=4.0)

# ============================================================================
#  STAGE 3 — DRONE DEPLOYMENT (all 5 zones)
# ============================================================================
def stage_3():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🚁 Stage 3: Drone Deployment</h2>', unsafe_allow_html=True)
    _phase_badge()
    gs=_graph_state(); alloc=gs.get("drone_allocation",{}); most=gs.get("most_affected_zones",[])
    if not alloc:
        st.warning("Drone allocation not in graph state yet.")
        _terminal()
        if st.button("◀ Back",key="b3"): st.session_state["stage"]=1; st.rerun()
        return
    if most:
        st.markdown(f'<div class="card">📍 <b>Top 5 Crisis Zones</b>: '
                    f'<span style="color:{THEME["cyan"]};">{", ".join(most)}</span></div>', unsafe_allow_html=True)
    # Show all 5 drones
    n = min(len(alloc), 5)
    cols = st.columns(n)
    for i,(d_id,z_id) in enumerate(list(alloc.items())[:n]):
        with cols[i]:
            st.markdown(f'<div class="card" style="text-align:center;">'
                        f'<b style="color:{THEME["cyan"]};">{d_id.upper()}</b><br>'
                        f'<span style="font-size:28px;">🚁</span><br>'
                        f'<b style="color:{THEME["orange"]};">→ {z_id}</b><br>'
                        f'<span style="color:{THEME["green"]};font-size:11px;">✅ DISPATCHED</span>'
                        f'</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Drone":k,"Zone":v,"Status":"✅ Dispatched"} for k,v in alloc.items()]),
                 use_container_width=True, hide_index=True)
    _terminal(); st.divider()
    _auto_advance(next_stage=3, delay=3.0)

# ============================================================================
#  STAGE 4 — DRONE IMAGERY (realistic HUD overlays)
# ============================================================================
def stage_4():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📡 Stage 4: Live Drone Imagery</h2>', unsafe_allow_html=True)
    _phase_badge()
    gs=_graph_state(); alloc=gs.get("drone_allocation",{}); counts=gs.get("people_counts",{})
    zone_image_map=gs.get("zone_image_map",{}); akv=list(alloc.items())

    # Telemetry strip
    cm1,cm2,cm3,cm4 = st.columns(4)
    with cm1: st.metric("Drones Active",len(alloc))
    with cm2: st.metric("Zones Covered",len(alloc))
    with cm3: st.metric("Total Victims",sum(counts.values()) if counts else "—")
    with cm4: st.metric("Disaster", "FLOOD")
    st.divider()

    # Collect images
    imgs = {}
    for zone_id, img_path in zone_image_map.items():
        full = img_path if os.path.isabs(img_path) else os.path.join(_ROOT,img_path)
        if os.path.exists(full):
            try: imgs[zone_id] = Image.open(full)
            except Exception: pass
    if not imgs:
        p = Path(os.path.join(_ROOT,"zone_images"))
        if p.exists():
            zone_ids = [v for _,v in akv]
            for i,f in enumerate(sorted(list(p.glob("*.jpg"))+list(p.glob("*.jpeg"))+list(p.glob("*.png")))):
                zid = zone_ids[i] if i<len(zone_ids) else f"ZONE_{i+1}"
                try: imgs[zid] = Image.open(f)
                except Exception: pass

    drone_for_zone = {v:k for k,v in akv}
    zones_to_show  = [v for _,v in akv[:5]]

    st.markdown(f'<h4 style="color:{THEME["cyan"]};">📷 LIVE DRONE FEEDS — {len(alloc)} UNITS ACTIVE</h4>', unsafe_allow_html=True)
    n_cols = min(len(zones_to_show),3)
    if n_cols:
        cols = st.columns(n_cols)
        for idx,zone_id in enumerate(zones_to_show):
            drone_id=drone_for_zone.get(zone_id,f"drone_{idx+1}"); pcount=counts.get(zone_id,0)
            with cols[idx%n_cols]:
                if zone_id in imgs:
                    ann = _drone_overlay(imgs[zone_id], zone_id=zone_id, drone_id=drone_id,
                                         people_count=pcount, is_flood=True)
                    st.image(ann, use_container_width=True)
                else:
                    st.markdown(f'<div style="background:#0a0a0a;border:2px dashed {THEME["red"]};'
                                f'height:180px;display:flex;align-items:center;justify-content:center;'
                                f'flex-direction:column;border-radius:4px;">'
                                f'<span style="font-size:28px;">🚁</span><br>'
                                f'<span style="color:{THEME["cyan"]};font-family:\'Share Tech Mono\';font-size:10px;">ACQUIRING FEED…</span><br>'
                                f'<span style="color:{THEME["yellow"]};font-family:\'Share Tech Mono\';font-size:9px;">ZONE {zone_id}</span>'
                                f'</div>', unsafe_allow_html=True)
                sev_c = THEME["red"] if pcount>5 else THEME["orange"] if pcount>0 else "#888"
                st.markdown(f'<div style="font-family:\'Share Tech Mono\';font-size:11px;padding:4px;background:{THEME["bg2"]};border-radius:2px;">'
                            f'🚁 <b style="color:{THEME["cyan"]};">{drone_id.upper()}</b> → <b>{zone_id}</b> | '
                            f'<span style="color:{sev_c};">👤 {pcount} victims</span></div>', unsafe_allow_html=True)

    # Victim detection annotations
    rp = Path(os.path.join(_ROOT,"zone_results"))
    annotated = {}
    if rp.exists():
        for f in sorted(rp.glob("*_analysis.jpg")):
            zk=f.stem.replace("_analysis","")
            try: annotated[zk]=Image.open(f)
            except Exception: pass
    if annotated:
        st.divider()
        st.markdown(f'<h4 style="color:{THEME["cyan"]};">🔍 Victim Detection</h4>', unsafe_allow_html=True)
        cols2=st.columns(3)
        for idx,(zone_id,img) in enumerate(annotated.items()):
            with cols2[idx%3]:
                ann2=_drone_overlay(img, zone_id=zone_id, drone_id=drone_for_zone.get(zone_id,f"drone_{idx+1}"),
                                    people_count=counts.get(zone_id,0), is_flood=True)
                st.image(ann2, use_container_width=True)
                st.caption(f"Zone **{zone_id}** — 👤 {counts.get(zone_id,0)} people")

    if counts:
        st.divider()
        st.dataframe(pd.DataFrame([{"Zone":k,"👤 People":v,"Status":"✅ Detected" if v>0 else "⚠️ 0"}
                                    for k,v in counts.items()]),use_container_width=True,hide_index=True)
        st.success(f"✅ **{sum(counts.values())} people** across **{len(counts)} zones**")
    _terminal(); st.divider()
    _auto_advance(next_stage=4, delay=4.0)

# ============================================================================
#  STAGE 5 — ZONE ANALYSIS
# ============================================================================
def stage_5():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📊 Stage 5: Zone Analysis</h2>', unsafe_allow_html=True)
    _phase_badge()
    gs=_graph_state(); people=gs.get("people_counts",{}); zm=gs.get("zone_map",{}); top=gs.get("most_affected_zones",[])
    c1,c2=st.columns(2)
    with c1:
        zones=top or list(zm.keys())[:15]
        rows=[{"Zone":z,"👤 People":people.get(z,0),"Severity":f'{zm.get(z,{}).get("severity",0):.3f}',
               "Flood":f'{zm.get(z,{}).get("flood_score",0):.3f}',"Dmg":f'{zm.get(z,{}).get("damage_score",0):.3f}',
               "Level":_sev_label(zm.get(z,{}).get("severity",0))} for z in zones]
        if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        if os.path.exists(DB_PATH):
            try:
                conn=sqlite3.connect(DB_PATH)
                df_db=pd.read_sql_query("SELECT zone_id,severity,flood_score,damage_score,people_count,last_updated FROM zones ORDER BY severity DESC LIMIT 10",conn)
                conn.close(); st.markdown("**📦 crisis.db**"); st.dataframe(df_db,use_container_width=True,hide_index=True)
            except Exception as e: st.caption(f"DB error: {e}")
    with c2:
        shown=0; rp=Path(os.path.join(_ROOT,"zone_results"))
        if rp.exists():
            for f in sorted(list(rp.glob("*.jpg"))+list(rp.glob("*.png"))):
                if f.name.startswith("route_map"): continue
                try:
                    st.image(_sat_overlay(Image.open(f),is_flood=True,img_name=f.name),caption=f.stem,use_container_width=True)
                    shown+=1
                except Exception: pass
                if shown>=4: break
        if not shown: st.info("Result images appear after Phase 1 runs.")
    _terminal(); st.divider()
    _auto_advance(next_stage=5, delay=3.0)

# ============================================================================
#  STAGE 6 — RESOURCE ALLOCATION
# ============================================================================
def stage_6():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📦 Stage 6: Resource Allocation</h2>', unsafe_allow_html=True)
    _phase_badge()
    gs=_graph_state(); plan=gs.get("rescue_plan",{})
    if not plan:
        st.warning("Rescue plan not in graph state yet.")
        _terminal()
        if st.button("◀ Back",key="b6"): st.session_state["stage"]=4; st.rerun()
        return
    st.success("✅ Rescue plan generated by Gemini LLM")
    rows=[]; totals={}
    for z,al in plan.items():
        row={"Zone":z}
        for rt,cnt in al.items(): row[rt]=cnt; totals[rt]=totals.get(rt,0)+cnt
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).fillna(0),use_container_width=True,hide_index=True)
    if totals:
        st.divider(); mc=st.columns(len(totals))
        for i,(k,v) in enumerate(totals.items()):
            with mc[i]: st.metric(k.replace("_"," ").title(),int(v))
    _terminal(); st.divider()
    _auto_advance(next_stage=6, delay=3.0)

# ============================================================================
#  STAGE 7 — HITL GATE #1: RESOURCE APPROVAL  ← MANUAL ONLY
# ============================================================================
def stage_7():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">✅ Stage 7: Admin Approval — Resources</h2>', unsafe_allow_html=True)
    _phase_badge()
    st.markdown(f'<div class="card-warn">⏸️ <b>HUMAN-IN-THE-LOOP GATE #1</b> — Review the rescue plan before route planning begins.</div>', unsafe_allow_html=True)
    gs   = _graph_state()
    plan = gs.get("rescue_plan", {})
    phase       = st.session_state.get("pipeline_phase", "idle")
    max_reached = st.session_state.get("max_stage", 0)

    if not plan:
        st.warning("No rescue plan — go back to Stage 6.")
        if st.button("◀ Back", key="b7x"): st.session_state["stage"] = 5; st.rerun()
        return

    # ── Reviewing mode: already approved, just show plan + forward button ─
    if gs.get("resource_approved") or phase in ("awaiting_route","running_phase2","running_phase3","complete"):
        st.success("✅ Resource allocation APPROVED")
        st.markdown("**Approved Rescue Resource Allocation:**")
        for z, al in plan.items():
            desc = " · ".join(f"{v}× {k}" for k, v in al.items() if v)
            st.markdown(f'<div class="card"><b style="color:{THEME["cyan"]};">Zone {z}</b>  →  {desc}</div>', unsafe_allow_html=True)
        st.divider()
        if st.button("▶ Go to Stage 8 — Route Planning", key="f7_fwd", use_container_width=True):
            st.session_state["stage"]     = 7
            st.session_state["max_stage"] = max(max_reached, 7)
            st.rerun()
        _terminal()
        return

    st.markdown("**Proposed Rescue Resource Allocation (Gemini LLM)**")
    for z, al in plan.items():
        desc = " · ".join(f"{v}× {k}" for k, v in al.items() if v)
        st.markdown(f'<div class="card"><b style="color:{THEME["cyan"]};">Zone {z}</b>  →  {desc}</div>', unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ APPROVE — TRIGGER ROUTE PLANNING", key="app1", use_container_width=True):
            _log("ADMIN ✓ Resources APPROVED — running route agent")
            st.session_state["pipeline_phase"] = "running_phase2"
            prog   = st.progress(0, "🗺️ Connecting to OpenStreetMap…")
            status = st.empty()
            try:
                status.info("🗺️ Route Agent computing road paths — this may take 30–90 s…")
                prog.progress(25, "Building road network graph…")
                _run_phase2(approved=True)
                prog.progress(100, "✅ Routes computed!")
                status.success("✅ Route planning complete!")
                st.balloons()
            except Exception as e:
                _log(f"[ERROR] Phase 2:\n{traceback.format_exc()}")
                prog.empty(); status.empty()
                st.error(f"Route planning error: {e}")
                st.rerun()
                return
            time.sleep(0.8)
            st.session_state["stage"]     = 7
            st.session_state["max_stage"] = max(max_reached, 7)
            st.rerun()
    with c2:
        if st.button("🔴 REJECT — RE-RUN LLM", key="hold1", use_container_width=True):
            import uuid; _log("ADMIN ✗ Rejected — restarting Phase 1")
            st.session_state["thread_id"] = f"aegis_{uuid.uuid4().hex[:8]}"
            img_path = st.session_state.get("flood_image_path") or "Images_for_testing/image.png"
            meta = st.session_state.get("image_meta", _DEFAULT_META.copy())
            st.session_state["pipeline_phase"] = "running_phase1"
            with st.spinner("🔄 Re-running Phase 1 …"):
                try: _run_phase1(img_path, meta)
                except Exception as e: _log(f"[ERROR] Re-run: {e}")
            st.rerun()
    _terminal()

# ============================================================================
#  STAGE 8 — ROUTE PLANNING
# ============================================================================
def stage_8():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🗺️ Stage 8: Route Planning</h2>', unsafe_allow_html=True)
    _phase_badge()
    gs=_graph_state(); routes=gs.get("route_plan",[])
    # Only show "running" spinner if phase2 is truly in-progress AND routes aren't ready yet
    if st.session_state.get("pipeline_phase")=="running_phase2" and not routes:
        st.info("🗺️ Route planning running …")
        _terminal()
        time.sleep(2); st.rerun()
        return
    # If phase got stuck on running_phase2 but routes are actually ready, fix the phase
    if st.session_state.get("pipeline_phase")=="running_phase2" and routes:
        st.session_state["pipeline_phase"] = "awaiting_route"
    if not routes:
        st.warning("Route plan not in graph state yet.")
        _terminal()
        if st.button("◀ Back",key="b8x"): st.session_state["stage"]=6; st.rerun()
        return
    st.success(f"✅ {sum(1 for r in routes if r.get('success'))} / {len(routes)} routes planned")
    st.dataframe(pd.DataFrame([{"Zone":r.get("zone"),"Resource":f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
                                "Units":r.get("unit_count",1),"From":r.get("origin_name"),
                                "Dist km":r.get("distance_km",0),"ETA min":r.get("eta_minutes",0),
                                "Status":"✓ OK" if r.get("success") else f'✗ {r.get("error","?")}'}
                               for r in routes]),use_container_width=True,hide_index=True)
    st.markdown("**Interactive Route Map — Real OSM Waypoints**")
    st_folium(_folium_map(), width=None, height=460, key="fmap8", returned_objects=[])
    _terminal(); st.divider()
    _auto_advance(next_stage=8, delay=5.0)

# ============================================================================
#  STAGE 9 — HITL GATE #2: ROUTE APPROVAL  ← MANUAL ONLY
# ============================================================================
def stage_9():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">✅ Stage 9: Admin Approval — Routes</h2>', unsafe_allow_html=True)
    _phase_badge()
    st.markdown(f'<div class="card-warn">⏸️ <b>HUMAN-IN-THE-LOOP GATE #2</b> — Approve routes before dispatching emergency resources.</div>', unsafe_allow_html=True)
    gs   = _graph_state()
    routes      = gs.get("route_plan", [])
    phase       = st.session_state.get("pipeline_phase", "idle")
    max_reached = st.session_state.get("max_stage", 0)

    if not routes:
        st.warning("No route plan — go back to Stage 8.")
        if st.button("◀ Back", key="b9x"): st.session_state["stage"] = 7; st.rerun()
        return

    # ── Reviewing mode: already approved, just show routes + forward button ─
    if gs.get("route_approved") or phase in ("running_phase3", "complete"):
        st.success("✅ Routes APPROVED")
        st.dataframe(pd.DataFrame([{
            "Resource": f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
            "Units":    r.get("unit_count", 1),
            "Zone":     r.get("zone"),
            "Dist km":  r.get("distance_km", 0),
            "ETA min":  r.get("eta_minutes", 0),
            "Status":   "✓ OK" if r.get("success") else "✗ FAILED",
        } for r in routes]), use_container_width=True, hide_index=True)
        st.divider()
        if st.button("▶ Go to Stage 10 — Communications", key="f9_fwd", use_container_width=True):
            st.session_state["stage"]     = 9
            st.session_state["max_stage"] = max(max_reached, 9)
            st.rerun()
        _terminal()
        return

    st.dataframe(pd.DataFrame([{
        "Resource": f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
        "Units":    r.get("unit_count", 1),
        "Zone":     r.get("zone"),
        "From":     r.get("origin_name"),
        "Dist km":  r.get("distance_km", 0),
        "ETA min":  r.get("eta_minutes", 0),
        "Status":   "✓ OK" if r.get("success") else "✗ FAILED",
    } for r in routes]), use_container_width=True, hide_index=True)

    st.divider(); c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ APPROVE ROUTES & DISPATCH", key="app2", use_container_width=True):
            _log("ADMIN ✓ Routes APPROVED — Phase 3 starting")
            st.session_state["pipeline_phase"] = "running_phase3"
            with st.spinner("📡 Communication Agent dispatching …"):
                try:
                    _run_phase3(approved=True)
                    st.success("✅ Dispatch ready!"); st.balloons()
                except Exception as e:
                    _log(f"[ERROR] Phase 3:\n{traceback.format_exc()}"); st.error(f"Phase 3 error: {e}")
            st.session_state["stage"]     = 9
            st.session_state["max_stage"] = max(max_reached, 9)
            st.rerun()
    with c2:
        if st.button("🔴 REJECT — RE-PLAN ROUTES", key="hold2", use_container_width=True):
            _log("ADMIN ✗ Routes REJECTED"); st.session_state["pipeline_phase"] = "running_phase2"
            with st.spinner("🔄 Re-running routes …"):
                try: _run_phase2(approved=True)
                except Exception as e: _log(f"[ERROR] {e}")
            st.rerun()
    _terminal()

# ============================================================================
#  STAGE 10 — COMMUNICATION AGENT
# ============================================================================
def stage_10():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📡 Stage 10: Communication Agent</h2>', unsafe_allow_html=True)
    _phase_badge()

    phase = st.session_state.get("pipeline_phase", "idle")

    gs       = _graph_state()
    dispatch = gs.get("dispatch_result", {})

    # Only block rendering if phase3 is truly in-progress AND dispatch isn't ready yet
    if phase == "running_phase3" and not dispatch:
        st.info("📡 Communication Agent running …")
        _terminal()
        time.sleep(2); st.rerun()
        return

    # If dispatch is ready but phase is stale, auto-correct it
    if phase == "running_phase3" and dispatch:
        st.session_state["pipeline_phase"] = "complete"
        phase = "complete"
    routes   = gs.get("route_plan", [])

    # ── Pipeline complete banner + image move ─────────────────────────────
    if phase == "complete":
        st.markdown(
            f'<div style="background:{THEME["bg2"]};border:2px solid {THEME["green"]};'
            f'border-radius:8px;padding:20px;text-align:center;margin-bottom:16px;">'
            f'<span style="color:{THEME["green"]};font-family:\'Rajdhani\';font-size:28px;font-weight:bold;">'
            f'🎯 AEGIS PIPELINE COMPLETE</span><br>'
            f'<span style="color:{THEME["text"]};font-size:13px;">All {len(STAGES)} stages executed via LangGraph</span>'
            f'</div>', unsafe_allow_html=True)

        # Move flood image once
        if not st.session_state.get("_image_moved"):
            flood_img = st.session_state.get("flood_image_path")
            if flood_img and Path(flood_img).exists():
                try:
                    dest = _move_to_scanned(flood_img)
                    _log(f"✅ Flood image moved → scanned_images/{dest.name}")
                except Exception as e:
                    _log(f"[WARN] Could not move image: {e}")
            st.session_state["_image_moved"] = True

        if st.session_state.get("_image_moved"):
            flood_img = st.session_state.get("flood_image_path", "")
            if flood_img:
                st.markdown(
                    f'<div class="card-ok">📁 <b>Image archived:</b> '
                    f'<code>{Path(flood_img).name}</code> → <code>scanned_images/</code></div>',
                    unsafe_allow_html=True)

    # ── Dispatch instructions ──────────────────────────────────────────────
    instructions = (dispatch or {}).get("instructions", {})
    if instructions or routes:
        st.markdown("**📋 Dispatch Instructions (Gemini LLM)**")
        if instructions:
            for z, instr in instructions.items():
                text = instr if isinstance(instr, str) else json.dumps(instr, indent=2)
                st.markdown(
                    f'<div class="card"><b style="color:{THEME["cyan"]};">Zone {z}</b><br>'
                    f'<pre style="margin:8px 0 0;font-size:11px;color:{THEME["text"]};white-space:pre-wrap;">{text}</pre></div>',
                    unsafe_allow_html=True)
        else:
            for r in routes:
                em    = _remoji(r.get("resource_type", ""))
                rtype = r.get("resource_type", "").replace("_", " ").title()
                st.markdown(
                    f'<div class="card"><b style="color:{THEME["cyan"]};">{em} {r.get("unit_count",1)}× {rtype} → Zone {r.get("zone")}</b><br>'
                    f'<span style="font-family:\'Share Tech Mono\';font-size:11px;">'
                    f'From: {r.get("origin_name","?")} · {r.get("distance_km","?")} km · ETA {r.get("eta_minutes","?")} min'
                    f'</span></div>', unsafe_allow_html=True)
    else:
        st.info("📡 Communication Agent dispatching — results will appear shortly.")

    summary = (dispatch or {}).get("summary", "")
    if summary: st.info(f"**Commander Summary:** {summary}")

    # ── SMS status ─────────────────────────────────────────────────────────
    st.markdown("**📱 SMS Dispatch Status**")
    from dotenv import load_dotenv; load_dotenv()
    sms_results = (dispatch or {}).get("sms_results", [])
    twilio_ok   = all([os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"),
                       os.getenv("TWILIO_PHONE_NUMBER"), os.getenv("YOUR_PHONE_NUMBER")])
    if sms_results:
        for res in sms_results:
            if res.get("success"):
                st.markdown(f'<div class="card-ok">✅ SMS sent → Zone <b>{res.get("zone","")}</b> · SID: <code>{res.get("sid","")}</code></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card-flood">❌ SMS FAILED → Zone <b>{res.get("zone","")}</b> · {res.get("error","unknown")}</div>', unsafe_allow_html=True)
    elif not twilio_ok:
        st.markdown(f'<div class="card-warn">⚠️ SMS not sent — Twilio not configured.</div>', unsafe_allow_html=True)

    audio = (dispatch or {}).get("audio_files", [])
    if audio:
        st.markdown("**🔊 Audio Dispatch Files**")
        for fpath in audio:
            if os.path.exists(fpath): st.audio(fpath); st.caption(fpath)

    _terminal(); st.divider()

    # ── Auto-countdown restart (only when complete + image moved) ─────────
    if phase == "complete" and st.session_state.get("_image_moved"):
        key = "_adv_restart"
        if not st.session_state.get(key):
            st.session_state[key] = time.time()
        elapsed   = time.time() - st.session_state[key]
        remaining = max(0.0, 15.0 - elapsed)

        bar_col, btn_col = st.columns([5, 1])
        with bar_col:
            st.progress(min(elapsed / 15.0, 1.0),
                        text=f"⏩ Restarting watcher in {remaining:.0f}s…")
        with btn_col:
            st.markdown('<div class="step-skip">', unsafe_allow_html=True)
            if st.button("Now →", key="_skip_restart"):
                remaining = 0
            st.markdown('</div>', unsafe_allow_html=True)

        if remaining <= 0:
            _do_full_reset()
            return
        time.sleep(min(remaining, 2.0))
        st.rerun()
        return

    # ── Always-visible restart button ─────────────────────────────────────
    if st.button("🔁 RESTART WATCHER NOW", key="done10", use_container_width=True):
        _do_full_reset()

# ============================================================================
#  MAIN
# ============================================================================
STAGE_FNS = [stage_1,stage_2,stage_3,stage_4,stage_5,stage_6,stage_7,stage_8,stage_9,stage_10]

def main():
    st.session_state.setdefault("stage",          0)
    st.session_state.setdefault("max_stage",      0)
    st.session_state.setdefault("log",            "")
    st.session_state.setdefault("pipeline_phase", "watching")
    st.session_state.setdefault("thread_id",      "aegis_main")
    _sidebar()
    st.markdown(f'<h1 style="color:{THEME["cyan"]};font-family:\'Rajdhani\';text-align:center;">🛰️ AEGIS · Flood Crisis Management AI</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{THEME["mono"]};text-align:center;font-family:\'Share Tech Mono\';">Automated Satellite Watcher · Flood Detection · LangGraph Pipeline</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<p style="color:{THEME["text"]};font-family:\'Share Tech Mono\';font-size:12px;">PIPELINE PROGRESS</p>', unsafe_allow_html=True)
    _stepper(); st.divider()
    STAGE_FNS[st.session_state.get("stage",0)]()

if __name__ == "__main__":
    main()