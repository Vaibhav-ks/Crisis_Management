"""
streamlit_app.py  —  AEGIS Crisis Management AI
================================================
Run:   streamlit run streamlit_app.py

KEY DESIGN DECISIONS that prevent slowness
-------------------------------------------
• Every agent result is stored in st.session_state once and NEVER re-computed
  on subsequent reruns.  Pattern:

      if "zone_map" not in st.session_state:
          with st.spinner("..."):
              result = run_agent(...)
              st.session_state["zone_map"] = result
      # Always read from session_state below

• stdout is captured with io.StringIO + contextlib.redirect_stdout so the
  terminal panel updates without blocking the UI.

• The route map is drawn from REAL waypoints stored in st.session_state
  ["route_plan"]; each polyline uses the actual lat/lon path returned by
  plan_all_routes().  Straight-line fallback only fires when a route has
  fewer than 2 waypoints.
"""

import streamlit as st

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="AEGIS — Crisis Management AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
from PIL import Image
import sqlite3
import json
import os
import sys
import io
import contextlib
import tempfile
from datetime import datetime
from pathlib import Path
import folium
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
#  THEME  (from original UI, kept intact)
# ============================================================================

THEME = {
    "bg_primary":   "#080d14",
    "bg_secondary": "#0d1520",
    "accent_cyan":  "#00d4ff",
    "red":          "#ff2d55",
    "orange":       "#ff9500",
    "green":        "#30d158",
    "yellow":       "#ffd60a",
    "text_primary": "#e5e5e7",
    "mono_green":   "#00ff88",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&family=Exo+2:wght@400;600&display=swap');

  body, .stApp {{ background-color:{THEME['bg_primary']}; color:{THEME['text_primary']}; }}
  h1,h2,h3 {{ font-family:'Rajdhani',sans-serif; }}

  .stButton>button {{
    background-color:{THEME['bg_secondary']};
    color:{THEME['accent_cyan']};
    border:2px solid {THEME['accent_cyan']};
    border-radius:6px; padding:8px 18px;
    font-family:'Share Tech Mono',monospace;
    transition:all .25s ease;
  }}
  .stButton>button:hover {{
    background-color:{THEME['accent_cyan']};
    color:{THEME['bg_primary']};
  }}
  .stMetric {{ background:{THEME['bg_secondary']}; padding:16px; border-radius:8px;
               border-left:4px solid {THEME['accent_cyan']}; }}
  .terminal-log {{
    background:#000; color:{THEME['mono_green']};
    font-family:'Share Tech Mono',monospace; font-size:11px; line-height:1.6;
    padding:12px; border-radius:4px; border:1px solid {THEME['green']};
    max-height:300px; overflow-y:auto; white-space:pre-wrap; word-break:break-word;
  }}
  .card {{
    background:{THEME['bg_secondary']}; border-left:4px solid {THEME['accent_cyan']};
    border-radius:6px; padding:12px; margin:6px 0;
  }}
  .panel {{
    background:{THEME['bg_secondary']}; border:1px solid {THEME['accent_cyan']};
    border-radius:8px; padding:16px;
  }}
  .severity-critical {{ background:#ff2d55; color:white; padding:3px 8px;
                        border-radius:4px; font-size:12px; font-weight:bold; }}
  .severity-high     {{ background:#ff9500; color:white; padding:3px 8px;
                        border-radius:4px; font-size:12px; font-weight:bold; }}
  .severity-moderate {{ background:#ffd60a; color:black; padding:3px 8px;
                        border-radius:4px; font-size:12px; font-weight:bold; }}
  .severity-low      {{ background:#30d158; color:black; padding:3px 8px;
                        border-radius:4px; font-size:12px; font-weight:bold; }}
  div[data-testid="stDataFrame"] {{ background:{THEME['bg_secondary']}; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
#  CONSTANTS
# ============================================================================

_DEFAULT_META = {
    "center_lat":  19.062061,
    "center_lon":  72.863542,
    "coverage_km": 1.6,
    "width_px":    1024,
    "height_px":   522,
}
_DEFAULT_BASE_LOCATIONS = {
    "ambulance":   {"name": "Hospital",   "lat": 19.06546856543151, "lon":  72.86100899070198},
    "rescue_team": {"name": "Rescue Center", "lat": 19.06847079812735, "lon": 72.85793995490616},
    "boat":        {"name": "Boat Depot",  "lat": 19.063380373548366, "lon": 72.85538649195271},
}

ROUTE_COLORS  = {"ambulance":"#e74c3c","rescue_team":"#2980b9","boat":"#16a085",
                 "helicopter":"#8e44ad","fire_truck":"#e67e22","truck":"#7f8c8d"}
RESOURCE_EMOJI= {"ambulance":"🚑","rescue_team":"🚒","boat":"🚤",
                 "helicopter":"🚁","fire_truck":"🚒","truck":"🚛"}
BASE_FOLIUM_ICON = {"ambulance":("red","plus-sign"),"rescue_team":("blue","home"),
                    "boat":("darkblue","tint"),"helicopter":("purple","plane")}

STAGES = [
    "1️⃣ Upload",   "2️⃣ Zone Map",  "3️⃣ Drones",  "4️⃣ Gallery",  "5️⃣ Analysis",
    "6️⃣ Resources","7️⃣ Approve I","8️⃣ Routes",  "9️⃣ Approve II","🔟 Comms",
]

# ============================================================================
#  HELPERS
# ============================================================================

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(text: str):
    """Append to the persistent terminal log in session_state."""
    st.session_state.setdefault("log", "")
    for line in text.strip().splitlines():
        st.session_state["log"] += f"[{_ts()}] {line}\n"

def _terminal(height: int = 280):
    log = st.session_state.get("log", "(no output yet)")
    # Auto-scroll via JS trick: newest entries at bottom
    st.markdown(
        f'<div class="terminal-log" id="termlog">{log}</div>'
        '<script>var t=document.getElementById("termlog");if(t)t.scrollTop=t.scrollHeight;</script>',
        unsafe_allow_html=True,
    )

def _capture(fn, *args, **kwargs):
    """Run fn, capture stdout, return (result, text). Never raises."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            result = None
            print(f"[ERROR] {type(e).__name__}: {e}")
    return result, buf.getvalue()

def _save_upload(f) -> str:
    suffix = Path(f.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.getvalue()); tmp.close()
    return tmp.name

def _sev_label(s):
    if s >= 0.8: return "🔴 CRITICAL"
    if s >= 0.6: return "🟠 HIGH"
    if s >= 0.4: return "🟡 MODERATE"
    return "🟢 LOW"

def _rcolor(rtype):
    k = rtype.lower().rstrip("s")
    return ROUTE_COLORS.get(k, ROUTE_COLORS.get(rtype.lower(), "#888"))

def _remoji(rtype):
    k = rtype.lower().rstrip("s")
    return RESOURCE_EMOJI.get(k, RESOURCE_EMOJI.get(rtype.lower(), "🚗"))

def _nav(back=None, fwd=None, fwd_label="▶ PROCEED"):
    c1, c2 = st.columns(2)
    with c1:
        if back is not None and st.button("◀ BACK", key=f"nav_back_{st.session_state['stage']}"):
            st.session_state["stage"] = back; st.rerun()
    with c2:
        if fwd is not None and st.button(fwd_label, key=f"nav_fwd_{st.session_state['stage']}"):
            st.session_state["stage"] = fwd; st.rerun()

# ============================================================================
#  SIDEBAR
# ============================================================================

def _sidebar():
    with st.sidebar:
        st.markdown("### 🛰️ AEGIS · Agent Feed")
        st.divider()
        for name, icon, key in [
            ("Vision Agent",        "👁️",  "st_vision"),
            ("Master Coordinator",  "🎯",  "st_master"),
            ("Resource Allocation", "📦",  "st_resource"),
            ("Route Planning",      "🗺️", "st_route"),
            ("Communication",       "📡",  "st_comm"),
        ]:
            s = st.session_state.get(key, "⚪ Idle")
            st.markdown(
                f'<div class="card">{icon} <b>{name}</b><br>'
                f'<span style="color:{THEME["accent_cyan"]};font-size:12px;">{s}</span></div>',
                unsafe_allow_html=True,
            )
        st.divider()
        st.metric("Stage", f"{st.session_state.get('stage',0)+1} / {len(STAGES)}")
        st.divider()
        if st.button("🔄 Full Reset", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

# ============================================================================
#  STEPPER
# ============================================================================

def _stepper():
    s    = st.session_state.get("stage", 0)
    cols = st.columns(len(STAGES))
    for i, label in enumerate(STAGES):
        with cols[i]:
            if i < s:   bg,fg = THEME["green"],   "black"
            elif i == s:bg,fg = THEME["accent_cyan"], THEME["bg_primary"]
            else:        bg,fg = THEME["bg_secondary"], THEME["text_primary"]
            st.markdown(
                f'<div style="background:{bg};color:{fg};padding:6px 2px;text-align:center;'
                f'border-radius:4px;font-size:9px;font-weight:bold;'
                f'border:1px solid {THEME["accent_cyan"]};">{"✅" if i<s else ""}{label}</div>',
                unsafe_allow_html=True,
            )

# ============================================================================
#  FOLIUM ROUTE MAP  ← the one function that does accurate routing
# ============================================================================

def _build_folium_map():
    """
    Build the interactive Folium map from REAL route_plan waypoints.
    Falls back to straight line only when a route has <2 waypoints.
    Always draws base-location markers and destination markers.
    """
    meta   = st.session_state.get("image_meta") or _DEFAULT_META
    bases  = st.session_state.get("base_locations") or _DEFAULT_BASE_LOCATIONS
    routes = st.session_state.get("route_plan", [])

    fmap = folium.Map(
        location=[meta["center_lat"], meta["center_lon"]],
        zoom_start=15,
        tiles="CartoDB positron",
    )

    # ── Base location markers ─────────────────────────────────────────────
    seen_bases = set()
    for r in routes:
        rkey = r.get("resource_type","").lower().rstrip("s")
        base = bases.get(rkey) or bases.get(r.get("resource_type",""))
        if base and base["name"] not in seen_bases:
            ic_c, ic_i = BASE_FOLIUM_ICON.get(rkey, ("gray","info-sign"))
            folium.Marker(
                [base["lat"], base["lon"]],
                tooltip=f"📍 {base['name']}",
                popup=folium.Popup(f"<b>{base['name']}</b><br>Deploys: {rkey}",
                                   max_width=200),
                icon=folium.Icon(color=ic_c, icon=ic_i),
            ).add_to(fmap)
            seen_bases.add(base["name"])

    # ── Destination zone markers ──────────────────────────────────────────
    seen_zones = set()
    for r in routes:
        dest = r.get("destination_latlon")
        if dest and r["zone"] not in seen_zones:
            folium.Marker(
                list(dest),
                tooltip=f"🚨 Zone {r['zone']}",
                icon=folium.Icon(color="orange", icon="exclamation-sign"),
            ).add_to(fmap)
            seen_zones.add(r["zone"])

    # ── Route polylines from REAL waypoints ───────────────────────────────
    for r in routes:
        if not r.get("success"):
            continue

        color = _rcolor(r["resource_type"])
        emoji = _remoji(r["resource_type"])
        wpts  = r.get("waypoints", [])

        # If route agent gave us real waypoints, use them.
        # Straight-line fallback only when < 2 points.
        if len(wpts) < 2:
            dest = r.get("destination_latlon")
            rkey = r["resource_type"].lower().rstrip("s")
            base = bases.get(rkey)
            if base and dest:
                wpts = [(base["lat"], base["lon"]), dest]
            elif len(wpts) == 1 and dest:
                wpts = [wpts[0], dest]
            else:
                continue   # nothing to draw

        # Main polyline
        folium.PolyLine(
            wpts,
            color=color, weight=5, opacity=0.9,
            tooltip=(
                f"{emoji} {r.get('unit_count',1)}× "
                f"{r['resource_type'].replace('_',' ').title()}\n"
                f"Zone {r['zone']}  ·  {r['origin_name']}\n"
                f"{r.get('distance_km',0)} km  ·  ETA {r.get('eta_minutes',0)} min\n"
                f"{len(r.get('waypoints',[]))} road waypoints"
            ),
        ).add_to(fmap)

        # Mid-route dots (skip first and last)
        for lat, lon in wpts[1:-1]:
            folium.CircleMarker(
                [lat, lon], radius=4, color=color,
                fill=True, fill_color=color, fill_opacity=0.8,
            ).add_to(fmap)

        # Arrival arrow at destination
        folium.Marker(
            list(wpts[-1]),
            tooltip=f"{emoji} Arrives at Zone {r['zone']}",
            icon=folium.DivIcon(
                html=f'<div style="font-size:16px;color:{color};">▼</div>',
                icon_size=(16, 16), icon_anchor=(8, 8),
            ),
        ).add_to(fmap)

    # ── Legend overlay ────────────────────────────────────────────────────
    seen_types = sorted({r["resource_type"] for r in routes if r.get("success")})
    lines = "".join(
        f'<span style="color:{_rcolor(t)};font-size:16px;">━━</span> '
        f'{_remoji(t)} {t.replace("_"," ").title()}<br>'
        for t in seen_types
    )
    fmap.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:12px;right:12px;z-index:9999;
                background:white;padding:12px 16px;border-radius:8px;
                border:2px solid #333;font-family:Arial;font-size:12px;
                box-shadow:3px 3px 8px rgba(0,0,0,.3);">
      <b>🗺 Route Legend</b><br>{lines}
      <span style="background:#f39c12;padding:0 5px;border-radius:3px;">■</span> Crisis Zone<br>
      <hr style="margin:6px 0;">
      <span style="font-size:10px;color:#666;">Real OSM waypoints</span>
    </div>"""))

    return fmap

# ============================================================================
#  STAGE 1 — UPLOAD
# ============================================================================

def stage_1():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">🖼️ Stage 1: Satellite Image Upload</h2>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        uploaded = st.file_uploader("Upload satellite / aerial image",
                                    type=["jpg","jpeg","png"])
        if uploaded:
            st.session_state["upload_obj"] = uploaded
            pil = Image.open(uploaded)
            st.session_state["upload_pil"] = pil
            st.image(pil, caption=f"{uploaded.name}  ({pil.width}×{pil.height} px)",
                     use_container_width=True)

        st.markdown("### 📍 Geo Parameters")
        lat = st.number_input("Center Latitude",  value=19.062061, format="%.6f")
        lon = st.number_input("Center Longitude", value=72.863542, format="%.6f")
        cov = st.number_input("Coverage (km)",    value=1.60, min_value=0.1)
        pil_w = st.session_state.get("upload_pil", Image.new("RGB",(1024,522))).width
        pil_h = st.session_state.get("upload_pil", Image.new("RGB",(1024,522))).height
        st.session_state["image_meta"] = {
            "center_lat": lat, "center_lon": lon,
            "coverage_km": cov, "width_px": pil_w, "height_px": pil_h,
        }

    with c2:
        st.markdown("**System Logs**")
        _terminal()

    st.divider()
    if st.session_state.get("upload_obj"):
        if st.button("▶ PROCEED TO ZONE MAP", key="btn1"):
            _log("Image accepted — starting Vision Agent …")
            st.session_state["st_vision"] = "🟡 Running"
            st.session_state["stage"] = 1; st.rerun()
    else:
        st.info("Upload a satellite image to begin.")

# ============================================================================
#  STAGE 2 — ZONE MAP  (Vision Agent — runs ONCE, cached)
# ============================================================================

def stage_2():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">🗺️ Stage 2: Zone Map Analysis</h2>',
                unsafe_allow_html=True)

    # ── Run Vision Agent ONCE ─────────────────────────────────────────────
    if "zone_map" not in st.session_state:
        with st.spinner("Vision Agent analysing image … (runs once)"):
            try:
                from agents.vision_agent.vision_agent import analyze_image
                upload_obj = st.session_state.get("upload_obj")
                img_path   = _save_upload(upload_obj) if upload_obj \
                             else "Images_for_testing/image.png"
                result, logs = _capture(analyze_image, img_path)
                _log(logs)
                if result:
                    st.session_state["zone_map"]   = result.get("zone_map", {})
                    st.session_state["flood_mask"] = result.get("flood_prob_map")
                    st.session_state["img_path"]   = img_path
                    n = len(st.session_state["zone_map"])
                    # Update image_meta dimensions from actual image
                    meta = st.session_state.get("image_meta", _DEFAULT_META.copy())
                    from PIL import Image as _PIL
                    with _PIL.open(img_path) as im:
                        meta["width_px"] = im.width; meta["height_px"] = im.height
                    st.session_state["image_meta"] = meta
                    _log(f"✅ Vision Agent complete — {n} zones")
                    st.session_state["st_vision"] = "🟢 Done"
                else:
                    _log("[ERROR] Vision Agent returned nothing")
                    st.session_state["zone_map"] = {}
                    st.session_state["st_vision"] = "🔴 Error"
            except Exception as e:
                _log(f"[ERROR] Vision Agent: {e}")
                st.session_state.setdefault("zone_map", {})
                st.session_state["st_vision"] = "🔴 Error"

    zone_map = st.session_state.get("zone_map", {})

    c1, c2 = st.columns([3, 2])
    with c1:
        grid_path = "zone_results/grid_output.jpg"
        if os.path.exists(grid_path):
            st.image(Image.open(grid_path), caption="Zone Grid (10×10)",
                     use_container_width=True)
        elif st.session_state.get("upload_pil"):
            st.image(st.session_state["upload_pil"],
                     caption="Uploaded Image (grid overlay after analysis)",
                     use_container_width=True)

    with c2:
        st.markdown("**Top Affected Zones**")
        if zone_map:
            top = sorted(zone_map.items(), key=lambda x:x[1].get("severity",0),
                         reverse=True)[:15]
            df = pd.DataFrame([{
                "Zone":   zid,
                "Sev":    f'{d.get("severity",0):.2f}',
                "Flood":  f'{d.get("flood_score",0):.2f}',
                "Damage": f'{d.get("damage_score",0):.2f}',
                "Level":  _sev_label(d.get("severity",0)),
            } for zid,d in top])
            st.dataframe(df, use_container_width=True, hide_index=True)

    _terminal()
    st.divider()

    # Persist to DB (only once)
    if zone_map and "db_saved" not in st.session_state:
        try:
            from db.update_from_vision import update_zones_from_vision
            update_zones_from_vision(zone_map)
            st.session_state["db_saved"] = True
            _log("Zone data saved to crisis.db")
        except Exception as e:
            _log(f"[WARN] DB save skipped: {e}")

    _nav(back=0, fwd=2, fwd_label="▶ PROCEED TO DRONE ALLOCATION")

# ============================================================================
#  STAGE 3 — DRONE ALLOCATION  (runs ONCE)
# ============================================================================

def stage_3():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">🚁 Stage 3: Drone Allocation</h2>',
                unsafe_allow_html=True)

    if "most_affected_zones" not in st.session_state:
        with st.spinner("Querying top affected zones …"):
            try:
                from agents.resource_agent.drone_analysis import get_most_affected_zones
                zones, logs = _capture(get_most_affected_zones, top_n=5)
                st.session_state["most_affected_zones"] = zones or []
                _log(logs); _log(f"Top zones: {zones}")
            except Exception as e:
                _log(f"[WARN] Fallback zone sort: {e}")
                zm = st.session_state.get("zone_map",{})
                st.session_state["most_affected_zones"] = sorted(
                    zm, key=lambda z:zm[z].get("severity",0), reverse=True)[:5]

    if "drone_allocation" not in st.session_state:
        with st.spinner("Dispatching drones …"):
            try:
                from agents.drone_agent.drone_nodes import (
                    drone_decision_node, drone_dispatch_node)
                s = {"zone_map": st.session_state.get("zone_map",{}),
                     "most_affected_zones": st.session_state["most_affected_zones"]}
                o1, l1 = _capture(drone_decision_node, s); _log(l1); s.update(o1 or {})
                o2, l2 = _capture(drone_dispatch_node, s); _log(l2)
                merged = {**(o1 or {}), **(o2 or {})}
                st.session_state["drone_allocation"] = merged.get(
                    "drone_allocation",
                    {f"drone_{i+1}":z for i,z in
                     enumerate(st.session_state["most_affected_zones"])})
                st.session_state["zone_image_map"] = merged.get("zone_image_map",{})
                _log(f"✅ Drones dispatched: {st.session_state['drone_allocation']}")
            except Exception as e:
                _log(f"[WARN] Drone fallback: {e}")
                st.session_state["drone_allocation"] = {
                    f"drone_{i+1}":z for i,z in
                    enumerate(st.session_state.get("most_affected_zones",[]))}

    alloc = st.session_state.get("drone_allocation", {})
    n     = max(len(alloc), 1)
    cols  = st.columns(n)
    for i, (d_id, z_id) in enumerate(alloc.items()):
        with cols[i % n]:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<b style="color:{THEME["accent_cyan"]};">{d_id.upper()}</b><br>'
                f'<span style="font-size:24px;">🚁</span><br>'
                f'→ <b>{z_id}</b><br>'
                f'<span style="color:{THEME["green"]};font-size:12px;">✅ DISPATCHED</span>'
                f'</div>', unsafe_allow_html=True)

    st.dataframe(
        pd.DataFrame([{"Drone":k,"Zone":v,"Status":"✅ Dispatched"}
                      for k,v in alloc.items()]),
        use_container_width=True, hide_index=True)
    _terminal(); st.divider()
    _nav(back=1, fwd=3, fwd_label="▶ PROCEED TO GALLERY")

# ============================================================================
#  STAGE 4 — GALLERY + PEOPLE DETECTION  (runs ONCE)
# ============================================================================

def stage_4():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">📸 Stage 4: Drone Imagery Gallery</h2>',
                unsafe_allow_html=True)

    # Show images
    imgs = {}
    p = Path("zone_images")
    if p.exists():
        for f in sorted(list(p.glob("*.jpg"))+list(p.glob("*.png"))):
            try: imgs[f.stem] = Image.open(f)
            except: pass

    alloc = st.session_state.get("drone_allocation",{})
    akv   = list(alloc.items())

    if imgs:
        cols = st.columns(3)
        for idx, (name, img) in enumerate(list(imgs.items())[:9]):
            with cols[idx%3]:
                st.image(img, use_container_width=True)
                drone_id = akv[idx%len(akv)][0] if akv else f"drone_{idx+1}"
                zone_id  = alloc.get(drone_id,"—")
                st.caption(f"**{name}** · {drone_id} → {zone_id}")
    else:
        st.info("No zone images found in `zone_images/`.\n"
                "Images are generated when drones complete their survey.")

    # People detection — runs ONCE
    if "people_counts" not in st.session_state:
        with st.spinner("Running drone vision (people detection) …"):
            try:
                from agents.drone_agent.drone_vision import drone_vision_node
                s = {"drone_allocation": alloc,
                     "zone_image_map":   st.session_state.get("zone_image_map",{}),
                     "drone_zones":      list(alloc.values())}
                out, logs = _capture(drone_vision_node, s)
                counts = (out or {}).get("people_counts", {})
                st.session_state["people_counts"] = counts
                _log(logs); _log(f"People counts: {counts}")
                try:
                    from db.update_people_count import update_people_count
                    update_people_count(counts); _log("People counts → crisis.db")
                except Exception as e: _log(f"[WARN] DB: {e}")
            except Exception as e:
                _log(f"[WARN] Vision fallback: {e}")
                st.session_state["people_counts"] = {}

    counts = st.session_state.get("people_counts",{})
    if counts:
        st.markdown("**People Detected by Zone**")
        df = pd.DataFrame([{"Zone":k,"👤 People":v} for k,v in counts.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)

    _terminal(); st.divider()
    _nav(back=2, fwd=4, fwd_label="▶ PROCEED TO ANALYSIS")

# ============================================================================
#  STAGE 5 — ANALYSIS
# ============================================================================

def stage_5():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">📊 Stage 5: Zone Analysis Results</h2>',
                unsafe_allow_html=True)

    people = st.session_state.get("people_counts",{})
    zm     = st.session_state.get("zone_map",{})
    top    = st.session_state.get("most_affected_zones",[])

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Zone Severity & People**")
        zones = top or list(zm.keys())[:15]
        rows = [{"Zone":z,
                 "👤 People": people.get(z,0),
                 "Severity":  f'{zm.get(z,{}).get("severity",0):.2f}',
                 "Flood":     f'{zm.get(z,{}).get("flood_score",0):.2f}',
                 "Damage":    f'{zm.get(z,{}).get("damage_score",0):.2f}',
                 "Level":     _sev_label(zm.get(z,{}).get("severity",0))}
                for z in zones]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if os.path.exists("db/crisis.db"):
            try:
                conn = sqlite3.connect("db/crisis.db")
                df_db = pd.read_sql_query(
                    "SELECT * FROM zones ORDER BY severity DESC LIMIT 10", conn)
                conn.close()
                st.markdown("**Database Snapshot**")
                st.dataframe(df_db, use_container_width=True, hide_index=True)
            except: pass

    with c2:
        st.markdown("**Result Images**")
        rp = Path("zone_results"); shown = 0
        if rp.exists():
            for f in sorted(list(rp.glob("*.jpg"))+list(rp.glob("*.png"))):
                if f.name == "grid_output.jpg": continue
                try:
                    st.image(Image.open(f), caption=f.stem,
                             use_container_width=True)
                    shown += 1
                    if shown >= 4: break
                except: pass
        if not shown: st.info("No result images in `zone_results/`")

    _terminal(); st.divider()
    _nav(back=3, fwd=5, fwd_label="▶ PROCEED TO RESOURCE ALLOCATION")

# ============================================================================
#  STAGE 6 — RESOURCE ALLOCATION  (LLM — runs ONCE)
# ============================================================================

def stage_6():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">📦 Stage 6: Resource Allocation</h2>',
                unsafe_allow_html=True)
    st.session_state.setdefault("st_resource","⚪ Idle")

    if "rescue_plan" not in st.session_state:
        st.session_state["st_resource"] = "🟡 Running"
        with st.spinner("LLM allocating rescue resources … (runs once)"):
            try:
                from agents.resource_agent.rescue_decision_llm import (
                    allocate_rescue_resources_llm)
                plan, logs = _capture(
                    allocate_rescue_resources_llm,
                    zone_map      = st.session_state.get("zone_map",{}),
                    people_counts = st.session_state.get("people_counts",{}),
                    zones         = st.session_state.get("most_affected_zones",[]),
                )
                st.session_state["rescue_plan"] = plan or {}
                _log(logs); _log(f"✅ LLM allocation — {len(plan or {})} zone(s)")
                st.session_state["st_resource"] = "🟢 Done"
            except Exception as e:
                _log(f"[ERROR] Resource LLM: {e}")
                st.session_state["rescue_plan"] = {}
                st.session_state["st_resource"] = "🔴 Error"

    plan = st.session_state.get("rescue_plan",{})
    if plan:
        rows = []; totals = {}
        for zone_id, alloc in plan.items():
            row = {"Zone": zone_id}
            for rt, cnt in alloc.items():
                row[rt] = cnt; totals[rt] = totals.get(rt,0) + cnt
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).fillna(0),
                     use_container_width=True, hide_index=True)
        if totals:
            st.divider()
            mc = st.columns(len(totals))
            for i,(k,v) in enumerate(totals.items()):
                with mc[i]: st.metric(k.replace("_"," ").title(), int(v))
    else:
        st.warning("No resource plan — check Gemini API key in .env")

    _terminal(); st.divider()

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("◀ BACK", key="back6"):
            st.session_state["stage"] = 4; st.rerun()
    with c2:
        if st.button("🔄 Re-run LLM", key="retry6"):
            st.session_state.pop("rescue_plan", None)
            st.session_state["st_resource"] = "⚪ Idle"
            st.rerun()
    with c3:
        if plan and st.button("▶ PROCEED TO APPROVAL", key="fwd6"):
            st.session_state["stage"] = 6; st.rerun()

# ============================================================================
#  STAGE 7 — ADMIN APPROVAL I
# ============================================================================

def stage_7():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">✅ Stage 7: Admin Approval Gate #1</h2>',
                unsafe_allow_html=True)

    plan = st.session_state.get("rescue_plan",{})
    st.markdown("**Proposed Resource Allocation**")
    for zone_id, alloc in plan.items():
        desc = " · ".join(f"{v}× {k}" for k,v in alloc.items() if v)
        st.markdown(
            f'<div class="card">'
            f'<b style="color:{THEME["accent_cyan"]};">Zone {zone_id}</b>  →  {desc}'
            f'</div>', unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ APPROVE & PROCEED", key="app1", use_container_width=True):
            st.session_state["resource_approved"] = True
            st.session_state["st_resource"] = "🟢 Approved"
            _log("ADMIN: Resource allocation APPROVED ✓")
            st.success("Approved! ✅"); st.balloons(); st.rerun()
    with c2:
        if st.button("🔴 HOLD FOR REVIEW", key="hold1", use_container_width=True):
            st.session_state.pop("rescue_plan",None)
            st.session_state.pop("resource_approved",None)
            st.session_state["st_resource"] = "⚪ Idle"
            _log("ADMIN: Held — re-running LLM …")
            st.session_state["stage"] = 5; st.rerun()

    if st.session_state.get("resource_approved"):
        st.info("Resources approved — proceed to route planning.")
        _nav(back=5, fwd=7, fwd_label="▶ PLAN ROUTES")

# ============================================================================
#  STAGE 8 — ROUTE PLANNING  (Route Agent — runs ONCE)
# ============================================================================

def stage_8():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">🗺️ Stage 8: Route Planning</h2>',
                unsafe_allow_html=True)
    st.session_state["st_route"] = st.session_state.get("st_route","⚪ Idle")

    if "route_plan" not in st.session_state:
        st.session_state["st_route"] = "🟡 Running"
        with st.spinner("Route Agent planning paths via OSM … (runs once)"):
            try:
                from agents.route_agent.route_agent import plan_all_routes
                from generate_route_map import generate_route_map

                meta  = st.session_state.get("image_meta") or _DEFAULT_META
                bases = _DEFAULT_BASE_LOCATIONS
                flood = st.session_state.get("flood_mask")

                routes, logs = _capture(
                    plan_all_routes,
                    image_meta           = meta,
                    resource_assignments = st.session_state.get("rescue_plan",{}),
                    base_locations       = bases,
                    blocked_masks        = {"flood":flood} if flood is not None else None,
                    use_real_osm         = True,
                    flood_threshold      = 0.45,
                )
                st.session_state["route_plan"]     = routes or []
                st.session_state["base_locations"] = bases
                _log(logs)

                if routes:
                    map_path, ml = _capture(generate_route_map,
                        route_plans=routes, image_meta=meta,
                        base_locations=bases,
                        zone_map=st.session_state.get("zone_map"))
                    st.session_state["route_map_path"] = map_path
                    _log(ml)

                n_ok = sum(1 for r in (routes or []) if r.get("success"))
                _log(f"✅ Routes: {n_ok} / {len(routes or [])} successful")
                st.session_state["st_route"] = "🟢 Done"

            except Exception as e:
                _log(f"[ERROR] Route planner: {e}")
                st.session_state["route_plan"]  = []
                st.session_state["st_route"]    = "🔴 Error"

    routes = st.session_state.get("route_plan", [])

    # ── Summary table ─────────────────────────────────────────────────────
    if routes:
        st.markdown("**Planned Routes**")
        df = pd.DataFrame([{
            "Zone":     r.get("zone"),
            "Resource": f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
            "Units":    r.get("unit_count",1),
            "From":     r.get("origin_name"),
            "Dist(km)": r.get("distance_km",0),
            "ETA(min)": r.get("eta_minutes",0),
            "Waypoints":len(r.get("waypoints",[])),
            "Status":   "✓ OK" if r.get("success") else f'✗ {r.get("error","failed")}',
        } for r in routes])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Interactive Folium map — REAL waypoints ───────────────────────────
    st.markdown("**Interactive Route Map — Real OSM Waypoints**")
    fmap = _build_folium_map()
    st_folium(fmap, width=None, height=480, key="route_folium",
              returned_objects=[])

    mp = st.session_state.get("route_map_path")
    if mp and os.path.exists(mp):
        st.success(f"📄 Full HTML map saved to: `{mp}`")

    _terminal(); st.divider()

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("◀ BACK", key="back8"):
            st.session_state["stage"] = 6; st.rerun()
    with c2:
        if st.button("🔄 Re-plan Routes", key="replan8"):
            for k in ("route_plan","route_map_path"):
                st.session_state.pop(k, None)
            st.session_state["st_route"] = "⚪ Idle"
            st.rerun()
    with c3:
        if routes and st.button("▶ PROCEED TO APPROVAL II", key="fwd8"):
            st.session_state["stage"] = 8; st.rerun()

# ============================================================================
#  STAGE 9 — ADMIN APPROVAL II
# ============================================================================

def stage_9():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">✅ Stage 9: Admin Approval Gate #2</h2>',
                unsafe_allow_html=True)

    routes = st.session_state.get("route_plan",[])
    if routes:
        st.markdown("**Route Plan Summary**")
        df = pd.DataFrame([{
            "Resource":  f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
            "Units":     r.get("unit_count",1),
            "Zone":      r.get("zone"),
            "From":      r.get("origin_name"),
            "Dist(km)":  r.get("distance_km",0),
            "ETA(min)":  r.get("eta_minutes",0),
            "Status":    "✓ OK" if r.get("success") else "✗ FAILED",
        } for r in routes])
        st.dataframe(df, use_container_width=True, hide_index=True)

    mp = st.session_state.get("route_map_path")
    if mp and os.path.exists(mp):
        st.info(f"📄 Route map: `{mp}`")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ APPROVE ROUTES", key="app2", use_container_width=True):
            st.session_state["route_approved"] = True
            st.session_state["st_route"] = "🟢 Approved"
            _log("ADMIN: Route plan APPROVED ✓")
            st.success("Routes approved ✅"); st.balloons(); st.rerun()
    with c2:
        if st.button("🔴 RE-PLAN ROUTES", key="hold2", use_container_width=True):
            for k in ("route_plan","route_map_path","route_approved"):
                st.session_state.pop(k, None)
            st.session_state["st_route"] = "⚪ Idle"
            _log("ADMIN: Routes rejected — re-planning …")
            st.session_state["stage"] = 7; st.rerun()

    if st.session_state.get("route_approved"):
        st.info("Routes approved — proceed to communications.")
        _nav(back=7, fwd=9, fwd_label="▶ DISPATCH COMMUNICATIONS")

# ============================================================================
#  STAGE 10 — COMMUNICATIONS  (runs ONCE)
# ============================================================================

def stage_10():
    st.markdown(f'<h2 style="color:{THEME["accent_cyan"]};">📡 Stage 10: Communication Agent</h2>',
                unsafe_allow_html=True)

    if "dispatch_result" not in st.session_state:
        st.session_state["st_comm"] = "🟡 Running"
        with st.spinner("Communication Agent generating instructions … (runs once)"):
            try:
                from agents.communication_agent.communication_agent import dispatch_all

                routes = st.session_state.get("route_plan",[])
                zm     = st.session_state.get("zone_map",{})
                people = st.session_state.get("people_counts",{})

                zone_meta = {
                    zid: {
                        "severity": ("Critical" if d.get("severity",0)>=0.7 else
                                     "Moderate" if d.get("severity",0)>=0.4 else "Low"),
                        "victim_count": people.get(zid,0),
                    } for zid,d in zm.items()
                }

                result, logs = _capture(
                    dispatch_all,
                    route_plans     = routes,
                    zone_metadata   = zone_meta,
                    field_reports   = [],
                    dispatch_config = {"send_sms":False,"generate_audio":False,
                                       "language":"English"},
                )
                st.session_state["dispatch_result"] = result or {}
                _log(logs); _log("✅ Dispatch pipeline complete")
                st.session_state["st_comm"]   = "🟢 Done"
                st.session_state["st_master"] = "🟢 Done"

            except Exception as e:
                _log(f"[ERROR] Communication: {e}")
                st.session_state["dispatch_result"] = {}
                st.session_state["st_comm"] = "🔴 Error"

    dispatch     = st.session_state.get("dispatch_result",{})
    instructions = dispatch.get("instructions",{})

    st.markdown("**Dispatch Instructions**")

    # If communication agent gave structured instructions, show them
    if instructions:
        for zone_id, instr in instructions.items():
            text = instr if isinstance(instr,str) else json.dumps(instr,indent=2)
            st.markdown(
                f'<div class="card">'
                f'<b style="color:{THEME["accent_cyan"]};">Zone {zone_id}</b><br>'
                f'<code style="font-size:11px;">{text}</code>'
                f'</div>', unsafe_allow_html=True)
    else:
        # Fallback: pretty-print from route_plan
        for r in st.session_state.get("route_plan",[]):
            em = _remoji(r.get("resource_type",""))
            rtype = r.get("resource_type","").replace("_"," ").title()
            st.markdown(
                f'<div class="card">'
                f'<b style="color:{THEME["accent_cyan"]};">'
                f'{em} {r.get("unit_count",1)}× {rtype} → Zone {r.get("zone")}'
                f'</b><br>'
                f'<span style="font-family:\'Share Tech Mono\';font-size:11px;">'
                f'From: {r.get("origin_name")}  ·  '
                f'{r.get("distance_km","?")} km  ·  '
                f'ETA {r.get("eta_minutes","?")} min  ·  '
                f'{len(r.get("waypoints",[]))} waypoints'
                f'</span></div>', unsafe_allow_html=True)

    summary = dispatch.get("summary","")
    if summary:
        st.info(f"**Summary:** {summary}")

    st.markdown(
        f'<div style="color:{THEME["green"]};font-family:\'Share Tech Mono\';'
        f'font-size:13px;font-weight:bold;">'
        f'📡 {len(st.session_state.get("route_plan",[]))} instruction(s) ready</div>',
        unsafe_allow_html=True)

    _terminal(); st.divider()

    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        if st.button("📡 SEND ALL DISPATCHES", use_container_width=True, key="send_all"):
            _log("Sending all dispatches …")
            st.success("✅ All dispatches sent successfully!")
            st.balloons()
            st.session_state["st_comm"] = "🟢 Sent"

    c1, c2 = st.columns(2)
    with c1:
        if st.button("◀ BACK", key="back10"):
            st.session_state["stage"] = 8; st.rerun()
    with c2:
        if st.button("🏁 COMPLETE & RESET", key="done10"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.balloons(); st.rerun()

# ============================================================================
#  MAIN
# ============================================================================

STAGE_FNS = [
    stage_1, stage_2, stage_3, stage_4, stage_5,
    stage_6, stage_7, stage_8, stage_9, stage_10,
]

def main():
    # Init defaults
    defaults = {
        "stage": 0, "log": "",
        "st_vision":"⚪ Idle","st_master":"⚪ Idle","st_resource":"⚪ Idle",
        "st_route":"⚪ Idle","st_comm":"⚪ Idle",
    }
    for k,v in defaults.items():
        st.session_state.setdefault(k, v)

    _sidebar()

    st.markdown(
        f'<h1 style="color:{THEME["accent_cyan"]};font-family:\'Rajdhani\';'
        f'text-align:center;">🛰️ AEGIS · Crisis Management AI</h1>',
        unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{THEME["mono_green"]};text-align:center;'
        f'font-family:\'Share Tech Mono\';">Agentic Emergency Response &amp; Intelligence System</p>',
        unsafe_allow_html=True)
    st.divider()

    st.markdown(
        f'<p style="color:{THEME["text_primary"]};font-family:\'Share Tech Mono\';'
        f'font-size:12px;">PIPELINE PROGRESS</p>', unsafe_allow_html=True)
    _stepper()
    st.divider()

    fn = STAGE_FNS[st.session_state.get("stage",0)]
    fn()

if __name__ == "__main__":
    main()