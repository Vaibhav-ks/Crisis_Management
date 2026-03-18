"""
generate_route_map.py
---------------------
Generates a Folium HTML route map from LIVE pipeline data immediately after
the admin approves the route plan.

Called by master_nodes.py → route_planner_node() after plan_all_routes().

Outputs:
    zone_results/route_map_<YYYYMMDD_HHMMSS>.html   — timestamped copy
    zone_results/route_map_latest.html               — always latest run

The Streamlit frontend reads route_map_latest.html to display the map.
"""

import os
import math
import shutil
import datetime

import folium


# ── Color / icon config ───────────────────────────────────────────────────────

ROUTE_COLORS = {
    "ambulance":   "#e74c3c",
    "rescue_team": "#2980b9",
    "boat":        "#16a085",
    "helicopter":  "#8e44ad",
    "fire_truck":  "#e67e22",
    "truck":       "#7f8c8d",
}

RESOURCE_ICONS = {
    "ambulance":   "🚑",
    "rescue_team": "🚒",
    "boat":        "🚤",
    "helicopter":  "🚁",
    "fire_truck":  "🚒",
    "truck":       "🚛",
}

OUTPUT_DIR = "zone_results"


def _color(resource_type: str) -> str:
    key = resource_type.lower().rstrip("s")
    return ROUTE_COLORS.get(key, ROUTE_COLORS.get(resource_type.lower(), "#888888"))


def _icon(resource_type: str) -> str:
    key = resource_type.lower().rstrip("s")
    return RESOURCE_ICONS.get(key, RESOURCE_ICONS.get(resource_type.lower(), "🚗"))


def _zone_center(zone_name: str, image_meta: dict) -> tuple:
    """Return (lat, lon) for a 0-based zone name given image_meta."""
    center_lat  = image_meta["center_lat"]
    center_lon  = image_meta["center_lon"]
    coverage_km = image_meta["coverage_km"]
    w_px        = image_meta["width_px"]
    h_px        = image_meta["height_px"]

    deg_lat = coverage_km / 111.0
    deg_lon = coverage_km / (111.0 * math.cos(math.radians(center_lat)))
    aspect  = h_px / w_px
    tl_lat  = center_lat + deg_lat * aspect / 2
    tl_lon  = center_lon - deg_lon / 2
    dlat    = deg_lat * aspect / h_px
    dlon    = deg_lon / w_px

    body = zone_name.strip().upper().lstrip("Z")
    if "_" in body:
        r, c = int(body.split("_")[0]), int(body.split("_")[1])
    else:
        r = int(body[0])
        c = int(body[1:]) if len(body) > 1 else 0

    px  = c * (w_px / 10) + (w_px / 10) / 2
    py  = r * (h_px / 10) + (h_px / 10) / 2
    lat = tl_lat - py * dlat
    lon = tl_lon + px * dlon
    return round(lat, 5), round(lon, 5)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_route_map(
    route_plans:    list,
    image_meta:     dict,
    base_locations: dict = None,
    zone_map:       dict = None,
) -> str:
    """
    Generate a Folium HTML map from the live route_plan output of plan_all_routes().

    Parameters
    ----------
    route_plans    : list of route dicts from plan_all_routes()
                     Required keys per dict: zone, resource_type, origin_name,
                     destination_latlon, success, waypoints, distance_km,
                     eta_minutes, unit_count
    image_meta     : dict — center_lat, center_lon, coverage_km, width_px, height_px
    base_locations : dict mapping singular resource type → {name, lat, lon}
    zone_map       : optional Vision Agent output for severity colouring

    Returns
    -------
    str — absolute path to the saved HTML file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    center_lat  = image_meta["center_lat"]
    center_lon  = image_meta["center_lon"]
    coverage_km = image_meta["coverage_km"]
    w_px        = image_meta["width_px"]
    h_px        = image_meta["height_px"]

    fmap = folium.Map(location=[center_lat, center_lon],
                      zoom_start=14, tiles="OpenStreetMap")

    # ── Zone grid ──────────────────────────────────────────────────────────
    deg_lat   = coverage_km / 111.0
    deg_lon   = coverage_km / (111.0 * math.cos(math.radians(center_lat)))
    aspect    = h_px / w_px
    cell_dlat = (deg_lat * aspect) / 10
    cell_dlon = deg_lon / 10

    crisis_zones = {r["zone"] for r in route_plans}

    for row in range(10):
        for col in range(10):
            name         = f"Z{row}{col}"
            clat, clon   = _zone_center(name, image_meta)
            is_crisis    = name in crisis_zones
            fill_color   = "#e74c3c" if is_crisis else "#ffffff"

            if zone_map and name in zone_map:
                sev = zone_map[name].get("severity", 0)
                if sev >= 0.7:
                    fill_color = "#e74c3c"
                elif sev >= 0.4:
                    fill_color = "#f39c12"
                elif sev > 0.0:
                    fill_color = "#f1c40f"

            tip = name
            if is_crisis:
                descs = ", ".join(
                    f"{r.get('unit_count', 1)}× {r['resource_type']}"
                    for r in route_plans if r["zone"] == name and r.get("success")
                )
                tip = f"🚨 {name} | {descs}" if descs else f"🚨 {name}"

            folium.Rectangle(
                bounds=[[clat - cell_dlat/2, clon - cell_dlon/2],
                        [clat + cell_dlat/2, clon + cell_dlon/2]],
                color="#444", weight=0.5, fill=True,
                fill_color=fill_color,
                fill_opacity=0.40 if is_crisis else 0.04,
                tooltip=tip,
            ).add_to(fmap)

    # ── Crisis zone markers ────────────────────────────────────────────────
    for zone_name in crisis_zones:
        clat, clon = _zone_center(zone_name, image_meta)
        desc = ", ".join(
            f"{r.get('unit_count', 1)}× {r['resource_type']}"
            for r in route_plans if r["zone"] == zone_name and r.get("success")
        )
        folium.Marker(
            [clat, clon],
            tooltip=f"🚨 {zone_name} | {desc}",
            icon=folium.Icon(color="red", icon="exclamation-sign"),
        ).add_to(fmap)

    # ── Base location markers ──────────────────────────────────────────────
    ICON_MAP = {
        "ambulance":   ("red",      "plus-sign"),
        "rescue_team": ("blue",     "home"),
        "boat":        ("darkblue", "tint"),
        "helicopter":  ("purple",   "plane"),
        "fire_truck":  ("orange",   "fire"),
    }
    if base_locations:
        seen_origins = set()
        for r in route_plans:
            if not r.get("success"):
                continue
            rkey = r["resource_type"].lower().rstrip("s")
            base = base_locations.get(rkey) or base_locations.get(r["resource_type"])
            if base and r["origin_name"] not in seen_origins:
                ic_c, ic_i = ICON_MAP.get(rkey, ("gray", "info-sign"))
                folium.Marker(
                    [base["lat"], base["lon"]],
                    tooltip=f"📍 {base['name']}",
                    popup=folium.Popup(
                        f"<b>{base['name']}</b><br>Deploys: {rkey}", max_width=220
                    ),
                    icon=folium.Icon(color=ic_c, icon=ic_i),
                ).add_to(fmap)
                seen_origins.add(r["origin_name"])

    # ── Route lines ────────────────────────────────────────────────────────
    for r in route_plans:
        if not r.get("success"):
            continue

        color = _color(r["resource_type"])
        icon  = _icon(r["resource_type"])
        wpts  = r.get("waypoints", [])

        if len(wpts) < 2:
            # Draw straight line from origin to destination
            dest = r.get("destination_latlon")
            rkey = r["resource_type"].lower().rstrip("s")
            base = (base_locations or {}).get(rkey)
            if base and dest:
                wpts = [(base["lat"], base["lon"]), dest]
            elif len(wpts) == 1 and dest:
                wpts = [wpts[0], dest]
            else:
                continue

        folium.PolyLine(
            wpts, color=color, weight=6, opacity=0.92,
            tooltip=(
                f"{icon} {r.get('unit_count', 1)}× {r['resource_type']}\n"
                f"Zone: {r['zone']}  |  From: {r['origin_name']}\n"
                f"Distance: {r['distance_km']} km  |  ETA: {r['eta_minutes']} min\n"
                f"Waypoints: {len(r.get('waypoints', []))}"
            ),
        ).add_to(fmap)

        for lat, lon in wpts[1:-1]:
            folium.CircleMarker([lat, lon], radius=4, color=color,
                                fill=True, fill_color=color,
                                fill_opacity=0.8).add_to(fmap)

        folium.Marker(
            list(wpts[-1]),
            tooltip=f"{icon} {r['resource_type']} → {r['zone']}",
            icon=folium.DivIcon(
                html=f'<div style="font-size:18px;color:{color};">▼</div>',
                icon_size=(20, 20), icon_anchor=(10, 10),
            ),
        ).add_to(fmap)

    # ── Legend ─────────────────────────────────────────────────────────────
    ts_label   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seen_types = {r["resource_type"] for r in route_plans if r.get("success")}
    legend_items = ""
    for rtype in seen_types:
        c = _color(rtype)
        i = _icon(rtype)
        legend_items += (
            f'<span style="color:{c};font-size:18px;">━━━</span> '
            f'{i} {rtype.replace("_"," ").title()}<br>'
        )

    summary_rows = ""
    for r in route_plans:
        ico = _icon(r["resource_type"])
        if r.get("success"):
            status = f"{r['distance_km']} km / {r['eta_minutes']} min"
        else:
            status = f"❌ {r.get('error', 'failed')}"
        summary_rows += (
            f"<tr>"
            f"<td>{ico} {r.get('unit_count',1)}× {r['resource_type']}</td>"
            f"<td><b>{r['zone']}</b></td>"
            f"<td>{r['origin_name']}</td>"
            f"<td>{status}</td>"
            f"<td>{len(r.get('waypoints',[]))} pts</td>"
            f"</tr>"
        )

    n_ok = len([r for r in route_plans if r.get("success")])
    overlay = f"""
    <div style="position:fixed;top:15px;right:15px;z-index:9999;
                background:white;padding:14px 18px;border-radius:10px;
                border:2px solid #333;font-family:Arial;font-size:13px;
                box-shadow:4px 4px 10px rgba(0,0,0,0.3);max-width:270px;">
      <b style="font-size:15px;">🗺 Route Agent — Live Map</b>
      <div style="font-size:11px;color:#888;margin-bottom:8px;">Generated: {ts_label}</div>
      {legend_items}
      <span style="background:#e74c3c;padding:0 6px;border-radius:3px;color:white;">■</span> Crisis Zone<br>
      <hr style="margin:8px 0;">
      <div style="font-size:11px;color:#555;">Routes from real OSM waypoints.<br>
      Click route lines for details.</div>
    </div>

    <div style="position:fixed;bottom:30px;right:15px;z-index:9999;
                background:white;padding:10px 14px;border-radius:10px;
                border:2px solid #333;font-family:Arial;font-size:12px;
                box-shadow:4px 4px 10px rgba(0,0,0,0.3);
                max-width:750px;overflow-x:auto;">
      <b>📋 Deployment Summary ({n_ok} successful routes)</b><br>
      <table style="border-collapse:collapse;margin-top:6px;font-size:12px;">
        <tr style="background:#f0f0f0;">
          <th style="padding:4px 8px;border:1px solid #ccc;">Resource</th>
          <th style="padding:4px 8px;border:1px solid #ccc;">Zone</th>
          <th style="padding:4px 8px;border:1px solid #ccc;">From</th>
          <th style="padding:4px 8px;border:1px solid #ccc;">ETA</th>
          <th style="padding:4px 8px;border:1px solid #ccc;">Waypoints</th>
        </tr>
        {summary_rows}
      </table>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(overlay))

    # ── Save ───────────────────────────────────────────────────────────────
    ts_file   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path   = os.path.join(OUTPUT_DIR, f"route_map_{ts_file}.html")
    latest    = os.path.join(OUTPUT_DIR, "route_map_latest.html")

    fmap.save(ts_path)
    shutil.copy2(ts_path, latest)

    print(f"[RouteMap] Saved  → {os.path.abspath(ts_path)}")
    print(f"[RouteMap] Latest → {os.path.abspath(latest)}")
    return os.path.abspath(latest)