"""
visualize_routes.py
-------------------
Standalone route visualiser for the Prayagraj flood scenario.

Run:
    python visualize_routes.py

Output:
    route_map.html  — opens in browser automatically

Also importable from generate_route_map.py or tests.
"""

import os
import sys
import math
import webbrowser

import networkx as nx
import folium
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO — Prayagraj flood near the Yamuna river
#
#  Zone names use 0-based indexing matching Vision Agent's grid_mapper.py:
#      Z00 = top-left, Z99 = bottom-right
#      "Z78" → row=7, col=8  (NOT row=7-1=6, col=8-1=7)
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_META = {
    "center_lat":  25.435,
    "center_lon":  81.846,
    "coverage_km": 6,
    "width_px":    640,
    "height_px":   640,
}

# Zone names: 0-based row/col.  Resource keys: SINGULAR (matches route_agent.py)
RESOURCE_ASSIGNMENTS = {
    "Z23": {"ambulance": 2, "rescue_team": 1, "boat": 0},
    "Z45": {"ambulance": 1, "rescue_team": 2, "boat": 0},
    "Z78": {"ambulance": 0, "rescue_team": 1, "boat": 2},
    "Z89": {"ambulance": 0, "rescue_team": 0, "boat": 3},
}

# Keys are SINGULAR to match base_locations lookup in route_agent.py
BASE_LOCATIONS = {
    "ambulance": {
        "name":  "Motilal Nehru Medical College",
        "lat":   25.4510, "lon": 81.8340,
        "icon":  "plus-sign", "color": "red",
    },
    "rescue_team": {
        "name":  "NDRF Station, Civil Lines",
        "lat":   25.4590, "lon": 81.8420,
        "icon":  "home", "color": "blue",
    },
    "boat": {
        "name":  "Sangam Boat Launch (Naini Ghat)",
        "lat":   25.4300, "lon": 81.8700,
        "icon":  "tint", "color": "darkblue",
    },
}

ROUTE_COLORS = {
    "ambulance":   "#e74c3c",
    "rescue_team": "#2980b9",
    "boat":        "#16a085",
}

RESOURCE_ICONS = {
    "ambulance":   "🚑",
    "rescue_team": "🚒",
    "boat":        "🚤",
}


# ─────────────────────────────────────────────────────────────────────────────
#  REALISTIC PRAYAGRAJ ROAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_prayagraj_graph():
    """
    Build a realistic road graph for central Prayagraj.
    Fallback when OSMnx has no internet access.
    """
    G = nx.MultiDiGraph()

    nodes = {
        "A": (25.460, 81.826, "Leader Road / Civil Lines W"),
        "B": (25.460, 81.840, "MG Marg / Civil Lines Centre"),
        "C": (25.460, 81.856, "Kamla Nehru Road / Civil Lines E"),
        "D": (25.450, 81.826, "Howrah-Delhi Railway / West"),
        "E": (25.450, 81.840, "Allahabad City Station Road"),
        "F": (25.450, 81.856, "MG Marg East"),
        "G": (25.450, 81.870, "George Town / NH30"),
        "H": (25.440, 81.826, "Zero Road West"),
        "I": (25.440, 81.840, "Zero Road Centre / Chowk"),
        "J": (25.440, 81.856, "GT Road / Allahabad City"),
        "K": (25.440, 81.870, "GT Road East / Naini Bridge approach"),
        "L": (25.430, 81.826, "Atarsuiya Road"),
        "M": (25.430, 81.840, "Kydganj Road Centre"),
        "N": (25.430, 81.856, "Kydganj East"),
        "O": (25.430, 81.870, "Naini Bridge Road"),
        "P": (25.420, 81.826, "Gaughat Road (FLOOD ZONE)"),
        "Q": (25.420, 81.840, "Naini Bridge South (FLOOD ZONE)"),
        "R": (25.420, 81.870, "Yamuna Bank / Naini (FLOOD ZONE)"),
    }

    for nid, (lat, lon, label) in nodes.items():
        G.add_node(nid, y=lat, x=lon, label=label)

    roads = [
        ("A","B","Leader Road",40), ("B","C","Kamla Nehru Road",40),
        ("D","E","Subhash Chowk Rd",30), ("E","F","MG Marg",50),
        ("F","G","MG Marg East",50), ("H","I","Zero Road",40),
        ("I","J","Grand Trunk Road",50), ("J","K","GT Road East",50),
        ("L","M","Kydganj Road",35), ("M","N","Kydganj East",35),
        ("N","O","Naini Bridge Rd",40), ("P","Q","Gaughat Road",20),
        ("Q","R","Yamuna Bank Road",20),
        ("A","D","Noorullah Road",35), ("D","H","Tilak Road",35),
        ("H","L","Atarsuiya Road",35), ("L","P","60 Feet Road",25),
        ("B","E","Phanrela Road",40), ("E","I","Allahabad Stn Rd",40),
        ("I","M","Sadar Road",40), ("M","Q","Sangam Marg",25),
        ("C","F","Hewett Road",40), ("F","J","Klopibagh Flyover",55),
        ("J","N","Naini Bridge Rd N",40), ("N","R","Naini Bridge",30),
        ("G","K","NH30 / NH319D",60), ("K","O","Yamuna Expressway",55),
        ("O","R","Naini Road S",30),
    ]

    for u, v, road_name, speed in roads:
        if u not in nodes or v not in nodes:
            continue
        lat_u, lon_u, _ = nodes[u]
        lat_v, lon_v, _ = nodes[v]
        dist_m = math.sqrt(
            ((lat_v - lat_u) * 111000) ** 2
            + ((lon_v - lon_u) * 111000 * math.cos(math.radians(lat_u))) ** 2
        )
        tt    = dist_m / (speed * 1000 / 3600)
        attrs = dict(length=dist_m, speed_kph=speed, travel_time=tt,
                     road_name=road_name, blocked=False)
        G.add_edge(u, v, key=0, **attrs)
        G.add_edge(v, u, key=0, **attrs)

    return G, nodes


def apply_flood_blockages(G, flood_polygon):
    blocked = []
    for u, v, key, data in G.edges(keys=True, data=True):
        ud, vd = G.nodes[u], G.nodes[v]
        mid = Point((ud["x"] + vd["x"]) / 2, (ud["y"] + vd["y"]) / 2)
        if flood_polygon.contains(mid):
            G[u][v][key]["travel_time"] = 1e9
            G[u][v][key]["blocked"]     = True
            rn = data.get("road_name", f"{u}-{v}")
            if rn not in blocked:
                blocked.append(rn)
    return blocked


def nearest_node_graph(G, lat, lon):
    best, best_d = None, float("inf")
    for nid, d in G.nodes(data=True):
        dist = math.sqrt((d["y"] - lat) ** 2 + (d["x"] - lon) ** 2)
        if dist < best_d:
            best_d, best = dist, nid
    return best


def dijkstra_route(G, origin, dest):
    try:
        path = nx.dijkstra_path(G, origin, dest, weight="travel_time")
        dist = tt = 0.0
        road_names = []
        for i in range(len(path) - 1):
            e = min(G[path[i]][path[i+1]].values(),
                    key=lambda d: d.get("travel_time", 1e9))
            dist += e.get("length", 0)
            tt   += e.get("travel_time", 0)
            rn    = e.get("road_name", "")
            if rn and (not road_names or road_names[-1] != rn):
                road_names.append(rn)
        return {
            "success": True,
            "waypoints": [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path],
            "distance_km": round(dist / 1000, 2),
            "eta_min":     round(tt / 60, 1),
            "road_names":  road_names,
        }
    except nx.NetworkXNoPath:
        return {"success": False, "error": "No path — destination surrounded by blocked roads"}


# ─────────────────────────────────────────────────────────────────────────────
#  GEO HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_transform(center_lat, center_lon, coverage_km, w, h):
    deg_lat = coverage_km / 111.0
    deg_lon = coverage_km / (111.0 * math.cos(math.radians(center_lat)))
    aspect  = h / w
    return {
        "tl_lat": center_lat + deg_lat * aspect / 2,
        "tl_lon": center_lon - deg_lon / 2,
        "dlat":   deg_lat * aspect / h,
        "dlon":   deg_lon / w,
        "w": w, "h": h,
    }


def zone_latlon_0based(zone_name, t, rows=10, cols=10):
    """
    Convert a 0-based zone name to (lat, lon).
    'Z35' → row=3, col=5 (0-based, matches Vision Agent grid_mapper.py).
    """
    body = zone_name.strip().upper().lstrip("Z")
    if "_" in body:
        r, c = int(body.split("_")[0]), int(body.split("_")[1])
    else:
        r = int(body[0])
        c = int(body[1:]) if len(body) > 1 else 0

    cell_h = t["h"] / rows
    cell_w = t["w"] / cols
    # 0-based: multiply directly (no subtract 1)
    px  = c * cell_w + cell_w / 2
    py  = r * cell_h + cell_h / 2
    lat = t["tl_lat"] - py * t["dlat"]
    lon = t["tl_lon"] + px * t["dlon"]
    return round(lat, 5), round(lon, 5)


# ─────────────────────────────────────────────────────────────────────────────
#  MAP BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_map(
    resource_assignments=None,
    base_locations=None,
    image_meta=None,
    output_path=None,
    open_browser=True,
):
    """
    Build and save the Folium route map.

    Parameters
    ----------
    resource_assignments : dict of zone → {singular_resource_type: count}
    base_locations       : dict of singular resource type → location dict
    image_meta           : dict with center_lat/lon, coverage_km, width_px, height_px
    output_path          : where to save HTML (default: route_map.html next to this file)
    open_browser         : open the file in browser after saving

    Returns
    -------
    str — absolute path to saved HTML
    """
    if resource_assignments is None:
        resource_assignments = RESOURCE_ASSIGNMENTS
    if base_locations is None:
        base_locations = BASE_LOCATIONS
    if image_meta is None:
        image_meta = IMAGE_META
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "route_map.html")

    print("\n" + "=" * 60)
    print("  ROUTE MAP — Prayagraj Flood Scenario")
    print("=" * 60)

    t = build_transform(
        image_meta["center_lat"], image_meta["center_lon"],
        image_meta["coverage_km"],
        image_meta["width_px"], image_meta["height_px"],
    )

    # ── Road graph: try OSMnx, fall back to hand-crafted ─────────────────
    G, nodes_dict, use_osm = None, None, False
    try:
        import osmnx as ox
        print("[Map] Downloading real OSM road network ...")
        G_osm = ox.graph_from_point(
            (image_meta["center_lat"], image_meta["center_lon"]),
            dist=3500, network_type="drive",
        )
        G_osm  = ox.add_edge_speeds(G_osm)
        G_osm  = ox.add_edge_travel_times(G_osm)
        G      = G_osm
        use_osm = True
        print(f"[Map] OSM graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    except Exception as e:
        print(f"[Map] OSMnx unavailable ({type(e).__name__}) — using Prayagraj graph.")
        G, nodes_dict = build_prayagraj_graph()
        print(f"[Map] Prayagraj graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # ── Flood zone: Yamuna riverbank south of Kydganj ─────────────────────
    flood_coords  = [(81.810, 25.426), (81.900, 25.426),
                     (81.900, 25.415), (81.810, 25.415)]
    flood_shapely = Polygon(flood_coords)
    blocked_roads = apply_flood_blockages(G, flood_shapely)
    print(f"[Map] Blocked roads: {blocked_roads}")

    if use_osm:
        def get_nearest(lat, lon):
            try:
                return ox.distance.nearest_nodes(G, X=lon, Y=lat)
            except Exception:
                return nearest_node_graph(G, lat, lon)
    else:
        def get_nearest(lat, lon):
            return nearest_node_graph(G, lat, lon)

    # Boats use an unblocked copy (they sail through floodwater)
    G_boat, _ = build_prayagraj_graph() if not use_osm else (G, None)

    # ── Plan routes ───────────────────────────────────────────────────────
    all_routes = []
    for zone_name, assignments in resource_assignments.items():
        dest_lat, dest_lon = zone_latlon_0based(zone_name, t)
        print(f"\n[Route] Zone {zone_name} → ({dest_lat}, {dest_lon})")

        for rtype, count in assignments.items():
            if not count:
                continue
            lookup = rtype.lower().strip()
            if lookup not in base_locations:
                lookup = lookup.rstrip("s")
            if lookup not in base_locations:
                print(f"  [WARN] No base location for '{rtype}' — skipping")
                continue

            routing_G   = G_boat if "boat" in lookup else G
            base        = base_locations[lookup]
            origin_node = nearest_node_graph(routing_G, base["lat"], base["lon"])
            dest_node   = nearest_node_graph(routing_G, dest_lat, dest_lon)
            result      = dijkstra_route(routing_G, origin_node, dest_node)

            all_routes.append({
                "zone": zone_name, "resource_type": rtype,
                "unit_count": count, "origin_name": base["name"],
                "dest_latlon": (dest_lat, dest_lon),
                **result,
            })

            if result["success"]:
                via = " → ".join(result["road_names"])
                print(f"  ✓ {count}× {rtype}: {result['distance_km']} km  "
                      f"{result['eta_min']} min  via {via}")
            else:
                print(f"  ✗ {rtype}: {result['error']}")

    # ── Build Folium map ──────────────────────────────────────────────────
    fmap = folium.Map(
        location=[image_meta["center_lat"], image_meta["center_lon"]],
        zoom_start=14, tiles="OpenStreetMap",
    )

    cell_dlat    = t["dlat"] * (t["h"] / 10)
    cell_dlon    = t["dlon"] * (t["w"] / 10)
    crisis_zones = set(resource_assignments.keys())

    for row in range(10):
        for col in range(10):
            name      = f"Z{row}{col}"
            clat, clon = zone_latlon_0based(name, t)
            is_crisis  = name in crisis_zones
            folium.Rectangle(
                bounds=[[clat - cell_dlat/2, clon - cell_dlon/2],
                        [clat + cell_dlat/2, clon + cell_dlon/2]],
                color="#444", weight=0.5, fill=True,
                fill_color="#e74c3c" if is_crisis else "#ffffff",
                fill_opacity=0.35 if is_crisis else 0.03,
                tooltip=f"{name} {'🚨 CRISIS' if is_crisis else ''}",
            ).add_to(fmap)

    for zone_name in crisis_zones:
        clat, clon = zone_latlon_0based(zone_name, t)
        assgn = resource_assignments[zone_name]
        desc  = ", ".join(f"{v}× {k}" for k, v in assgn.items() if v > 0)
        folium.Marker(
            [clat, clon],
            tooltip=f"🚨 {zone_name} | {desc}",
            icon=folium.Icon(color="red", icon="exclamation-sign"),
        ).add_to(fmap)

    folium.Polygon(
        locations=[[lat, lon] for lon, lat in flood_coords],
        color="#1a6fa8", weight=2,
        fill=True, fill_color="#3498db", fill_opacity=0.45,
        tooltip="🌊 Yamuna Flood Zone — roads blocked",
        dash_array="6 3",
    ).add_to(fmap)
    folium.Marker(
        [25.420, 81.845],
        tooltip=f"🌊 FLOODED — blocked: {', '.join(blocked_roads[:4])}",
        icon=folium.Icon(color="lightblue", icon="tint"),
    ).add_to(fmap)

    if not use_osm:
        drawn = set()
        for u, v, data in G.edges(data=True):
            key = tuple(sorted([str(u), str(v)]))
            if key in drawn:
                continue
            drawn.add(key)
            ud, vd     = G.nodes[u], G.nodes[v]
            is_blocked = data.get("travel_time", 0) > 1e6
            folium.PolyLine(
                [[ud["y"], ud["x"]], [vd["y"], vd["x"]]],
                color="#cc2200" if is_blocked else "#666666",
                weight=3 if is_blocked else 2,
                opacity=0.85 if is_blocked else 0.5,
                dash_array="8 4" if is_blocked else None,
                tooltip=f"{'🚫 BLOCKED — ' if is_blocked else ''}"
                        f"{data.get('road_name', 'road')}",
            ).add_to(fmap)
        for nid, d in G.nodes(data=True):
            folium.CircleMarker(
                [d["y"], d["x"]], radius=5, color="#333",
                fill=True, fill_color="#fff", fill_opacity=1,
                tooltip=f"{nid}: {d.get('label','')}",
            ).add_to(fmap)

    for rtype, base in base_locations.items():
        folium.Marker(
            [base["lat"], base["lon"]],
            tooltip=f"📍 {base['name']}",
            popup=folium.Popup(f"<b>{base['name']}</b><br>Deploys: {rtype}", max_width=220),
            icon=folium.Icon(color=base.get("color","gray"),
                             icon=base.get("icon","info-sign")),
        ).add_to(fmap)

    for r in all_routes:
        if not r["success"]:
            continue
        color = ROUTE_COLORS.get(r["resource_type"], "#888")
        icon  = RESOURCE_ICONS.get(r["resource_type"], "🚗")
        wpts  = r["waypoints"]
        via   = " → ".join(r.get("road_names", []))

        folium.PolyLine(
            wpts, color=color, weight=6, opacity=0.92,
            tooltip=(
                f"{icon} {r['unit_count']}× {r['resource_type']}\n"
                f"Zone: {r['zone']}  |  From: {r['origin_name']}\n"
                + (f"Via: {via}\n" if via else "")
                + f"Distance: {r['distance_km']} km  |  ETA: {r['eta_min']} min"
            ),
        ).add_to(fmap)
        for lat, lon in wpts[1:-1]:
            folium.CircleMarker([lat, lon], radius=4, color=color,
                                fill=True, fill_color=color, fill_opacity=0.8).add_to(fmap)
        if len(wpts) >= 2:
            folium.Marker(
                list(wpts[-1]),
                tooltip=f"↓ {r['resource_type']} arrives at {r['zone']}",
                icon=folium.DivIcon(
                    html=f'<div style="font-size:18px;color:{color};">▼</div>',
                    icon_size=(20, 20), icon_anchor=(10, 10),
                ),
            ).add_to(fmap)

    summary_rows = ""
    for r in all_routes:
        ico    = RESOURCE_ICONS.get(r["resource_type"], "🚗")
        status = (f"{r['distance_km']} km / {r['eta_min']} min"
                  if r["success"] else "❌ NO ROUTE")
        via    = " → ".join(r.get("road_names", [])[:3])
        summary_rows += (
            f"<tr>"
            f"<td>{ico} {r['unit_count']}× {r['resource_type']}</td>"
            f"<td><b>{r['zone']}</b></td>"
            f"<td>{r.get('origin_name','')}</td>"
            f"<td>{status}</td>"
            f"<td style='color:#666;font-size:11px'>{via}</td>"
            f"</tr>"
        )

    legend_html = f"""
    <div style="position:fixed;top:15px;right:15px;z-index:9999;
                background:white;padding:14px 18px;border-radius:10px;
                border:2px solid #333;font-family:Arial;font-size:13px;
                box-shadow:4px 4px 10px rgba(0,0,0,0.3);max-width:260px;">
      <b style="font-size:15px;">🗺 Route Agent</b>
      <div style="font-size:11px;color:#888;margin-bottom:8px;">Prayagraj Flood Scenario</div>
      <span style="color:#e74c3c;font-size:18px;">━━━</span> Ambulances<br>
      <span style="color:#2980b9;font-size:18px;">━━━</span> Rescue Teams<br>
      <span style="color:#16a085;font-size:18px;">━━━</span> Boats<br>
      <span style="color:#cc2200;font-size:14px;">╌╌╌</span> Blocked Roads<br>
      <span style="background:#e74c3c;padding:0 6px;border-radius:3px;color:white;">■</span> Crisis Zone<br>
      <span style="background:#3498db;padding:0 6px;border-radius:3px;color:white;">■</span> Flood Zone<br>
      <hr style="margin:8px 0;">
      <div style="font-size:11px;color:#555;">Click route lines for details</div>
    </div>

    <div style="position:fixed;bottom:30px;right:15px;z-index:9999;
                background:white;padding:10px 14px;border-radius:10px;
                border:2px solid #333;font-family:Arial;font-size:12px;
                box-shadow:4px 4px 10px rgba(0,0,0,0.3);max-width:700px;overflow-x:auto;">
      <b>📋 Deployment Summary</b><br>
      <table style="border-collapse:collapse;margin-top:6px;font-size:12px;">
        <tr style="background:#f0f0f0;">
          <th style="padding:4px 8px;border:1px solid #ccc;">Resource</th>
          <th style="padding:4px 8px;border:1px solid #ccc;">Zone</th>
          <th style="padding:4px 8px;border:1px solid #ccc;">From</th>
          <th style="padding:4px 8px;border:1px solid #ccc;">ETA</th>
          <th style="padding:4px 8px;border:1px solid #ccc;">Via</th>
        </tr>
        {summary_rows}
      </table>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"\n[Map] Saved → {abs_path}")
    if open_browser:
        webbrowser.open(f"file://{abs_path}")
    return abs_path


if __name__ == "__main__":
    build_map()