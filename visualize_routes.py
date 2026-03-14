"""
visualize_routes_real.py
------------------------
Realistic Prayagraj flood scenario.

On YOUR machine this uses REAL OSM roads (pip install osmnx).
Falls back to a hand-crafted Prayagraj road graph if no internet.

Run:
    python visualize_routes_real.py

Output:
    route_map_real.html   ← opens in browser automatically
"""

import os, sys, webbrowser, math
import numpy as np
import networkx as nx
import folium
from shapely.geometry import Point, Polygon, LineString

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO  — realistic Prayagraj flood near Yamuna
#
#  The Yamuna river runs south of the city.
#  Crisis zones Z78, Z79, Z87, Z88 are near the riverbank → flooded.
#  Boats are deployed there. Ambulances go to the northern collapsed zones.
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_META = {
    "center_lat":  25.435,
    "center_lon":  81.846,
    "coverage_km": 6,
    "width_px":    640,
    "height_px":   640,
}

# Resources coming from Resource Agent
RESOURCE_ASSIGNMENTS = {
    "Z23": {"ambulances": 2, "rescue_teams": 1, "boats": 0},   # Civil Lines collapse
    "Z45": {"ambulances": 1, "rescue_teams": 2, "boats": 0},   # Chowk area
    "Z78": {"ambulances": 0, "rescue_teams": 1, "boats": 2},   # Yamuna bank flood
    "Z89": {"ambulances": 0, "rescue_teams": 0, "boats": 3},   # Naini flood zone
}

# Real base locations in Prayagraj
BASE_LOCATIONS = {
    "ambulances":   {
        "name":  "Motilal Nehru Medical College",
        "lat":   25.4510, "lon": 81.8340,
        "icon":  "plus-sign", "color": "red",
    },
    "rescue_teams": {
        "name":  "NDRF Station, Civil Lines",
        "lat":   25.4590, "lon": 81.8420,
        "icon":  "home", "color": "blue",
    },
    "boats": {
        "name":  "Sangam Boat Launch (Naini Ghat)",
        "lat":   25.4300, "lon": 81.8700,
        "icon":  "tint", "color": "darkblue",
    },
}

ROUTE_COLORS = {
    "ambulances":   "#e74c3c",
    "rescue_teams": "#2980b9",
    "boats":        "#16a085",
}

# ─────────────────────────────────────────────────────────────────────────────
#  REALISTIC PRAYAGRAJ ROAD GRAPH
#  Hand-crafted from actual map coordinates.
#  Major roads: MG Marg, GT Road, Civil Lines, Kydganj, Zero Road etc.
# ─────────────────────────────────────────────────────────────────────────────

def build_prayagraj_graph():
    """
    Build a realistic road graph for central Prayagraj using actual
    road coordinates. Used as fallback when OSMnx has no internet.

    Node layout (approximate real locations):
                                        [NH30 bypass]
    Civil Lines North: A──────B──────C
                       |      |      |
    MG Marg:           D──────E──────F──────G
                       |      |      |      |
    Zero Road/GT:      H──────I──────J──────K
                       |      |      |      |
    Kydganj:           L──────M──────N──────O
                       |      |             |
    Naini/Yamuna:      P──────Q─────────────R   ← FLOOD ZONE
    """
    G = nx.MultiDiGraph()

    nodes = {
        # Civil Lines / North
        "A": (25.460, 81.826, "Leader Road / Civil Lines W"),
        "B": (25.460, 81.840, "MG Marg / Civil Lines Centre"),
        "C": (25.460, 81.856, "Kamla Nehru Road / Civil Lines E"),

        # MG Marg belt
        "D": (25.450, 81.826, "Howrah-Delhi Railway / West"),
        "E": (25.450, 81.840, "Allahabad City Station Road"),
        "F": (25.450, 81.856, "MG Marg East"),
        "G": (25.450, 81.870, "George Town / NH30"),

        # Zero Road / Grand Trunk Road
        "H": (25.440, 81.826, "Zero Road West"),
        "I": (25.440, 81.840, "Zero Road Centre / Chowk"),
        "J": (25.440, 81.856, "GT Road / Allahabad City"),
        "K": (25.440, 81.870, "GT Road East / Naini Bridge approach"),

        # Kydganj Road
        "L": (25.430, 81.826, "Atarsuiya Road"),
        "M": (25.430, 81.840, "Kydganj Road Centre"),
        "N": (25.430, 81.856, "Kydganj East"),
        "O": (25.430, 81.870, "Naini Bridge Road"),

        # Yamuna Flood Zone (low-lying, near river)
        "P": (25.420, 81.826, "Gaughat Road ⚠ FLOOD"),
        "Q": (25.420, 81.840, "Naini Bridge South ⚠ FLOOD"),
        "R": (25.420, 81.870, "Yamuna Bank / Naini ⚠ FLOOD"),
    }

    for nid, (lat, lon, label) in nodes.items():
        G.add_node(nid, y=lat, x=lon, label=label)

    # road edges — (u, v, road_name, speed_kph)
    roads = [
        # East-West roads
        ("A","B", "Leader Road",        40),
        ("B","C", "Kamla Nehru Road",   40),
        ("D","E", "Subhash Chowk Rd",   30),
        ("E","F", "MG Marg",            50),
        ("F","G", "MG Marg East",       50),
        ("H","I", "Zero Road",          40),
        ("I","J", "Grand Trunk Road",   50),
        ("J","K", "GT Road East",       50),
        ("L","M", "Kydganj Road",       35),
        ("M","N", "Kydganj East",       35),
        ("N","O", "Naini Bridge Rd",    40),
        ("P","Q", "Gaughat Road",       20),   # flood, slow
        ("Q","R", "Yamuna Bank Road",   20),   # flood, slow

        # North-South roads
        ("A","D", "Noorullah Road",     35),
        ("D","H", "Tilak Road",         35),
        ("H","L", "Atarsuiya Road",     35),
        ("L","P", "60 Feet Road",       25),   # enters flood zone
        ("B","E", "Phanrela Road",      40),
        ("E","I", "Allahabad Stn Rd",   40),
        ("I","M", "Sadar Road",         40),
        ("M","Q", "Sangam Marg",        25),   # enters flood zone
        ("C","F", "Hewett Road",        40),
        ("F","J", "Klopibagh Flyover",  55),
        ("J","N", "Naini Bridge Rd N",  40),
        ("N","R", "Naini Bridge",       30),   # crosses flood zone
        ("G","K", "NH30 / NH319D",      60),
        ("K","O", "Yamuna Expressway",  55),
        ("O","R", "Naini Road S",       30),
    ]

    for u, v, road_name, speed in roads:
        if u not in nodes or v not in nodes:
            continue
        lat_u, lon_u, _ = nodes[u]
        lat_v, lon_v, _ = nodes[v]
        dist_m = math.sqrt(((lat_v-lat_u)*111000)**2 + ((lon_v-lon_u)*111000*math.cos(math.radians(lat_u)))**2)
        tt = dist_m / (speed * 1000 / 3600)

        attrs = dict(length=dist_m, speed_kph=speed, travel_time=tt,
                     road_name=road_name, blocked=False)
        G.add_edge(u, v, key=0, **attrs)
        G.add_edge(v, u, key=0, **attrs)   # bidirectional

    return G, nodes


def apply_flood_blockages(G, flood_polygon_shapely):
    """
    Block any road whose midpoint falls inside the flood polygon.
    Returns list of blocked road names for reporting.
    """
    blocked_roads = []
    for u, v, key, data in G.edges(keys=True, data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        mid = Point((u_data["x"] + v_data["x"])/2,
                    (u_data["y"] + v_data["y"])/2)
        if flood_polygon_shapely.contains(mid):
            G[u][v][key]["travel_time"] = 1e9
            G[u][v][key]["blocked"] = True
            road = data.get("road_name", f"{u}-{v}")
            if road not in blocked_roads:
                blocked_roads.append(road)
    return blocked_roads


def nearest_node_real(G, lat, lon):
    best, best_d = None, float("inf")
    for nid, d in G.nodes(data=True):
        dist = math.sqrt((d["y"]-lat)**2 + (d["x"]-lon)**2)
        if dist < best_d:
            best_d, best = dist, nid
    return best


def dijkstra_route(G, origin_node, dest_node):
    try:
        path = nx.dijkstra_path(G, origin_node, dest_node, weight="travel_time")
        dist, tt = 0.0, 0.0
        for i in range(len(path)-1):
            e = min(G[path[i]][path[i+1]].values(), key=lambda d: d.get("travel_time", 1e9))
            dist += e.get("length", 0)
            tt   += e.get("travel_time", 0)
        waypoints = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
        road_names = []
        for i in range(len(path)-1):
            e = min(G[path[i]][path[i+1]].values(), key=lambda d: d.get("travel_time", 1e9))
            rn = e.get("road_name", "")
            if rn and (not road_names or road_names[-1] != rn):
                road_names.append(rn)
        return {
            "success": True, "path": path, "waypoints": waypoints,
            "distance_km": round(dist/1000, 2), "eta_min": round(tt/60, 1),
            "road_names": road_names
        }
    except nx.NetworkXNoPath:
        return {"success": False, "error": "No path — destination surrounded by blocked roads"}


# ─────────────────────────────────────────────────────────────────────────────
#  GEO helpers (inline, no import needed)
# ─────────────────────────────────────────────────────────────────────────────

def build_transform(center_lat, center_lon, coverage_km, w, h):
    deg_lat = coverage_km / 111.0
    deg_lon = coverage_km / (111.0 * math.cos(math.radians(center_lat)))
    aspect  = h / w
    return {
        "tl_lat": center_lat + deg_lat*aspect/2,
        "tl_lon": center_lon - deg_lon/2,
        "dlat":   deg_lat*aspect / h,
        "dlon":   deg_lon / w,
        "w": w, "h": h
    }

def zone_latlon(zone_name, t, rows=10, cols=10):
    body = zone_name.strip().upper().lstrip("Z")
    if "_" in body:
        r, c = int(body.split("_")[0]), int(body.split("_")[1])
    else:
        r, c = int(body[0]), int(body[1:]) if len(body)>1 else 1
    cell_h = t["h"] / rows
    cell_w = t["w"] / cols
    px = (c-1)*cell_w + cell_w/2
    py = (r-1)*cell_h + cell_h/2
    lat = t["tl_lat"] - py * t["dlat"]
    lon = t["tl_lon"] + px * t["dlon"]
    return round(lat,5), round(lon,5)


# ─────────────────────────────────────────────────────────────────────────────
#  MAP BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_map():
    print("\n" + "="*60)
    print("  ROUTE MAP — Realistic Prayagraj Flood Scenario")
    print("="*60)

    t = build_transform(**{k: IMAGE_META[k] for k in
        ["center_lat","center_lon","coverage_km"]},
        w=IMAGE_META["width_px"], h=IMAGE_META["height_px"])

    # ── Try real OSMnx first, fall back to hand-crafted ──────────────────
    G = None
    nodes_dict = None
    use_osm = False

    try:
        import osmnx as ox
        print("[Map] Downloading real OSM road network …")
        G_osm = ox.graph_from_point(
            (IMAGE_META["center_lat"], IMAGE_META["center_lon"]),
            dist=3500, network_type="drive"
        )
        G_osm = ox.add_edge_speeds(G_osm)
        G_osm = ox.add_edge_travel_times(G_osm)
        G = G_osm
        use_osm = True
        print(f"[Map] OSM graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    except Exception as e:
        print(f"[Map] OSMnx unavailable ({type(e).__name__}). Using realistic Prayagraj graph.")
        G, nodes_dict = build_prayagraj_graph()
        print(f"[Map] Prayagraj graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # ── Define flood zone: Yamuna riverbank (bottom portion of city) ──────
    # Real Yamuna bank area south of Kydganj
    flood_polygon_coords = [
        (81.810, 25.426),  # SW
        (81.900, 25.426),  # SE
        (81.900, 25.415),  # far SE
        (81.810, 25.415),  # far SW
    ]
    flood_shapely = Polygon(flood_polygon_coords)

    # Apply blockages
    blocked_roads = apply_flood_blockages(G, flood_shapely)
    print(f"[Map] Blocked roads: {blocked_roads}")

    # ── Nearest-node function ─────────────────────────────────────────────
    if use_osm:
        import osmnx as ox
        def get_nearest(lat, lon):
            return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    else:
        def get_nearest(lat, lon):
            return nearest_node_real(G, lat, lon)

    # ── Unblocked graph for boats (they sail through floodwater) ─────────
    G_boat, _ = build_prayagraj_graph() if not use_osm else (G, None)

    # ── Plan all routes ───────────────────────────────────────────────────
    all_routes = []
    for zone_name, assignments in RESOURCE_ASSIGNMENTS.items():
        dest_lat, dest_lon = zone_latlon(zone_name, t)
        print(f"\n[Route] Zone {zone_name} → ({dest_lat}, {dest_lon})")

        for rtype, count in assignments.items():
            if count == 0:
                continue
            base_key = rtype
            if base_key not in BASE_LOCATIONS:
                base_key = rtype.rstrip("s")
            if base_key not in BASE_LOCATIONS:
                continue

            # boats use unblocked graph — they navigate through floodwater
            routing_G = G_boat if "boat" in rtype else G
            base = BASE_LOCATIONS[base_key]
            origin_node = nearest_node_real(routing_G, base["lat"], base["lon"])
            dest_node   = nearest_node_real(routing_G, dest_lat, dest_lon)
            result = dijkstra_route(routing_G, origin_node, dest_node)

            all_routes.append({
                "zone": zone_name, "resource_type": rtype,
                "unit_count": count, "origin_name": base["name"],
                "dest_latlon": (dest_lat, dest_lon),
                **result
            })

            if result["success"]:
                print(f"  ✓ {count}x {rtype}: {result['distance_km']}km, "
                      f"{result['eta_min']}min via {' → '.join(result['road_names'])}")
            else:
                print(f"  ✗ {rtype}: {result['error']}")

    # ── Build Folium map ──────────────────────────────────────────────────
    fmap = folium.Map(
        location=[IMAGE_META["center_lat"], IMAGE_META["center_lon"]],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # Zone grid
    cell_dlat = t["dlat"] * (t["h"] / 10)
    cell_dlon = t["dlon"] * (t["w"] / 10)
    crisis_zones = set(RESOURCE_ASSIGNMENTS.keys())

    for row in range(1, 11):
        for col in range(1, 11):
            name = f"Z{row}{col}" if col <= 9 else f"Z{row}_{col}"
            clat, clon = zone_latlon(name, t)
            is_crisis = name in crisis_zones
            folium.Rectangle(
                bounds=[[clat-cell_dlat/2, clon-cell_dlon/2],
                        [clat+cell_dlat/2, clon+cell_dlon/2]],
                color="#444",
                weight=0.5,
                fill=True,
                fill_color="#e74c3c" if is_crisis else "#ffffff",
                fill_opacity=0.35 if is_crisis else 0.03,
                tooltip=f"{name} {'🚨 CRISIS' if is_crisis else ''}",
            ).add_to(fmap)

    # Crisis zone labels
    for zone_name in crisis_zones:
        clat, clon = zone_latlon(zone_name, t)
        assignments = RESOURCE_ASSIGNMENTS[zone_name]
        desc = ", ".join(f"{v}x {k}" for k,v in assignments.items() if v>0)
        folium.Marker(
            location=[clat, clon],
            tooltip=f"🚨 {zone_name} | {desc}",
            icon=folium.Icon(color="red", icon="exclamation-sign"),
        ).add_to(fmap)

    # Flood zone overlay
    folium.Polygon(
        locations=[[lat, lon] for lon, lat in flood_polygon_coords],
        color="#1a6fa8",
        weight=2,
        fill=True,
        fill_color="#3498db",
        fill_opacity=0.45,
        tooltip="🌊 Yamuna Flood Zone — roads blocked",
        dash_array="6 3",
    ).add_to(fmap)

    folium.Marker(
        location=[25.420, 81.845],
        tooltip=f"🌊 FLOODED — Yamuna Riverbank\nBlocked: {', '.join(blocked_roads[:4])}",
        icon=folium.Icon(color="lightblue", icon="tint"),
    ).add_to(fmap)

    # Draw road graph
    if not use_osm:
        drawn = set()
        for u, v, data in G.edges(data=True):
            key = tuple(sorted([str(u), str(v)]))
            if key in drawn:
                continue
            drawn.add(key)
            u_d, v_d = G.nodes[u], G.nodes[v]
            is_blocked = data.get("travel_time", 0) > 1e6
            folium.PolyLine(
                locations=[[u_d["y"], u_d["x"]], [v_d["y"], v_d["x"]]],
                color="#cc2200" if is_blocked else "#666666",
                weight=3 if is_blocked else 2,
                opacity=0.85 if is_blocked else 0.5,
                dash_array="8 4" if is_blocked else None,
                tooltip=f"{'🚫 BLOCKED — ' if is_blocked else ''}{data.get('road_name','road')}",
            ).add_to(fmap)
        # node dots
        for nid, d in G.nodes(data=True):
            folium.CircleMarker(
                [d["y"],d["x"]], radius=5, color="#333",
                fill=True, fill_color="#fff", fill_opacity=1,
                tooltip=f"{nid}: {d.get('label','')}",
            ).add_to(fmap)

    # Base location markers
    for rtype, base in BASE_LOCATIONS.items():
        folium.Marker(
            location=[base["lat"], base["lon"]],
            tooltip=f"📍 {base['name']}",
            popup=folium.Popup(f"<b>{base['name']}</b><br>Deploys: {rtype}", max_width=220),
            icon=folium.Icon(color=base["color"], icon=base["icon"]),
        ).add_to(fmap)

    # Draw routes
    for r in all_routes:
        if not r["success"]:
            continue
        color = ROUTE_COLORS.get(r["resource_type"], "#888")
        wpts  = r["waypoints"]

        # Main route line
        folium.PolyLine(
            locations=wpts,
            color=color,
            weight=6,
            opacity=0.92,
            tooltip=(
                f"{'🚑' if 'amb' in r['resource_type'] else '🚤' if 'boat' in r['resource_type'] else '🚒'} "
                f"{r['unit_count']}x {r['resource_type']}\n"
                f"Zone: {r['zone']}  |  From: {r['origin_name']}\n"
                f"Via: {' → '.join(r.get('road_names',[]))}\n"
                f"Distance: {r['distance_km']} km  |  ETA: {r['eta_min']} min"
            ),
        ).add_to(fmap)

        # Waypoint dots along route
        for lat, lon in wpts[1:-1]:
            folium.CircleMarker(
                [lat, lon], radius=4, color=color,
                fill=True, fill_color=color, fill_opacity=0.8,
            ).add_to(fmap)

        # Arrow at destination
        if len(wpts) >= 2:
            folium.Marker(
                location=list(wpts[-1]),
                tooltip=f"↓ {r['resource_type']} arrive at {r['zone']}",
                icon=folium.DivIcon(
                    html=f'<div style="font-size:18px; color:{color};">▼</div>',
                    icon_size=(20,20), icon_anchor=(10,10)
                ),
            ).add_to(fmap)

    # Route summary popup (click map title)
    summary_rows = ""
    for r in all_routes:
        icon = "🚑" if "amb" in r["resource_type"] else "🚤" if "boat" in r["resource_type"] else "🚒"
        status = f"{r['distance_km']}km / {r['eta_min']}min" if r["success"] else "❌ NO ROUTE"
        via = " → ".join(r.get("road_names", [])[:3])
        summary_rows += (
            f"<tr>"
            f"<td>{icon} {r['unit_count']}x {r['resource_type']}</td>"
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
      <span style="color:#666;font-size:18px;">━━━</span> Open Roads<br>
      <span style="background:#e74c3c;padding:0 6px;border-radius:3px;color:white;">■</span> Crisis Zone<br>
      <span style="background:#3498db;padding:0 6px;border-radius:3px;color:white;">■</span> Yamuna Flood Area<br>
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

    # Save
    out = os.path.join(os.path.dirname(__file__), "route_map_real.html")
    fmap.save(out)
    print(f"\n[Map] Saved → {out}")
    webbrowser.open(f"file://{os.path.abspath(out)}")
    return out


if __name__ == "__main__":
    build_map()