"""
road_network.py
---------------
Handles everything to do with the ROAD GRAPH.

Responsibilities:
  1. Download the real road network from OpenStreetMap using OSMnx.
  2. Convert the Vision Agent's segmentation masks (flood / debris / fire)
     into blocked polygons.
  3. Remove or heavily penalise edges (roads) that pass through those polygons.
  4. Return a clean NetworkX graph ready for routing.
  5. Provide a synthetic fallback graph for offline / unit-test use.
"""

import os
import math
import pickle
import numpy as np
import networkx as nx

try:
    import osmnx as ox
    ox.settings.use_cache   = True
    ox.settings.log_console = False
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ── In-memory caches (persist across calls within the same process) ────────────
# Avoids re-loading large pickle files and re-projecting graphs on every call.

_GRAPH_CACHE: dict = {}      # (lat4, lon4, radius_m) → nx.MultiDiGraph
_PROJ_CACHE:  dict = {}      # id(G) → projected_G (for nearest_node speed)

# ── Disk cache ────────────────────────────────────────────────────────────────

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_road_cache")


def _cache_path(lat: float, lon: float, radius_m: int) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"osm_{lat:.4f}_{lon:.4f}_{radius_m}m.pkl")


# ── 1. Download road graph ────────────────────────────────────────────────────

def download_road_network(center_lat: float, center_lon: float,
                          radius_m: int = 3000,
                          force_refresh: bool = False) -> nx.MultiDiGraph:
    """
    Download the drivable road network within radius_m metres of a point.
    Results are cached in memory (fastest) then on disk so subsequent calls
    are near-instant.

    Returns a NetworkX MultiDiGraph where:
        Nodes = road intersections  (attrs: x=lon, y=lat)
        Edges = road segments       (attrs: length, speed_kph, travel_time)
    """
    if not OSMNX_AVAILABLE:
        raise ImportError(
            "osmnx is not installed.\n"
            "Run:  pip install osmnx\n"
            "Or pass use_real_osm=False for offline mode."
        )

    mem_key = (round(center_lat, 4), round(center_lon, 4), radius_m)

    # ── Level 1: In-memory cache (fastest — no disk I/O) ──────────────────
    if not force_refresh and mem_key in _GRAPH_CACHE:
        G = _GRAPH_CACHE[mem_key]
        print(f"[RoadNetwork] In-memory cache hit — {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G

    fpath = _cache_path(center_lat, center_lon, radius_m)

    # ── Level 2: Disk cache ────────────────────────────────────────────────
    if not force_refresh and os.path.exists(fpath):
        print(f"[RoadNetwork] Loading cached OSM graph: {fpath}")
        with open(fpath, "rb") as fh:
            G = pickle.load(fh)
        print(f"[RoadNetwork] Disk cache hit — {len(G.nodes)} nodes, {len(G.edges)} edges")
        _GRAPH_CACHE[mem_key] = G  # warm the in-memory cache
        return G

    print(f"[RoadNetwork] Downloading OSM roads around "
          f"({center_lat:.4f}, {center_lon:.4f}), radius={radius_m} m ...")
    try:
        G = ox.graph_from_point(
            (center_lat, center_lon),
            dist=radius_m,
            network_type="drive",
            simplify=True,
        )
    except Exception as e:
        raise ConnectionError(
            f"[RoadNetwork] OSMnx download failed: {e}\n"
            "Check internet connection, or set use_real_osm=False."
        ) from e

    # Always reassign — OSMnx ≥ 1.x returns the modified graph.
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print(f"[RoadNetwork] Downloaded {len(G.nodes)} nodes, {len(G.edges)} edges.")

    with open(fpath, "wb") as fh:
        pickle.dump(G, fh)
    print(f"[RoadNetwork] Graph cached to {fpath}")
    _GRAPH_CACHE[mem_key] = G  # warm the in-memory cache
    return G


# ── 2. Nearest graph node ─────────────────────────────────────────────────────

def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Return the OSM node ID closest to (lat, lon).

    Projects graph to UTM first to avoid scikit-learn dependency.
    The projected graph is cached in _PROJ_CACHE (keyed by id(G)) so the
    expensive projection is only done ONCE per unique graph object — not
    on every call.
    Falls back to approximate Euclidean search if projection fails.
    """
    if not OSMNX_AVAILABLE:
        raise ImportError("Install osmnx: pip install osmnx")

    g_id = id(G)

    try:
        # Use cached projected graph (avoids O(V+E) re-projection each call)
        if g_id not in _PROJ_CACHE:
            _PROJ_CACHE[g_id] = ox.project_graph(G)
        G_proj = _PROJ_CACHE[g_id]

        import pyproj
        crs = G_proj.graph.get("crs", "EPSG:4326")
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x_proj, y_proj = transformer.transform(lon, lat)
        try:
            return ox.distance.nearest_nodes(G_proj, X=x_proj, Y=y_proj)
        except AttributeError:
            return ox.nearest_nodes(G_proj, X=x_proj, Y=y_proj)
    except Exception:
        # Fast Euclidean fallback (degree-space with latitude correction)
        best_node, best_dist = None, float("inf")
        cos_lat = math.cos(math.radians(lat))
        for nid, data in G.nodes(data=True):
            dlat = (data["y"] - lat) * 111_000
            dlon = (data["x"] - lon) * 111_000 * cos_lat
            d    = dlat * dlat + dlon * dlon  # skip sqrt — only comparing
            if d < best_dist:
                best_dist, best_node = d, nid
        return best_node


# ── 3. Convert segmentation mask → blocked polygons ──────────────────────────

def mask_to_polygons(mask: np.ndarray, geo_transform: dict,
                     downsample: int = 10) -> list:
    """
    Convert a binary numpy mask into a list of Shapely Polygons in GPS coords.

    The caller is responsible for binarising float probability maps at the
    appropriate threshold before passing here (done in route_agent.py).

    Parameters
    ----------
    mask          : 2-D numpy array (True / non-zero = blocked)
    geo_transform : dict from build_geo_transform()
    downsample    : stride for downsampling to limit memory use

    Returns
    -------
    list of shapely.geometry.Polygon  (empty list if no blocked pixels)
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("Install shapely: pip install shapely")

    from .geo_reference import pixel_to_latlon

    ys, xs = np.where(mask)
    ys = ys[::downsample]
    xs = xs[::downsample]

    if len(xs) == 0:
        return []

    points = []
    for px, py in zip(xs, ys):
        lat, lon = pixel_to_latlon(float(px), float(py), geo_transform)
        points.append((lon, lat))

    if len(points) < 3:
        return []

    poly = Polygon(points).convex_hull
    return [poly]


# ── 4. Remove blocked roads ───────────────────────────────────────────────────

def remove_blocked_roads(G: nx.MultiDiGraph, blocked_polygons: list,
                         penalty_weight: float = 1e9) -> nx.MultiDiGraph:
    """
    Set a huge travel_time penalty on edges whose midpoint lies inside a
    blocked polygon so Dijkstra routes around them.

    IMPORTANT: Works on a copy of G so the in-memory cached graph is NEVER
    mutated — flood masks from one pipeline run must not persist into the next.

    Using a penalty rather than deletion keeps the graph connected so a longer
    safe route can always be found.
    """
    if not SHAPELY_AVAILABLE or not blocked_polygons:
        return G

    # Work on a copy to protect the cached original
    G = G.copy()

    blocked_count = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        mid_lon = (u_data["x"] + v_data["x"]) / 2
        mid_lat = (u_data["y"] + v_data["y"]) / 2
        mid_pt  = Point(mid_lon, mid_lat)
        for poly in blocked_polygons:
            if poly.contains(mid_pt):
                G[u][v][key]["travel_time"] = penalty_weight
                G[u][v][key]["blocked"]     = True
                blocked_count += 1
                break

    print(f"[RoadNetwork] Penalised {blocked_count} road segment(s) inside hazard zones.")
    return G


# ── 5. Synthetic graph for offline / unit-test mode ──────────────────────────

def build_synthetic_graph(center_lat: float = 25.435,
                          center_lon: float = 81.846) -> nx.MultiDiGraph:
    """
    Build a small 3×3 grid road graph for offline / testing use.
    No internet required.

    Grid layout (each edge ≈ 550 m, speed 40 km/h):

        0 ── 1 ── 2
        |         |
        3 ── 4 ── 5
        |         |
        6 ── 7 ── 8

    The graph is centred on center_lat/center_lon so nearest-node lookups
    work correctly for any deployment area.
    """
    G = nx.MultiDiGraph()

    lat_step = 0.005
    lon_step = 0.005 / math.cos(math.radians(center_lat))

    base_lat = center_lat - lat_step
    base_lon = center_lon - lon_step

    node_positions = {
        0: (base_lat + lat_step*2, base_lon              ),
        1: (base_lat + lat_step*2, base_lon + lon_step   ),
        2: (base_lat + lat_step*2, base_lon + lon_step*2 ),
        3: (base_lat + lat_step,   base_lon              ),
        4: (base_lat + lat_step,   base_lon + lon_step   ),
        5: (base_lat + lat_step,   base_lon + lon_step*2 ),
        6: (base_lat,              base_lon              ),
        7: (base_lat,              base_lon + lon_step   ),
        8: (base_lat,              base_lon + lon_step*2 ),
    }

    for nid, (lat, lon) in node_positions.items():
        G.add_node(nid, y=lat, x=lon)

    road_edges = [
        (0,1), (1,2), (0,3), (2,5),
        (3,4), (4,5), (3,6), (5,8),
        (6,7), (7,8),
    ]

    speed_kph = 40.0
    for u, v in road_edges:
        lat_u, lon_u = node_positions[u]
        lat_v, lon_v = node_positions[v]
        dist_m = (((lat_v-lat_u)*111_000)**2 + ((lon_v-lon_u)*111_000)**2)**0.5
        travel_time = dist_m / (speed_kph * 1000 / 3600)
        attrs = dict(length=dist_m, speed_kph=speed_kph,
                     travel_time=travel_time, blocked=False)
        G.add_edge(u, v, key=0, **attrs)
        G.add_edge(v, u, key=0, **attrs)

    return G


def nearest_node_synthetic(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Find the nearest node by Euclidean distance in degree-space."""
    best_node, best_dist = None, float("inf")
    for nid, data in G.nodes(data=True):
        d = ((data["y"] - lat)**2 + (data["x"] - lon)**2)**0.5
        if d < best_dist:
            best_dist, best_node = d, nid
    return best_node