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

WHY OSMnx?
  OSMnx wraps OpenStreetMap data. One call downloads every road and
  intersection for a lat/lon bounding box as a NetworkX DiGraph where
  nodes = intersections and edges = road segments with real speed metadata.

═══════════════════════════════════════════════════════════════════════════════
BUGS FOUND AND FIXED
═══════════════════════════════════════════════════════════════════════════════

BUG 1 ─ ox.config() removed in OSMnx ≥ 1.3.0
  The old code never called ox.config() here, but the route_agent.py used to
  set settings via the removed API. We now set them via ox.settings directly.
  Impact: AttributeError crash at import time → OSMNX_AVAILABLE stays False
           even when osmnx IS installed, silently forcing synthetic mode forever.
  Fix:    ox.settings.use_cache = True
          ox.settings.log_console = False

BUG 2 ─ ox.add_edge_speeds / ox.add_edge_travel_times return value discarded
  Old code:   ox.add_edge_speeds(G)          # return value thrown away
  New code:   G = ox.add_edge_speeds(G)
  Impact: In OSMnx ≥ 1.x these functions return the modified graph. Not
          reassigning means travel_time is never written onto edges, so
          nx.dijkstra_path(weight="travel_time") crashes with KeyError
          or returns a nonsense path with weight=None.
  Fix:    Always reassign the return value.

BUG 3 ─ ox.nearest_nodes() moved to ox.distance.nearest_nodes() in ≥ 1.x
  Old code already used ox.distance.nearest_nodes() — CORRECT.
  But if someone runs OSMnx < 1.x the new call fails, so we add a fallback.
  Fix:    try ox.distance.nearest_nodes(); except AttributeError: ox.nearest_nodes()

BUG 4 ─ No disk cache → re-downloads the same area on every run (30 s each)
  Fix:    pickle-based cache keyed by (lat, lon, radius_m) in _road_cache/

BUG 5 ─ mask_to_polygons receives float arrays (0.0–1.0) from Vision Agent
  detect_flood() returns a float probability map, not a boolean mask.
  np.where(float_array) treats ANY non-zero value as True, so even a
  flood_score of 0.001 blocks roads. The threshold should be applied first.
  Fix:    Caller (route_agent.py) binarises with a configurable threshold
          before passing to mask_to_polygons. mask_to_polygons now accepts
          both bool and float arrays and documents this clearly.
"""

import os
import pickle
import numpy as np
import networkx as nx

# ── Optional heavy dependencies ───────────────────────────────────────────────
# Imported lazily so the module can be imported without network / GPU access.

try:
    import osmnx as ox

    # FIX 1: ox.config() was removed in OSMnx ≥ 1.3.0. Use ox.settings instead.
    ox.settings.use_cache   = True   # cache tile downloads to disk automatically
    ox.settings.log_console = False  # suppress verbose OSMnx logging

    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ── Disk cache ────────────────────────────────────────────────────────────────

# Cache lives next to this source file so it persists across runs
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_road_cache")


def _cache_path(lat: float, lon: float, radius_m: int) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"osm_{lat:.4f}_{lon:.4f}_{radius_m}m.pkl")


# ── 1. Download road graph ────────────────────────────────────────────────────

def download_road_network(center_lat: float, center_lon: float,
                          radius_m: int = 3000,
                          force_refresh: bool = False) -> nx.MultiDiGraph:
    """
    Download the drivable road network within `radius_m` metres of a point.

    Parameters
    ----------
    center_lat    : latitude  of the area centre
    center_lon    : longitude of the area centre
    radius_m      : download radius in metres (default 3 km)
    force_refresh : ignore disk cache and re-download from OSM

    Returns
    -------
    G : NetworkX MultiDiGraph
        Nodes  = road intersections  (attrs: x=lon, y=lat)
        Edges  = road segments       (attrs: length, speed_kph, travel_time)
    """
    if not OSMNX_AVAILABLE:
        raise ImportError(
            "osmnx is not installed.\n"
            "Run:  pip install osmnx\n"
            "Or pass use_real_osm=False to plan_all_routes() for offline mode."
        )

    # ── Check disk cache (BUG 4 fix) ─────────────────────────────────────
    fpath = _cache_path(center_lat, center_lon, radius_m)
    if not force_refresh and os.path.exists(fpath):
        print(f"[RoadNetwork] Loading cached graph from {fpath}")
        with open(fpath, "rb") as fh:
            G = pickle.load(fh)
        print(f"[RoadNetwork] Loaded {len(G.nodes)} nodes, {len(G.edges)} edges from cache.")
        return G

    # ── Download from OpenStreetMap ───────────────────────────────────────
    print(f"[RoadNetwork] Downloading OSM roads around "
          f"({center_lat:.4f}, {center_lon:.4f}), radius={radius_m}m …")
    try:
        G = ox.graph_from_point(
            (center_lat, center_lon),
            dist=radius_m,
            network_type="drive",  # drivable roads only
            simplify=True,         # merge degree-2 nodes (no real intersection)
        )
    except Exception as e:
        raise ConnectionError(
            f"[RoadNetwork] OSMnx download failed: {e}\n"
            "Check internet connection, or use use_real_osm=False."
        ) from e

    # FIX 2: Always reassign — in OSMnx ≥ 1.x these return the modified graph.
    # Not reassigning means travel_time is never set and Dijkstra crashes.
    G = ox.add_edge_speeds(G)        # adds 'speed_kph' attribute to each edge
    G = ox.add_edge_travel_times(G)  # adds 'travel_time' (seconds) to each edge

    print(f"[RoadNetwork] Downloaded {len(G.nodes)} nodes, {len(G.edges)} edges.")

    # ── Save to disk cache ────────────────────────────────────────────────
    with open(fpath, "wb") as fh:
        pickle.dump(G, fh)
    print(f"[RoadNetwork] Graph cached to {fpath}")

    return G


# ── 2. Nearest graph node  (BUG 3 fix) ───────────────────────────────────────

def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Return the OSM node ID of the road intersection closest to (lat, lon).

    FIX 3: ox.nearest_nodes() was moved to ox.distance.nearest_nodes() in
    OSMnx ≥ 1.x.  We try the new location first, fall back to the old one.
    """
    if not OSMNX_AVAILABLE:
        raise ImportError("Install osmnx: pip install osmnx")
    try:
        # OSMnx ≥ 1.x  (current)
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except AttributeError:
        # OSMnx < 1.x  (legacy)
        return ox.nearest_nodes(G, X=lon, Y=lat)


# ── 3. Convert segmentation mask → blocked polygons ──────────────────────────

def mask_to_polygons(mask: np.ndarray, geo_transform: dict,
                     downsample: int = 10) -> list:
    """
    Convert a binary (or float) numpy mask into a list of Shapely Polygons
    in real-world (lon, lat) coordinates.

    The Vision Agent's detect_flood() returns a FLOAT probability map (0–1).
    FIX 5: The caller in route_agent.py is responsible for binarising it
    at the appropriate threshold before calling this function.
    This function accepts both bool and float arrays:
      - bool  array: np.where treats True pixels as blocked.
      - float array: np.where treats any non-zero value as blocked.

    Strategy: collect all blocked pixel centres → convex hull → one polygon.
    For more refined per-blob contours, use cv2.findContours instead.

    Parameters
    ----------
    mask          : 2-D numpy array (True / non-zero = blocked)
    geo_transform : dict from build_geo_transform()
    downsample    : take every Nth pixel to limit memory usage (1M → 100k)

    Returns
    -------
    list of shapely.geometry.Polygon  (empty list if no blocked pixels)
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("Install shapely: pip install shapely")

    from .geo_reference import pixel_to_latlon  # lazy to avoid circular import

    ys, xs = np.where(mask)           # row-indices, col-indices of True pixels
    ys = ys[::downsample]             # downsample for speed
    xs = xs[::downsample]

    if len(xs) == 0:
        return []

    # Convert pixel positions to (lon, lat) — Shapely uses (x=lon, y=lat)
    points = []
    for px, py in zip(xs, ys):
        lat, lon = pixel_to_latlon(float(px), float(py), geo_transform)
        points.append((lon, lat))

    if len(points) < 3:
        return []

    # One convex hull wrapping all blocked pixels.
    # Diagram of what this looks like:
    #   blocked pixels:  ● ●        convex hull:   ________
    #                  ● ● ● ●                    /        \
    #                    ● ●                      |          |
    #                                              \________/
    poly = Polygon(points).convex_hull
    return [poly]


# ── 4. Remove blocked roads ───────────────────────────────────────────────────

def remove_blocked_roads(G: nx.MultiDiGraph, blocked_polygons: list,
                         penalty_weight: float = 1e9) -> nx.MultiDiGraph:
    """
    For each road edge whose midpoint lies inside a blocked polygon, set its
    travel_time to a huge penalty so Dijkstra/A* will never choose it.

    WHY not delete edges?
      Deleting can disconnect the graph entirely, making nx.dijkstra_path
      raise NetworkXNoPath even when a longer safe route exists.
      A penalty keeps the graph connected while still routing around hazards.

    Parameters
    ----------
    G                : road graph from download_road_network()
    blocked_polygons : Shapely Polygons from mask_to_polygons()
    penalty_weight   : travel_time (seconds) assigned to blocked edges

    Returns
    -------
    G with updated travel_time on blocked edges
    """
    if not SHAPELY_AVAILABLE or not blocked_polygons:
        return G

    blocked_count = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        # Midpoint test — testing full geometry is ~10x slower for similar results
        mid_lon = (u_data["x"] + v_data["x"]) / 2
        mid_lat = (u_data["y"] + v_data["y"]) / 2
        mid_pt  = Point(mid_lon, mid_lat)
        for poly in blocked_polygons:
            if poly.contains(mid_pt):
                G[u][v][key]["travel_time"] = penalty_weight
                G[u][v][key]["blocked"]     = True
                blocked_count += 1
                break  # no need to test remaining polygons for this edge

    print(f"[RoadNetwork] Blocked {blocked_count} road segment(s).")
    return G


# ── 5. Synthetic graph for offline / unit-test mode ──────────────────────────

def build_synthetic_graph(center_lat: float = 25.435,
                          center_lon: float = 81.846) -> nx.MultiDiGraph:
    """
    Build a small fake road graph for local/offline testing.
    No internet required.

    Grid layout (each edge ≈ 550 m, speed 40 km/h):

        0 ── 1 ── 2
        |         |
        3 ── 4 ── 5
        |         |
        6 ── 7 ── 8

    Parameters
    ----------
    center_lat : latitude  of the grid centre  (default: Prayagraj)
    center_lon : longitude of the grid centre

    BUG FIX: The old graph was always hardcoded to Prayagraj (25.435, 81.846).
    When image_meta used different coordinates (e.g. Delhi, Mumbai),
    nearest_node_synthetic() measured distance from those far-away coordinates
    to a Prayagraj-centred graph — every origin AND destination snapped to the
    exact same node, giving 0 km distance and 1-waypoint routes for everything.

    Fix: accept center_lat/center_lon so the graph is built AROUND the actual
    area being analysed. route_agent.py passes image_meta["center_lat/lon"] here.
    """
    G = nx.MultiDiGraph()

    # step ≈ 550 m at this latitude; scales correctly with cos(lat) for longitude
    import math
    lat_step = 0.005                                      # ≈ 555 m N-S
    lon_step = 0.005 / math.cos(math.radians(center_lat))  # same distance E-W

    # Offset from centre: grid spans 2 steps each way
    base_lat = center_lat - lat_step        # bottom of grid
    base_lon = center_lon - lon_step        # left  of grid

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
        (0,1), (1,2),
        (0,3), (2,5),
        (3,4), (4,5),
        (3,6), (5,8),
        (6,7), (7,8),
    ]

    speed_kph = 40.0
    for u, v in road_edges:
        lat_u, lon_u = node_positions[u]
        lat_v, lon_v = node_positions[v]
        dist_m = (((lat_v-lat_u)*111_000)**2 + ((lon_v-lon_u)*111_000)**2)**0.5
        travel_time = dist_m / (speed_kph * 1000 / 3600)

        G.add_edge(u, v, key=0, length=dist_m,
                   speed_kph=speed_kph, travel_time=travel_time, blocked=False)
        G.add_edge(v, u, key=0, length=dist_m,
                   speed_kph=speed_kph, travel_time=travel_time, blocked=False)

    return G


def nearest_node_synthetic(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Find nearest node by Euclidean distance in degree-space.
    Drop-in replacement for nearest_node() when OSMnx is not available.
    """
    best_node = None
    best_dist = float("inf")
    for nid, data in G.nodes(data=True):
        d = ((data["y"] - lat)**2 + (data["x"] - lon)**2)**0.5
        if d < best_dist:
            best_dist = d
            best_node = nid
    return best_node