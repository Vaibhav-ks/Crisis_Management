"""
road_network.py
---------------
Handles everything to do with the ROAD GRAPH.

Steps this file is responsible for:
  1. Download the real road network from OpenStreetMap using OSMnx.
  2. Convert the Vision Agent's segmentation masks (flood / debris / fire)
     into blocked polygons.
  3. Remove or heavily penalise edges (roads) that pass through those polygons.
  4. Return a clean NetworkX graph ready for routing.

WHY OSMnx?
  OSMnx wraps OpenStreetMap's data.  One function call downloads every road,
  lane, and intersection for any lat/lon bounding box.  It returns a NetworkX
  DiGraph where nodes are intersections and edges are road segments with real
  travel-speed metadata.
"""

import numpy as np
import networkx as nx

# OSMnx and Shapely are imported lazily so the file can be imported for
# unit-testing without network access.
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Download road graph
# ---------------------------------------------------------------------------

def download_road_network(center_lat: float, center_lon: float,
                          radius_m: int = 3000) -> nx.MultiDiGraph:
    """
    Download the drivable road network within `radius_m` metres of a point.

    Parameters
    ----------
    center_lat  : latitude  of the area centre
    center_lon  : longitude of the area centre
    radius_m    : how far out to download roads (default 3 km)

    Returns
    -------
    G : NetworkX MultiDiGraph (A directed graph with multiple edges.)
        Nodes  = intersections, with attributes  x (lon) and y (lat)
        Edges  = road segments,  with attributes  length, speed_kph, travel_time
    """
    if not OSMNX_AVAILABLE:
        raise ImportError("Install OSMnx:  pip install osmnx")

    print(f"[RoadNetwork] Downloading roads around ({center_lat}, {center_lon}) …")

    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist=radius_m,
        network_type="drive"      # only drivable roads
    )

    # Add speed and travel-time attributes to every edge
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    print(f"[RoadNetwork] Downloaded {len(G.nodes)} nodes, {len(G.edges)} edges.")
    return G


# ---------------------------------------------------------------------------
# 2. Convert segmentation mask → blocked polygons
# ---------------------------------------------------------------------------

def mask_to_polygons(binary_mask: np.ndarray, geo_transform: dict,
                     downsample: int = 10) -> list:
    """
    Convert a binary numpy mask (flood / debris / fire) into a list of
    Shapely Polygons in real-world (lon, lat) coordinates.

    We use a simple approach: find connected blobs of True pixels and
    wrap each blob in a convex hull.  This is fast and good enough for routing.

    Parameters
    ----------
    binary_mask   : 2-D boolean numpy array (True = blocked)
    ex -0 0 0 1 1
        0 0 1 1 1      1 = flooded
        0 0 0 0 0

    geo_transform : from build_geo_transform()
    downsample    : take every Nth pixel to speed up processing

    Returns
    -------
    list of shapely.geometry.Polygon
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("Install Shapely:  pip install shapely")

    from .geo_reference import pixel_to_latlon

    # find all blocked pixel coordinates (downsampled)
    ys, xs = np.where(binary_mask)

    # downsample
    ys = ys[::downsample]
    xs = xs[::downsample]   #agr mask me one million pixels h to 100k pixels 

    if len(xs) == 0:
        return []

    # convert every blocked pixel to lon/lat
    points = []
    for px, py in zip(xs, ys):
        lat, lon = pixel_to_latlon(float(px), float(py), geo_transform)
        points.append((lon, lat))   # Shapely uses (x=lon, y=lat)

    if len(points) < 3:
        return []

    # build one convex hull around all blocked points
    # (for a more refined approach you could use cv2.findContours)
    poly = Polygon(points).convex_hull
    return [poly]

# Flooded pixels
#       ● ●
#    ● ● ● ●
#       ● ●
# convex hull aisa kuch bna dega ab ye pura blocked h 
#   ________
#  /        \
# |          |
#  \________/

# ---------------------------------------------------------------------------
# 3. Remove blocked roads from the graph
# ---------------------------------------------------------------------------

def remove_blocked_roads(G: nx.MultiDiGraph, blocked_polygons: list,
                         penalty_weight: float = 1e9) -> nx.MultiDiGraph:
    """
    For each road edge that passes through a blocked polygon, set its
    travel_time to a very large number so Dijkstra / A* will never use it.

    We do NOT delete edges because that can disconnect the graph.
    Instead we give them a near-infinite travel_time cost.

    Parameters
    ----------
    G                : road graph from download_road_network()
    blocked_polygons : list of Shapely Polygons (from mask_to_polygons)
    penalty_weight   : travel time penalty applied to blocked edges

    Returns
    -------
    G with updated travel_time on blocked edges
    """
    if not SHAPELY_AVAILABLE or not blocked_polygons:
        return G

    blocked_count = 0

    for u, v, key, data in G.edges(keys=True, data=True):
        # get midpoint of edge  (node u coordinates)
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        mid_lon = (u_data["x"] + v_data["x"]) / 2
        mid_lat = (u_data["y"] + v_data["y"]) / 2
        mid_point = Point(mid_lon, mid_lat)  #testing full road geometry is expensive

        for poly in blocked_polygons:
            if poly.contains(mid_point):
                G[u][v][key]["travel_time"] = penalty_weight
                G[u][v][key]["blocked"]     = True
                blocked_count += 1
                break

    print(f"[RoadNetwork] Blocked {blocked_count} road segments.")
    return G

#we are not deleting cuz deleting may cause the graph to disconnect 
# ---------------------------------------------------------------------------
# 4. Find nearest graph node to a lat/lon
# ---------------------------------------------------------------------------

def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Return the OSM node ID of the road intersection closest to (lat, lon).
    """
    if not OSMNX_AVAILABLE:
        raise ImportError("Install OSMnx:  pip install osmnx")

    node_id = ox.distance.nearest_nodes(G, X=lon, Y=lat)
    return node_id


# ---------------------------------------------------------------------------
# FALLBACK: build a tiny synthetic graph for testing WITHOUT internet
# ---------------------------------------------------------------------------

def build_synthetic_graph() -> nx.MultiDiGraph:
    """
    Build a small fake road graph for local testing.
    No internet required.

    Layout (each edge ~500 m, speed 40 km/h):

        0 ── 1 ── 2
        |         |
        3 ── 4 ── 5
        |         |
        6 ── 7 ── 8

    Nodes have (x=lon, y=lat) attributes based near Prayagraj, UP.
    """
    G = nx.MultiDiGraph()

    base_lat = 25.435
    base_lon = 81.846
    step     = 0.005   # ~550 m per step

    node_positions = {
        0: (base_lat + step*2, base_lon          ),
        1: (base_lat + step*2, base_lon + step   ),
        2: (base_lat + step*2, base_lon + step*2 ),
        3: (base_lat + step,   base_lon          ),
        4: (base_lat + step,   base_lon + step   ),
        5: (base_lat + step,   base_lon + step*2 ),
        6: (base_lat,          base_lon          ),
        7: (base_lat,          base_lon + step   ),
        8: (base_lat,          base_lon + step*2 ),
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
        # approximate distance using node positions
        lat_u, lon_u = node_positions[u]
        lat_v, lon_v = node_positions[v]
        dist_m = (((lat_v - lat_u)*111000)**2 + ((lon_v - lon_u)*111000)**2) ** 0.5
        travel_time = dist_m / (speed_kph * 1000 / 3600)   # seconds

        # add both directions
        G.add_edge(u, v, key=0, length=dist_m,
                   speed_kph=speed_kph, travel_time=travel_time, blocked=False)
        G.add_edge(v, u, key=0, length=dist_m,
                   speed_kph=speed_kph, travel_time=travel_time, blocked=False)

    return G


def nearest_node_synthetic(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Find the nearest node in a synthetic graph using Euclidean distance.
    (Replaces OSMnx's nearest_nodes for offline testing.)
    """
    best_node = None
    best_dist = float("inf")

    for nid, data in G.nodes(data=True):
        d = ((data["y"] - lat)**2 + (data["x"] - lon)**2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_node = nid

    return best_node