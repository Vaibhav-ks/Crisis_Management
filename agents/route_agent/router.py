"""
router.py
---------
Core routing logic.

Given:
  - A road graph (NetworkX MultiDiGraph with travel_time on edges)
  - An origin node ID
  - A destination node ID

This file runs Dijkstra's algorithm to find the shortest SAFE path,
calculates travel time and distance, and returns a clean route plan dict.

WHY Dijkstra and not A*?
  NetworkX ships Dijkstra out of the box and it is fast enough for city-scale
  graphs.  A* is faster on very large graphs but needs a heuristic function
  and is harder to set up.  You can swap in A* later with one line change.
"""

import math
import networkx as nx


# ---------------------------------------------------------------------------
# Path finding
# ---------------------------------------------------------------------------

def find_route(G: nx.MultiDiGraph, origin_node: int,
               dest_node: int) -> dict:
    """
    Run Dijkstra's shortest-path on the road graph.

    Uses 'travel_time' as the edge weight so blocked roads
    (travel_time = 1e9) are automatically avoided.

    Parameters
    ----------
    G            : road graph (nodes have x=lon, y=lat)
    origin_node  : node ID of start point
    dest_node    : node ID of destination

    Returns
    -------
    dict with keys:
        success        : bool
        node_path      : list of node IDs
        distance_km    : float
        eta_minutes    : float
        error          : str  (only present if success=False)
    """
    try:
        node_path = nx.dijkstra_path(
            G,
            source=origin_node,
            target=dest_node,
            weight="travel_time"
        )
    except nx.NetworkXNoPath:
        return {
            "success": False,
            "node_path": [],
            "distance_km": 0.0,
            "eta_minutes": 0.0,
            "error": "No path found — destination may be completely surrounded by blocked roads."
        }
    except nx.NodeNotFound as e:
        return {
            "success": False,
            "node_path": [],
            "distance_km": 0.0,
            "eta_minutes": 0.0,
            "error": f"Node not found in graph: {e}"
        }

    # accumulate distance and travel time along the path
    total_distance_m = 0.0
    total_time_s     = 0.0
    blocked_avoided  = 0

    for i in range(len(node_path) - 1):
        u = node_path[i]
        v = node_path[i + 1]

        # MultiDiGraph can have parallel edges — pick the one with
        # the lowest travel_time (OSMnx default behaviour)
        edge_data = min(
            G[u][v].values(),
            key=lambda d: d.get("travel_time", float("inf"))
        )

        total_distance_m += edge_data.get("length", 0.0)
        total_time_s     += edge_data.get("travel_time", 0.0)

        # count how many high-penalty edges were in alternative paths
        # (not used by Dijkstra but useful for reporting)
        if edge_data.get("blocked", False):
            blocked_avoided += 1

    return {
        "success":       True,
        "node_path":     node_path,
        "distance_km":   round(total_distance_m / 1000, 3),
        "eta_minutes":   round(total_time_s / 60, 2),
        "blocked_avoided": blocked_avoided,
        "error":         None
    }


# ---------------------------------------------------------------------------
# Convert node path → human-readable waypoints
# ---------------------------------------------------------------------------

def path_to_waypoints(G: nx.MultiDiGraph, node_path: list) -> list:
    """
    Convert a list of node IDs into a list of (lat, lon) waypoint tuples.

    Returns
    -------
    [ (lat, lon), (lat, lon), … ]
    """
    waypoints = []
    for nid in node_path:
        node_data = G.nodes[nid]
        lat = node_data.get("y", 0.0)
        lon = node_data.get("x", 0.0)
        waypoints.append((round(lat, 6), round(lon, 6)))
    return waypoints


# ---------------------------------------------------------------------------
# Build the final structured route plan
# ---------------------------------------------------------------------------

def build_route_plan(zone_name: str, resource_type: str,
                     origin_name: str, destination_latlon: tuple,
                     route_result: dict, waypoints: list) -> dict:
    """
    Package everything into one clean output dictionary.
    This is what gets sent to the Communication Agent.

    Parameters
    ----------
    zone_name          : e.g. 'Z12'
    resource_type      : e.g. 'ambulance', 'boat', 'rescue_team'
    origin_name        : human label e.g. 'Fire Station A'
    destination_latlon : (lat, lon) of zone center
    route_result       : dict from find_route()
    waypoints          : list from path_to_waypoints()

    Returns
    -------
    Structured route plan dict
    """
    if not route_result["success"]:
        return {
            "zone":               zone_name,
            "resource_type":      resource_type,
            "origin":             origin_name,
            "destination_latlon": destination_latlon,
            "success":            False,
            "error":              route_result["error"],
            "waypoints":          [],
            "distance_km":        0.0,
            "eta_minutes":        0.0,
            "blocked_roads_avoided": 0,
        }

    return {
        "zone":               zone_name,
        "resource_type":      resource_type,
        "origin":             origin_name,
        "destination_latlon": destination_latlon,
        "success":            True,
        "error":              None,
        "waypoints":          waypoints,
        "distance_km":        route_result["distance_km"],
        "eta_minutes":        route_result["eta_minutes"],
        "blocked_roads_avoided": route_result.get("blocked_avoided", 0),
    }


# ---------------------------------------------------------------------------
# Dynamic rerouting  (called when a new blockage is reported mid-mission)
# ---------------------------------------------------------------------------

def reroute(G: nx.MultiDiGraph, current_node: int, dest_node: int,
            newly_blocked_edges: list) -> dict:
    """
    Re-calculate the route because a road just got blocked.

    Parameters
    ----------
    G                    : road graph
    current_node         : where the team is RIGHT NOW
    dest_node            : destination (unchanged)
    newly_blocked_edges  : list of (u, v) tuples for newly blocked roads

    Returns
    -------
    Updated route_result dict from find_route()
    """
    # apply new blockages
    for u, v in newly_blocked_edges:
        if G.has_edge(u, v):
            for key in G[u][v]:
                G[u][v][key]["travel_time"] = 1e9
                G[u][v][key]["blocked"]     = True
        if G.has_edge(v, u):
            for key in G[v][u]:
                G[v][u][key]["travel_time"] = 1e9
                G[v][u][key]["blocked"]     = True

    print(f"[Router] Rerouting after {len(newly_blocked_edges)} new blockage(s).")
    return find_route(G, current_node, dest_node)