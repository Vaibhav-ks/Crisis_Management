"""
router.py
---------
Core routing logic.

Given:
  - A road graph (NetworkX MultiDiGraph with travel_time on edges)
  - An origin node ID
  - A destination node ID

Runs Dijkstra's algorithm to find the shortest safe path,
calculates travel time and distance, and returns a clean route plan dict.
"""

import math
import networkx as nx


# ---------------------------------------------------------------------------
# Path finding
# ---------------------------------------------------------------------------

def find_route(G: nx.MultiDiGraph, origin_node: int, dest_node: int) -> dict:
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
    dict with keys: success, node_path, distance_km, eta_minutes, error
    """
    try:
        node_path = nx.dijkstra_path(
            G, source=origin_node, target=dest_node, weight="travel_time"
        )
    except nx.NetworkXNoPath:
        return {
            "success": False, "node_path": [], "distance_km": 0.0,
            "eta_minutes": 0.0,
            "error": "No path found — destination may be surrounded by blocked roads."
        }
    except nx.NodeNotFound as e:
        return {
            "success": False, "node_path": [], "distance_km": 0.0,
            "eta_minutes": 0.0, "error": f"Node not found in graph: {e}"
        }

    total_distance_m = 0.0
    total_time_s     = 0.0
    blocked_avoided  = 0

    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        # MultiDiGraph can have parallel edges — pick lowest travel_time
        edge_data = min(G[u][v].values(), key=lambda d: d.get("travel_time", float("inf")))
        total_distance_m += edge_data.get("length", 0.0)
        total_time_s     += edge_data.get("travel_time", 0.0)
        if edge_data.get("blocked", False):
            blocked_avoided += 1

    return {
        "success":         True,
        "node_path":       node_path,
        "distance_km":     round(total_distance_m / 1000, 3),
        "eta_minutes":     round(total_time_s / 60, 2),
        "blocked_avoided": blocked_avoided,
        "error":           None,
    }


# ---------------------------------------------------------------------------
# Convert node path → waypoints
# ---------------------------------------------------------------------------

def path_to_waypoints(G: nx.MultiDiGraph, node_path: list) -> list:
    """
    Convert a list of node IDs into (lat, lon) waypoint tuples.
    Returns [ (lat, lon), (lat, lon), … ]
    """
    return [
        (round(G.nodes[nid].get("y", 0.0), 6),
         round(G.nodes[nid].get("x", 0.0), 6))
        for nid in node_path
    ]


# ---------------------------------------------------------------------------
# Build the final structured route plan
# ---------------------------------------------------------------------------

def build_route_plan(zone_name: str, resource_type: str,
                     origin_name: str, destination_latlon: tuple,
                     route_result: dict, waypoints: list) -> dict:
    """
    Package everything into one clean output dictionary sent to the
    Communication Agent and Route Map generator.
    """
    base = {
        "zone":               zone_name,
        "resource_type":      resource_type,
        "origin_name":        origin_name,
        "destination_latlon": destination_latlon,
    }

    if not route_result["success"]:
        return {**base,
                "success": False, "error": route_result["error"],
                "waypoints": [], "distance_km": 0.0, "eta_minutes": 0.0,
                "blocked_roads_avoided": 0}

    return {**base,
            "success":               True,
            "error":                 None,
            "waypoints":             waypoints,
            "distance_km":           route_result["distance_km"],
            "eta_minutes":           route_result["eta_minutes"],
            "blocked_roads_avoided": route_result.get("blocked_avoided", 0)}


# ---------------------------------------------------------------------------
# Dynamic rerouting
# ---------------------------------------------------------------------------

def reroute(G: nx.MultiDiGraph, current_node: int, dest_node: int,
            newly_blocked_edges: list) -> dict:
    """
    Re-calculate the route because a road just got blocked mid-mission.

    Parameters
    ----------
    G                    : road graph
    current_node         : where the team is RIGHT NOW
    dest_node            : destination (unchanged)
    newly_blocked_edges  : list of (u, v) tuples for newly blocked roads
    """
    for u, v in newly_blocked_edges:
        for direction in [(u, v), (v, u)]:
            if G.has_edge(*direction):
                for key in G[direction[0]][direction[1]]:
                    G[direction[0]][direction[1]][key]["travel_time"] = 1e9
                    G[direction[0]][direction[1]][key]["blocked"]     = True

    print(f"[Router] Rerouting after {len(newly_blocked_edges)} new blockage(s).")
    return find_route(G, current_node, dest_node)