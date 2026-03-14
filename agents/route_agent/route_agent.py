"""
route_agent.py
--------------
MAIN ENTRY POINT for the Route Agent.

Call  plan_all_routes()  from the Master Coordinator.

This function:
  1. Builds a geo-transform from image metadata
  2. Converts each zone name → lat/lon destination
  3. Downloads (or loads synthetic) road network
  4. Applies blocked-road masks from the Vision Agent
  5. Routes each resource assignment from the Resource Agent
  6. Returns a list of route plan dicts for the Communication Agent

Input format (what the Coordinator passes in)
─────────────────────────────────────────────

image_meta = {
    "center_lat":   25.435,
    "center_lon":   81.846,
    "coverage_km":  5,
    "width_px":     640,
    "height_px":    640,
}

resource_assignments = {
    "Z12": {
        "ambulances":    2,
        "rescue_teams":  1,
        "boats":         0,
    },
    "Z34": {
        "ambulances":    0,
        "rescue_teams":  2,
        "boats":         1,
    }
}

base_locations = {
    "ambulance":    {"name": "City Hospital",    "lat": 25.440, "lon": 81.840},
    "rescue_team":  {"name": "Rescue Station A", "lat": 25.430, "lon": 81.855},
    "boat":         {"name": "Boat Depot",        "lat": 25.425, "lon": 81.848},
}

blocked_masks = {          # optional — numpy bool arrays from Vision Agent
    "flood":  <np.ndarray>,
    "debris": <np.ndarray>,
    "fire":   <np.ndarray>,
}
"""

import numpy as np
from typing import Optional

from .geo_reference     import build_geo_transform
from .zone_coordinates  import get_zone_latlon
from .road_network      import (
    download_road_network,
    build_synthetic_graph,
    nearest_node,
    nearest_node_synthetic,
    mask_to_polygons,
    remove_blocked_roads,
    OSMNX_AVAILABLE,
)
from .router import find_route, path_to_waypoints, build_route_plan


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def plan_all_routes(
    image_meta:           dict,
    resource_assignments: dict,
    base_locations:       dict,
    blocked_masks:        Optional[dict] = None,
    use_real_osm:         bool           = False,   # False = synthetic graph (no internet)
) -> list:
    """
    Plan routes for every resource assignment.

    Returns
    -------
    List of route plan dicts  (one per resource type per zone).
    Each dict is ready to be consumed by the Communication Agent.
    """

    print("\n" + "="*60)
    print("  ROUTE AGENT  —  planning routes")
    print("="*60)

    # ── Step 1: Build geo transform ────────────────────────────────────────
    geo_transform = build_geo_transform(
        center_lat      = image_meta["center_lat"],
        center_lon      = image_meta["center_lon"],
        coverage_km     = image_meta["coverage_km"],
        image_width_px  = image_meta["width_px"],
        image_height_px = image_meta["height_px"],
    )
    print(f"[RouteAgent] Geo-transform built. Top-left: "
          f"({geo_transform['top_left_lat']:.4f}, {geo_transform['top_left_lon']:.4f})")

    # ── Step 2: Build road graph ───────────────────────────────────────────
    center_lat = image_meta["center_lat"]
    center_lon = image_meta["center_lon"]

    if use_real_osm and OSMNX_AVAILABLE:
        G = download_road_network(center_lat, center_lon, radius_m=4000)
        _nearest_fn = nearest_node
    else:
        print("[RouteAgent] Using SYNTHETIC road graph (offline mode).")
        G = build_synthetic_graph()
        _nearest_fn = nearest_node_synthetic

    # ── Step 3: Apply blocked masks ────────────────────────────────────────
    if blocked_masks:
        all_blocked_polygons = []
        for mask_type, mask_array in blocked_masks.items():
            if mask_array is not None and isinstance(mask_array, np.ndarray):
                polys = mask_to_polygons(mask_array, geo_transform)
                print(f"[RouteAgent] '{mask_type}' mask → {len(polys)} blocked polygon(s)")
                all_blocked_polygons.extend(polys)

        if all_blocked_polygons:
            G = remove_blocked_roads(G, all_blocked_polygons)

    # ── Step 4: Route every resource to every zone ─────────────────────────
    all_routes = []

    for zone_name, assignments in resource_assignments.items():

        # figure out where this zone is on the ground
        dest_lat, dest_lon = get_zone_latlon(zone_name, geo_transform)
        dest_node          = _nearest_fn(G, dest_lat, dest_lon)

        print(f"\n[RouteAgent] Zone {zone_name} → ({dest_lat}, {dest_lon}), node {dest_node}")

        for resource_type, count in assignments.items():
            if count == 0:
                continue

            # normalise: try plural first, then singular, then plural of singular
            lookup_key = resource_type
            if lookup_key not in base_locations:
                # try stripping trailing 's'
                singular = resource_type.rstrip("s")
                if singular in base_locations:
                    lookup_key = singular
                # try adding 's'
                elif resource_type + "s" in base_locations:
                    lookup_key = resource_type + "s"
                else:
                    print(f"  [WARN] No base location for resource type '{resource_type}'. Skipping.")
                    continue

            base   = base_locations[lookup_key]
            origin_node = _nearest_fn(G, base["lat"], base["lon"])

            print(f"  Routing {count}x {resource_type} "
                  f"from '{base['name']}' (node {origin_node}) "
                  f"to {zone_name} (node {dest_node})")

            # run Dijkstra
            route_result = find_route(G, origin_node, dest_node)

            if route_result["success"]:
                waypoints = path_to_waypoints(G, route_result["node_path"])
            else:
                waypoints = []

            # package result
            plan = build_route_plan(
                zone_name          = zone_name,
                resource_type      = resource_type,
                origin_name        = base["name"],
                destination_latlon = (dest_lat, dest_lon),
                route_result       = route_result,
                waypoints          = waypoints,
            )
            # attach how many units are heading on this route
            plan["unit_count"] = count

            # log summary
            if route_result["success"]:
                print(f"    ✓ {plan['distance_km']} km  |  ETA {plan['eta_minutes']} min  "
                      f"|  {len(waypoints)} waypoints")
            else:
                print(f"    ✗ FAILED: {route_result['error']}")

            all_routes.append(plan)

    print(f"\n[RouteAgent] Done. {len(all_routes)} route(s) planned.\n")
    return all_routes


# ---------------------------------------------------------------------------
# Pretty printer  (useful during development)
# ---------------------------------------------------------------------------

def print_routes(routes: list):
    """Print all route plans in a readable format."""
    print("\n" + "="*60)
    print("  ROUTE PLANS SUMMARY")
    print("="*60)
    for r in routes:
        status = "✓" if r["success"] else "✗"
        print(f"\n  [{status}] {r['unit_count']}x {r['resource_type']}  "
              f"→  Zone {r['zone']}")
        print(f"       From        : {r['origin']}")
        print(f"       Destination : {r['destination_latlon']}")
        if r["success"]:
            print(f"       Distance    : {r['distance_km']} km")
            print(f"       ETA         : {r['eta_minutes']} min")
            print(f"       Waypoints   : {len(r['waypoints'])} nodes")
            print(f"       Blocked avoided: {r['blocked_roads_avoided']}")
        else:
            print(f"       ERROR: {r['error']}")