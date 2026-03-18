"""
route_agent.py
--------------
Main entry point for the Route Agent.

Called as plan_all_routes(...) from master_nodes.py after admin approval.

Pipeline:
  1. Build a geo-transform from image metadata (pixel ↔ GPS conversion)
  2. Convert each zone name → lat/lon destination on the ground
  3. Download (or load cached) real OSM road network, or use synthetic graph
  4. Apply blocked-road masks from the Vision Agent (flood probability map)
  5. Route each resource assignment from the Resource Agent (rescue_plan)
  6. Return a list of route plan dicts ready for communication/display

─────────────────────────────────────────────────────────────────────────────
VISION AGENT OUTPUT  (zone_map)
─────────────────────────────────────────────────────────────────────────────
{
    "zone_map": {
        "Z00": {"flood_score": 0.12, "damage_score": 0.0,  "severity": 0.072},
        "Z01": {"flood_score": 0.80, "damage_score": 0.45, "severity": 0.66},
        ...  (100 zones: Z00 to Z99, 0-based row/col)
    }
}
detect_flood() additionally returns a float H×W array (0–1) used as
blocked_masks["flood"] to remove flooded roads.

─────────────────────────────────────────────────────────────────────────────
RESOURCE AGENT OUTPUT  (rescue_plan)
─────────────────────────────────────────────────────────────────────────────
{
    "Z35": {"boats": 2, "ambulances": 1, "rescue_teams": 2},
    "Z01": {"boats": 0, "ambulances": 1, "rescue_teams": 1},
    ...
}
Both plural ("boats") and singular ("boat") LLM output keys are handled
via the _RESOURCE_KEY_MAP normalisation table.
"""

import numpy as np
from typing import Optional

from .geo_reference    import build_geo_transform
from .zone_coordinates import get_zone_latlon
from .road_network     import (
    download_road_network,
    build_synthetic_graph,
    nearest_node,
    nearest_node_synthetic,
    mask_to_polygons,
    remove_blocked_roads,
    OSMNX_AVAILABLE,
)
from .router import find_route, path_to_waypoints, build_route_plan


# ── Resource key normalisation ────────────────────────────────────────────────
# Maps any LLM output variant → singular key used in base_locations.

_RESOURCE_KEY_MAP = {
    "boats":        "boat",
    "ambulances":   "ambulance",
    "rescue_teams": "rescue_team",
    "helicopters":  "helicopter",
    "trucks":       "truck",
    "fire_trucks":  "fire_truck",
    # Singular forms pass through unchanged
    "boat":         "boat",
    "ambulance":    "ambulance",
    "rescue_team":  "rescue_team",
    "helicopter":   "helicopter",
    "truck":        "truck",
    "fire_truck":   "fire_truck",
}


def _canonical_resource_key(resource_type: str) -> Optional[str]:
    """Return the canonical singular key, or None if unknown."""
    key = resource_type.lower().strip()
    if key in _RESOURCE_KEY_MAP:
        return _RESOURCE_KEY_MAP[key]
    stripped = key.rstrip("s")
    if stripped in _RESOURCE_KEY_MAP:
        return _RESOURCE_KEY_MAP[stripped]
    return None


# ── Main public function ──────────────────────────────────────────────────────

def plan_all_routes(
    image_meta:           dict,
    resource_assignments: dict,
    base_locations:       dict,
    blocked_masks:        Optional[dict] = None,
    use_real_osm:         bool           = True,
    flood_threshold:      float          = 0.45,
) -> list:
    """
    Plan routes for every resource assignment produced by the Resource Agent.

    Parameters
    ----------
    image_meta : dict  — GPS coverage info for the satellite image
        center_lat, center_lon  : GPS centre of image
        coverage_km             : km covered by image width
        width_px, height_px     : image dimensions in pixels

    resource_assignments : rescue_plan dict from allocate_rescue_resources_llm()
        e.g. {"Z35": {"boats": 2, "ambulances": 1}, "Z01": {"rescue_teams": 2}}

    base_locations : dict mapping singular resource type → location info
        e.g. {
            "ambulance":   {"name": "City Hospital",    "lat": 25.440, "lon": 81.840},
            "rescue_team": {"name": "Rescue Station A", "lat": 25.430, "lon": 81.855},
            "boat":        {"name": "Boat Depot",        "lat": 25.425, "lon": 81.848},
        }

    blocked_masks : optional dict of numpy arrays from Vision Agent
        e.g. {"flood": <float ndarray 0-1>}
        Float arrays are binarised at flood_threshold before road blocking.

    use_real_osm : bool
        True  → download real OSM road graph (needs internet + osmnx installed)
        False → use synthetic 3×3 grid graph (offline, good for testing)

    flood_threshold : float (default 0.45)
        Flood map pixels ≥ this threshold are treated as blocked roads.

    Returns
    -------
    List of route plan dicts, one per (resource_type × zone) pair.
    """

    print("\n" + "=" * 60)
    print("  ROUTE AGENT  —  Planning All Routes")
    print("=" * 60)

    # ── Step 1: Build geo transform ────────────────────────────────────────
    geo_transform = build_geo_transform(
        center_lat      = image_meta["center_lat"],
        center_lon      = image_meta["center_lon"],
        coverage_km     = image_meta["coverage_km"],
        image_width_px  = image_meta["width_px"],
        image_height_px = image_meta["height_px"],
    )
    print(f"[RouteAgent] Geo-transform ready  "
          f"top-left=({geo_transform['top_left_lat']:.5f}, "
          f"{geo_transform['top_left_lon']:.5f})  "
          f"coverage={image_meta['coverage_km']} km")

    # ── Step 2: Build road graph ───────────────────────────────────────────
    center_lat = image_meta["center_lat"]
    center_lon = image_meta["center_lon"]

    if use_real_osm and OSMNX_AVAILABLE:
        print("[RouteAgent] Mode: REAL OSM road network (online)")
        G = download_road_network(center_lat, center_lon, radius_m=5000)
        _nearest_fn = nearest_node
        print(f"[RouteAgent] Road graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    else:
        if use_real_osm and not OSMNX_AVAILABLE:
            print("[RouteAgent] WARNING: osmnx not installed — falling back to synthetic graph")
        else:
            print("[RouteAgent] Mode: SYNTHETIC road graph (offline)")
        G = build_synthetic_graph(center_lat=center_lat, center_lon=center_lon)
        _nearest_fn = nearest_node_synthetic
        print(f"[RouteAgent] Synthetic graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # ── Step 3: Apply flood/hazard masks from Vision Agent ─────────────────
    if blocked_masks:
        all_blocked_polygons = []
        for mask_type, mask_array in blocked_masks.items():
            if mask_array is None or not isinstance(mask_array, np.ndarray):
                continue

            if mask_array.dtype in (np.float32, np.float64, float):
                binary       = (mask_array >= flood_threshold)
                frac_blocked = binary.mean() * 100
                print(f"[RouteAgent] '{mask_type}' mask: "
                      f"threshold={flood_threshold}  {frac_blocked:.1f}% of pixels blocked")
            else:
                binary = mask_array.astype(bool)

            polys = mask_to_polygons(binary, geo_transform)
            print(f"[RouteAgent] '{mask_type}' → {len(polys)} blocked polygon(s)")
            all_blocked_polygons.extend(polys)

        if all_blocked_polygons:
            G = remove_blocked_roads(G, all_blocked_polygons)

    # ── Step 4: Route every resource to every zone ─────────────────────────
    all_routes = []

    zone_count = len(resource_assignments)
    print(f"\n[RouteAgent] Routing resources across {zone_count} zone(s) ...\n")

    for zone_name, assignments in resource_assignments.items():

        try:
            dest_lat, dest_lon = get_zone_latlon(zone_name, geo_transform)
        except (ValueError, KeyError) as e:
            print(f"[RouteAgent] WARNING: Cannot resolve zone '{zone_name}': {e} — skipping")
            continue

        dest_node = _nearest_fn(G, dest_lat, dest_lon)
        print(f"  Zone {zone_name}  dest=({dest_lat:.5f}, {dest_lon:.5f})  node={dest_node}")

        for resource_type, count in assignments.items():
            if not count:
                continue

            lookup_key = _canonical_resource_key(resource_type)
            if lookup_key is None or lookup_key not in base_locations:
                if resource_type in base_locations:
                    lookup_key = resource_type
                else:
                    print(f"    [WARN] No base location for '{resource_type}' "
                          f"(normalised='{lookup_key}')  "
                          f"available={list(base_locations.keys())} — skipping")
                    continue

            base        = base_locations[lookup_key]
            origin_node = _nearest_fn(G, base["lat"], base["lon"])

            print(f"    {count}× {resource_type:<14}  "
                  f"'{base['name']}'  node={origin_node}  →  {zone_name}  node={dest_node}")

            route_result = find_route(G, origin_node, dest_node)
            waypoints    = (path_to_waypoints(G, route_result["node_path"])
                            if route_result["success"] else [])

            plan = build_route_plan(
                zone_name          = zone_name,
                resource_type      = resource_type,
                origin_name        = base["name"],
                destination_latlon = (dest_lat, dest_lon),
                route_result       = route_result,
                waypoints          = waypoints,
            )
            plan["unit_count"] = count

            if route_result["success"]:
                print(f"      ✓  {plan['distance_km']} km  "
                      f"ETA={plan['eta_minutes']} min  "
                      f"waypoints={len(waypoints)}  "
                      f"blocked_avoided={route_result.get('blocked_avoided', 0)}")
            else:
                print(f"      ✗  FAILED: {route_result['error']}")

            all_routes.append(plan)

    successful = sum(1 for r in all_routes if r.get("success"))
    failed     = len(all_routes) - successful
    print(f"\n[RouteAgent] Complete — {successful} route(s) successful"
          + (f", {failed} failed" if failed else "") + "\n")
    return all_routes


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_routes(routes: list):
    """Print a detailed summary of all planned routes to the terminal."""
    print("\n" + "=" * 60)
    print("  ROUTE PLANS  —  Full Summary")
    print("=" * 60)

    for r in routes:
        tick = "✓" if r.get("success") else "✗"
        print(f"\n  [{tick}] {r.get('unit_count', 1)}× {r['resource_type']}"
              f"  →  Zone {r['zone']}")
        print(f"       Origin      : {r['origin_name']}")
        print(f"       Destination : {r['destination_latlon']}")

        if r.get("success"):
            print(f"       Distance    : {r['distance_km']} km")
            print(f"       ETA         : {r['eta_minutes']} min")
            print(f"       Waypoints   : {len(r['waypoints'])}")
            for i, (lat, lon) in enumerate(r["waypoints"]):
                print(f"                     {i+1:02d}. ({lat:.6f}, {lon:.6f})")
        else:
            print(f"       ERROR       : {r.get('error')}")

    print()