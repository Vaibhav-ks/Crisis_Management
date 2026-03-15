"""
route_agent.py
--------------
MAIN ENTRY POINT for the Route Agent.

Called as  plan_all_routes(...)  from master_nodes.py after admin approval.

What this does:
  1. Build a geo-transform from image metadata (pixel ↔ GPS conversion)
  2. Convert each zone name → lat/lon destination on the ground
  3. Download (or load cached) real OSM road network, or use synthetic graph
  4. Apply blocked-road masks from the Vision Agent (flood prob map)
  5. Route each resource assignment from the Resource Agent (rescue_plan)
  6. Return a list of route plan dicts ready for communication/display

═══════════════════════════════════════════════════════════════════════════════
WHAT THE VISION AGENT SENDS (zone_map output)
═══════════════════════════════════════════════════════════════════════════════

vision_agent.analyze_image() returns:
{
    "zone_map": {
        "Z00": {"flood_score": 0.12, "damage_score": 0.0,  "severity": 0.072},
        "Z01": {"flood_score": 0.80, "damage_score": 0.45, "severity": 0.66},
        ...  (100 zones total: Z00 to Z99, 0-based row/col)
    }
}

Additionally, detect_flood() returns a raw float numpy array (H×W, values 0–1)
which the route agent uses as blocked_masks["flood"] to remove flooded roads.

═══════════════════════════════════════════════════════════════════════════════
WHAT THE RESOURCE AGENT SENDS (rescue_plan output)
═══════════════════════════════════════════════════════════════════════════════

rescue_decision_llm.allocate_rescue_resources_llm() returns:
{
    "Z35": {"boats": 2, "ambulances": 1, "rescue_teams": 2},
    "Z01": {"boats": 0, "ambulances": 1, "rescue_teams": 1},
    ...
}
Note: The LLM sometimes returns singular keys ("boat", "ambulance") instead
of plural ("boats", "ambulances"). The route agent handles both.

═══════════════════════════════════════════════════════════════════════════════
BUGS FOUND AND FIXED IN THIS FILE
═══════════════════════════════════════════════════════════════════════════════

BUG 1 ─ Zone name format mismatch (0-based vs 1-based)
  Vision Agent grid_mapper.py:  zone_id = f"Z{gy}{gx}"  → "Z00", "Z35", "Z99"
  Old zone_coordinates.py expected 1-based names like "Z11", "Z35" would mean
  row=3, col=5 in 0-based but zone_center_pixels computed it as row=3-1=2 in
  1-based, shifting every destination by one full grid cell.
  Fix: zone_coordinates.py rewritten to use 0-based indexing throughout.
       Also wrapped in try/except here so a bad LLM zone name doesn't crash
       the whole route planning pass.

BUG 2 ─ flood_prob_map is a float array, not a bool mask
  detect_flood() returns values 0.0–1.0. Passing it directly to
  mask_to_polygons → np.where(float_array) blocks ANY road near a pixel
  with even 0.001 flood probability. Rivers and low-lying areas become
  completely blocked even when totally passable.
  Fix: binarise the flood map with a configurable threshold (default 0.45)
       before passing to mask_to_polygons. Only zones with flood_score ≥
       threshold are treated as road blockages.

BUG 3 ─ LLM returns plural resource keys but base_locations uses singular
  LLM output: {"boats": 2, "ambulances": 1, "rescue_teams": 2}
  base_locations keys: "boat", "ambulance", "rescue_team"
  Old code used rstrip("s") which is fragile:
    "rescue_teams".rstrip("s") → "rescue_team" ✓ (lucky)
    "ambulances".rstrip("s")   → "ambulance"   ✓ (lucky)
    "boats".rstrip("s")        → "boat"         ✓ (lucky)
    Any new resource type might break silently.
  Fix: explicit lookup table _RESOURCE_KEY_MAP covering known variants.
       If a key is not in the table, we still try rstrip("s") as final fallback.

BUG 4 ─ route_planner_node was commented out in master_graph.py
  After admin_resource approval the graph went to END without ever calling
  plan_all_routes(). The route_plan field in MasterState was never populated.
  Fix: route_planner_node and admin_route_node are now implemented and
       uncommented in master_nodes.py and master_graph.py (see those files).

BUG 5 ─ image_meta not passed through pipeline
  plan_all_routes() needs image_meta (center_lat, center_lon, coverage_km,
  width_px, height_px) but master_nodes.py never stored or forwarded it.
  The vision_node receives only an image path — it doesn't know the GPS
  coverage of that image.
  Fix: image_meta is now a required field in MasterState. The caller
  (run_system.py or tests) must provide it alongside satellite_image.
  Default Prayagraj values are used when not supplied.
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
# BUG 3 fix: explicit table instead of fragile rstrip("s").

_RESOURCE_KEY_MAP = {
    # Plural form (LLM output)  →  singular (base_locations key)
    "boats":         "boat",
    "ambulances":    "ambulance",
    "rescue_teams":  "rescue_team",
    "helicopters":   "helicopter",
    "trucks":        "truck",
    "fire_trucks":   "fire_truck",
    # Also accept singular directly (LLM sometimes omits the plural)
    "boat":          "boat",
    "ambulance":     "ambulance",
    "rescue_team":   "rescue_team",
    "helicopter":    "helicopter",
    "truck":         "truck",
    "fire_truck":    "fire_truck",
}


def _canonical_resource_key(resource_type: str) -> Optional[str]:
    """Return the canonical singular key, or None if unknown."""
    key = resource_type.lower().strip()
    if key in _RESOURCE_KEY_MAP:
        return _RESOURCE_KEY_MAP[key]
    # Final fallback: strip trailing 's' (handles rare edge cases)
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
    use_real_osm:         bool           = False,
    flood_threshold:      float          = 0.45,
) -> list:
    """
    Plan routes for every resource assignment produced by the Resource Agent.

    Parameters
    ----------
    image_meta : dict with keys:
        center_lat   float  — GPS latitude  of the image centre
        center_lon   float  — GPS longitude of the image centre
        coverage_km  float  — kilometres covered by the image width
        width_px     int    — image width  in pixels
        height_px    int    — image height in pixels

    resource_assignments : rescue_plan from allocate_rescue_resources_llm()
        e.g. {"Z35": {"boats": 2, "ambulances": 1}, "Z01": {"rescue_teams": 2}}

    base_locations : dict mapping singular resource type → location info
        e.g. {
            "ambulance":   {"name": "City Hospital",    "lat": 25.440, "lon": 81.840},
            "rescue_team": {"name": "Rescue Station A", "lat": 25.430, "lon": 81.855},
            "boat":        {"name": "Boat Depot",        "lat": 25.425, "lon": 81.848},
        }

    blocked_masks : optional dict of numpy arrays from Vision Agent
        e.g. {"flood": <float ndarray 0-1 from detect_flood()>}
        BUG 2 fix: float arrays are binarised here at flood_threshold before use.

    use_real_osm : bool
        True  → download real OSM road graph (needs internet + osmnx)
        False → use synthetic 3×3 grid graph (offline safe, good for testing)

    flood_threshold : float (default 0.45)
        Pixels in the flood map with value ≥ this are treated as blocked roads.
        BUG 2 fix: prevents near-zero flood values from blocking all roads.

    Returns
    -------
    List of route plan dicts, one per (resource_type × zone) pair.
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
    print(f"[RouteAgent] Geo-transform ready. "
          f"Top-left=({geo_transform['top_left_lat']:.4f}, "
          f"{geo_transform['top_left_lon']:.4f}), "
          f"coverage={image_meta['coverage_km']} km")

    # ── Step 2: Build road graph ───────────────────────────────────────────
    center_lat = image_meta["center_lat"]
    center_lon = image_meta["center_lon"]

    if use_real_osm and OSMNX_AVAILABLE:
        print("[RouteAgent] Using REAL OSM road graph (online mode).")
        G = download_road_network(center_lat, center_lon, radius_m=5000)
        _nearest_fn = nearest_node
    else:
        if use_real_osm and not OSMNX_AVAILABLE:
            print("[RouteAgent] WARNING: use_real_osm=True but osmnx not installed. "
                  "Falling back to synthetic graph.")
        else:
            print("[RouteAgent] Using SYNTHETIC road graph (offline mode).")
        # BUG FIX: pass actual center coords so graph is built around the real area,
        # not hardcoded Prayagraj. Without this, every city snaps all nodes to one
        # point and every route comes back as 0 km.
        G = build_synthetic_graph(center_lat=center_lat, center_lon=center_lon)
        _nearest_fn = nearest_node_synthetic

    # ── Step 3: Apply blocked masks from Vision Agent ──────────────────────
    # BUG 2 fix: binarise float probability maps before building polygons
    if blocked_masks:
        all_blocked_polygons = []
        for mask_type, mask_array in blocked_masks.items():
            if mask_array is None or not isinstance(mask_array, np.ndarray):
                continue

            # Binarise float map: values ≥ flood_threshold are "blocked"
            if mask_array.dtype in (np.float32, np.float64, float):
                binary = (mask_array >= flood_threshold)
                frac_blocked = binary.mean() * 100
                print(f"[RouteAgent] '{mask_type}' mask: threshold={flood_threshold}, "
                      f"{frac_blocked:.1f}% of pixels blocked")
            else:
                binary = mask_array.astype(bool)

            polys = mask_to_polygons(binary, geo_transform)
            print(f"[RouteAgent] '{mask_type}' → {len(polys)} blocked polygon(s)")
            all_blocked_polygons.extend(polys)

        if all_blocked_polygons:
            G = remove_blocked_roads(G, all_blocked_polygons)

    # ── Step 4: Route every resource to every zone ─────────────────────────
    all_routes = []

    for zone_name, assignments in resource_assignments.items():

        # BUG 1 fix: parse 0-based zone names; skip on error rather than crash
        try:
            dest_lat, dest_lon = get_zone_latlon(zone_name, geo_transform)
        except (ValueError, KeyError) as e:
            print(f"\n[RouteAgent] WARNING: Cannot resolve zone '{zone_name}': {e} — skipping.")
            continue

        dest_node = _nearest_fn(G, dest_lat, dest_lon)
        print(f"\n[RouteAgent] Zone {zone_name} → GPS ({dest_lat:.5f}, {dest_lon:.5f}), "
              f"road node {dest_node}")

        for resource_type, count in assignments.items():
            if not count:
                continue

            # BUG 3 fix: normalise LLM resource key → base_locations key
            lookup_key = _canonical_resource_key(resource_type)
            if lookup_key is None or lookup_key not in base_locations:
                # Try raw key as last resort
                if resource_type in base_locations:
                    lookup_key = resource_type
                else:
                    print(f"  [WARN] No base location for '{resource_type}' "
                          f"(canonical='{lookup_key}'). "
                          f"Available: {list(base_locations.keys())}. Skipping.")
                    continue

            base        = base_locations[lookup_key]
            origin_node = _nearest_fn(G, base["lat"], base["lon"])

            print(f"  Routing {count}× {resource_type}  "
                  f"'{base['name']}' (node {origin_node}) → {zone_name} (node {dest_node})")

            # Dijkstra shortest path (travel_time as edge weight)
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
                print(f"    ✓ {plan['distance_km']} km  |  "
                      f"ETA {plan['eta_minutes']} min  |  "
                      f"{len(waypoints)} waypoints")
            else:
                print(f"    ✗ FAILED: {route_result['error']}")

            all_routes.append(plan)

    print(f"\n[RouteAgent] Done. {len(all_routes)} route(s) planned.\n")
    return all_routes


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_routes(routes: list):
    """Print all route plans in a readable summary table."""
    print("\n" + "="*60)
    print("  ROUTE PLANS SUMMARY")
    print("="*60)
    for r in routes:
        status = "✓" if r["success"] else "✗"
        print(f"\n  [{status}] {r['unit_count']}× {r['resource_type']}  →  Zone {r['zone']}")
        print(f"       From        : {r['origin']}")
        print(f"       Destination : {r['destination_latlon']}")
        if r["success"]:
            print(f"       Distance    : {r['distance_km']} km")
            print(f"       ETA         : {r['eta_minutes']} min")
            print(f"       Waypoints   : {len(r['waypoints'])} nodes")
            print(f"       Blocked avoided: {r['blocked_roads_avoided']}")
        else:
            print(f"       ERROR: {r['error']}")