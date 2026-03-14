"""
test_route_agent.py
-------------------
Tests for the Route Agent.  ALL tests use the synthetic graph —
no internet, no OSMnx downloads, runs instantly.

Run:
    pytest test_route_agent.py -v

Or run the full demo directly:
    python test_route_agent.py

Tests are grouped as:
  Unit tests   — test one function at a time
  Integration  — test the full plan_all_routes() pipeline
"""

import sys
import os
import math
import numpy as np

# ── path setup so we can import agents.route_agent ────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── imports ────────────────────────────────────────────────────────────────
from agents.route_agent.geo_reference    import build_geo_transform, pixel_to_latlon
from agents.route_agent.zone_coordinates import parse_zone_name, get_zone_latlon, get_all_zone_coordinates
from agents.route_agent.road_network     import (
    build_synthetic_graph,
    nearest_node_synthetic,
    remove_blocked_roads,
)
from agents.route_agent.router           import find_route, path_to_waypoints, build_route_plan, reroute
from agents.route_agent.route_agent      import plan_all_routes, print_routes


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED FIXTURES  (reused across tests)
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_META = {
    "center_lat":  25.435,
    "center_lon":  81.846,
    "coverage_km": 5,
    "width_px":    640,
    "height_px":   640,
}

BASE_LOCATIONS = {
    "ambulance":   {"name": "City Hospital",    "lat": 25.440, "lon": 81.840},
    "rescue_team": {"name": "Rescue Station A", "lat": 25.430, "lon": 81.855},
    "boat":        {"name": "Boat Depot",        "lat": 25.425, "lon": 81.848},
}

RESOURCE_ASSIGNMENTS = {
    "Z12": {"ambulances": 2, "rescue_teams": 1, "boats": 0},
    "Z34": {"ambulances": 0, "rescue_teams": 2, "boats": 1},
    "Z55": {"ambulances": 1, "rescue_teams": 0, "boats": 0},
}

# ─────────────────────────────────────────────────────────────────────────────
#  UNIT TEST 1 — geo_reference.py
# ─────────────────────────────────────────────────────────────────────────────

def test_geo_transform_build():
    """build_geo_transform should return all expected keys."""
    t = build_geo_transform(25.435, 81.846, 5.0, 640, 640)
    required_keys = ["top_left_lat", "top_left_lon",
                     "lat_per_pixel", "lon_per_pixel",
                     "image_width_px", "image_height_px"]
    for key in required_keys:
        assert key in t, f"Missing key: {key}"
    print("  [PASS] test_geo_transform_build")


def test_pixel_to_latlon_center():
    """Center pixel should map back to approximately the center lat/lon."""
    t = build_geo_transform(25.435, 81.846, 5.0, 640, 640)
    lat, lon = pixel_to_latlon(320, 320, t)   # centre of a 640×640 image
    assert abs(lat - 25.435) < 0.01, f"Center lat off: {lat}"
    assert abs(lon - 81.846) < 0.01, f"Center lon off: {lon}"
    print(f"  [PASS] test_pixel_to_latlon_center  →  ({lat}, {lon})")


def test_pixel_to_latlon_top_left():
    """Top-left pixel (0,0) should be north-west of center."""
    t = build_geo_transform(25.435, 81.846, 5.0, 640, 640)
    lat, lon = pixel_to_latlon(0, 0, t)
    assert lat > 25.435, "Top-left should be north (higher lat)"
    assert lon < 81.846, "Top-left should be west  (lower lon)"
    print(f"  [PASS] test_pixel_to_latlon_top_left  →  ({lat}, {lon})")


# ─────────────────────────────────────────────────────────────────────────────
#  UNIT TEST 2 — zone_coordinates.py
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_zone_name():
    """Zone name parser should handle all valid formats."""
    assert parse_zone_name("Z12")   == (1, 2)
    assert parse_zone_name("Z34")   == (3, 4)
    assert parse_zone_name("Z1_10") == (1, 10)
    assert parse_zone_name("Z10_5") == (10, 5)
    print("  [PASS] test_parse_zone_name")


def test_zone_latlon_within_image():
    """Every zone center should fall inside the image's lat/lon bounding box."""
    t       = build_geo_transform(25.435, 81.846, 5.0, 640, 640)
    all_zones = get_all_zone_coordinates(t)

    assert len(all_zones) == 100, f"Expected 100 zones, got {len(all_zones)}"

    for name, (lat, lon) in all_zones.items():
        assert t["top_left_lat"] >= lat >= (t["top_left_lat"] - t["lat_per_pixel"] * 640), \
            f"Zone {name} lat {lat} out of bounds"
        assert t["top_left_lon"] <= lon <= (t["top_left_lon"] + t["lon_per_pixel"] * 640), \
            f"Zone {name} lon {lon} out of bounds"

    print(f"  [PASS] test_zone_latlon_within_image  ({len(all_zones)} zones checked)")


# ─────────────────────────────────────────────────────────────────────────────
#  UNIT TEST 3 — road_network.py
# ─────────────────────────────────────────────────────────────────────────────

def test_synthetic_graph_structure():
    """Synthetic graph should have 9 nodes and expected edges."""
    G = build_synthetic_graph()
    assert len(G.nodes) == 9,   f"Expected 9 nodes, got {len(G.nodes)}"
    assert len(G.edges) >= 18,  f"Expected ≥18 directed edges, got {len(G.edges)}"

    # every edge should have travel_time
    for u, v, data in G.edges(data=True):
        assert "travel_time" in data, f"Edge {u}→{v} missing travel_time"
        assert data["travel_time"] > 0

    print(f"  [PASS] test_synthetic_graph_structure  "
          f"({len(G.nodes)} nodes, {len(G.edges)} edges)")


def test_nearest_node():
    """nearest_node_synthetic should find the closest node."""
    G = build_synthetic_graph()
    # node 0 is at (25.445, 81.846) — top-left corner
    node = nearest_node_synthetic(G, 25.445, 81.846)
    assert node == 0, f"Expected node 0, got {node}"
    print(f"  [PASS] test_nearest_node  →  node {node}")


def test_block_roads():
    """Blocking an edge should set its travel_time to 1e9."""
    from shapely.geometry import Polygon as ShapelyPolygon

    G = build_synthetic_graph()

    # build a polygon that covers the 0→1 edge (top edge of graph)
    # node 0 ≈ (25.445, 81.846), node 1 ≈ (25.445, 81.851)
    block_poly = ShapelyPolygon([
        (81.843, 25.443), (81.854, 25.443),
        (81.854, 25.447), (81.843, 25.447),
    ])

    G_blocked = remove_blocked_roads(G, [block_poly])

    # edge 0→1 should now be very expensive
    edge_time = G_blocked[0][1][0]["travel_time"]
    assert edge_time >= 1e8, f"Expected blocked edge, got travel_time={edge_time}"
    print(f"  [PASS] test_block_roads  →  edge 0→1 travel_time = {edge_time:.0e}")


# ─────────────────────────────────────────────────────────────────────────────
#  UNIT TEST 4 — router.py
# ─────────────────────────────────────────────────────────────────────────────

def test_find_route_basic():
    """Route from node 0 to node 8 should succeed and have sensible metrics."""
    G = build_synthetic_graph()
    result = find_route(G, origin_node=0, dest_node=8)

    assert result["success"],           f"Route failed: {result['error']}"
    assert len(result["node_path"]) >= 2
    assert result["distance_km"] > 0
    assert result["eta_minutes"] > 0

    print(f"  [PASS] test_find_route_basic  →  "
          f"{result['distance_km']} km, {result['eta_minutes']} min, "
          f"path={result['node_path']}")


def test_find_route_same_node():
    """Routing from a node to itself should return a single-node path."""
    G = build_synthetic_graph()
    result = find_route(G, origin_node=4, dest_node=4)
    assert result["success"]
    assert result["node_path"] == [4]
    assert result["distance_km"] == 0.0
    print("  [PASS] test_find_route_same_node")


def test_reroute_after_blockage():
    """After blocking a key road, reroute should find an alternative path."""
    G = build_synthetic_graph()

    # get original path from 0 to 8
    original = find_route(G, 0, 8)
    original_path = original["node_path"]

    # block the first road on the original path
    u, v = original_path[0], original_path[1]
    updated = reroute(G, current_node=0, dest_node=8,
                      newly_blocked_edges=[(u, v)])

    assert updated["success"], f"Reroute failed: {updated['error']}"
    print(f"  [PASS] test_reroute_after_blockage  →  "
          f"new path: {updated['node_path']}")


def test_path_to_waypoints():
    """Waypoints should be (lat, lon) tuples for every node in path."""
    G = build_synthetic_graph()
    result = find_route(G, 0, 8)
    waypoints = path_to_waypoints(G, result["node_path"])

    assert len(waypoints) == len(result["node_path"])
    for lat, lon in waypoints:
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert 25.0 < lat < 26.0,  f"Lat out of expected range: {lat}"
        assert 81.0 < lon < 82.0,  f"Lon out of expected range: {lon}"

    print(f"  [PASS] test_path_to_waypoints  →  {waypoints}")


# ─────────────────────────────────────────────────────────────────────────────
#  INTEGRATION TEST — plan_all_routes()
# ─────────────────────────────────────────────────────────────────────────────

def test_plan_all_routes_structure():
    """Full pipeline should return a list of properly structured route plans."""
    routes = plan_all_routes(
        image_meta           = IMAGE_META,
        resource_assignments = RESOURCE_ASSIGNMENTS,
        base_locations       = BASE_LOCATIONS,
        blocked_masks        = None,
        use_real_osm         = False,
    )

    # count expected routes (only non-zero counts)
    expected = 0
    for zone, assignments in RESOURCE_ASSIGNMENTS.items():
        for rtype, count in assignments.items():
            if count > 0:
                # account for plural/singular normalization
                singular = rtype.rstrip("s")
                if rtype in BASE_LOCATIONS or singular in BASE_LOCATIONS:
                    expected += 1

    assert len(routes) == expected, \
        f"Expected {expected} route plans, got {len(routes)}"

    required_keys = ["zone", "resource_type", "origin", "destination_latlon",
                     "success", "waypoints", "distance_km", "eta_minutes",
                     "unit_count"]
    for r in routes:
        for key in required_keys:
            assert key in r, f"Route plan missing key '{key}': {r}"

    print(f"  [PASS] test_plan_all_routes_structure  →  {len(routes)} routes")


def test_plan_with_flood_mask():
    """Full pipeline with a flood mask should still produce routes."""
    # fake flood mask: top quarter of image is flooded
    flood_mask = np.zeros((640, 640), dtype=bool)
    flood_mask[:160, :] = True     # top 160 rows blocked

    routes = plan_all_routes(
        image_meta           = IMAGE_META,
        resource_assignments = {"Z12": {"ambulances": 1, "rescue_teams": 0, "boats": 0}},
        base_locations       = BASE_LOCATIONS,
        blocked_masks        = {"flood": flood_mask},
        use_real_osm         = False,
    )

    assert len(routes) >= 1
    print(f"  [PASS] test_plan_with_flood_mask  →  {len(routes)} route(s) returned")


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO SCENARIO  (run with `python test_route_agent.py`)
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "█"*60)
    print("  ROUTE AGENT — FULL DEMO  (Flood + Building Collapse Scenario)")
    print("█"*60)

    image_meta = {
        "center_lat":  25.435,
        "center_lon":  81.846,
        "coverage_km": 5,
        "width_px":    640,
        "height_px":   640,
    }

    # simulate: resource agent output
    resource_assignments = {
        "Z12": {"ambulances": 2, "rescue_teams": 1, "boats": 1},
        "Z34": {"ambulances": 1, "rescue_teams": 2, "boats": 0},
        "Z78": {"ambulances": 0, "rescue_teams": 1, "boats": 2},
    }

    base_locations = {
        "ambulances":    {"name": "District Hospital",  "lat": 25.440, "lon": 81.840},
        "rescue_teams":  {"name": "NDRF Station",       "lat": 25.430, "lon": 81.855},
        "boats":         {"name": "River Boat Depot",   "lat": 25.425, "lon": 81.848},
    }

    # simulate: Vision Agent flood mask (bottom-left quadrant flooded)
    flood_mask = np.zeros((640, 640), dtype=bool)
    flood_mask[320:, :320] = True

    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = base_locations,
        blocked_masks        = {"flood": flood_mask},
        use_real_osm         = False,
    )

    print_routes(routes)


# ─────────────────────────────────────────────────────────────────────────────
#  PYTEST RUNNER + DIRECT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    # unit: geo_reference
    test_geo_transform_build,
    test_pixel_to_latlon_center,
    test_pixel_to_latlon_top_left,
    # unit: zone_coordinates
    test_parse_zone_name,
    test_zone_latlon_within_image,
    # unit: road_network
    test_synthetic_graph_structure,
    test_nearest_node,
    test_block_roads,
    # unit: router
    test_find_route_basic,
    test_find_route_same_node,
    test_reroute_after_blockage,
    test_path_to_waypoints,
    # integration
    test_plan_all_routes_structure,
    test_plan_with_flood_mask,
]

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RUNNING ALL ROUTE AGENT TESTS")
    print("="*60 + "\n")

    passed = 0
    failed = 0

    for test_fn in ALL_TESTS:
        print(f"→ {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results:  {passed} passed  |  {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n  All tests passed! Running demo …\n")
        run_demo()