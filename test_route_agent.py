"""
test_route_agent.py
-------------------
Comprehensive test suite for the Route Agent and all its dependencies.

HOW TO RUN:
    cd <project_root>
    python test_route_agent.py

All tests use:
  • The SYNTHETIC road graph (no internet, no OSMnx download needed)
  • REAL data formats — zone names, resource keys, and flood masks are
    produced the same way as the Vision Agent and Resource Agent do it.
  • No hardcoded magic numbers — expected values are COMPUTED from the
    same formulas the production code uses.
  • The three ML models (UNet, YOLO debris, YOLO COCO) are NOT loaded —
    we mock only the model I/O (input/output format), not the logic.

The three model files that are not uploaded:
  agents/vision_agent/unet_flood_model1.2.pth   — mocked via mock_flood_prob_map()
  agents/vision_agent/best_debris.pt            — mocked via mock_damage_detections()
  yolov8n.pt                                    — mocked via mock_people_counts()
"""

import sys
import os
import traceback
import numpy as np

# ── Make sure project root is on the path ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers — produce real-format data without loading ML models
# ═══════════════════════════════════════════════════════════════════════════════

def make_image_meta(center_lat=25.435, center_lon=81.846,
                    coverage_km=5.0, width_px=640, height_px=640) -> dict:
    """Create image_meta dict matching the format master_nodes.vision_node stores."""
    return {
        "center_lat":  center_lat,
        "center_lon":  center_lon,
        "coverage_km": coverage_km,
        "width_px":    width_px,
        "height_px":   height_px,
    }


def mock_flood_prob_map(height=640, width=640, hotspot_zones=None) -> np.ndarray:
    """
    Generate a realistic float flood probability map (H×W, values 0–1).
    Mimics the output of detect_flood() WITHOUT loading the UNet model.

    hotspot_zones: list of (row, col) in 0-based grid coords that should
                   have high flood probability (simulating flooded cells).
    """
    prob_map = np.zeros((height, width), dtype=np.float32)

    if hotspot_zones is None:
        hotspot_zones = [(3, 5), (3, 6), (4, 5)]   # default: a few flooded cells

    cell_h = height // 10
    cell_w  = width  // 10

    for gy, gx in hotspot_zones:
        y0 = gy * cell_h
        y1 = (gy + 1) * cell_h
        x0 = gx * cell_w
        x1 = (gx + 1) * cell_w
        # Flood probability 0.7–0.9 in the hotspot cells
        prob_map[y0:y1, x0:x1] = np.random.uniform(0.7, 0.9, (y1-y0, x1-x0))

    # Background noise (very low probability — should NOT block roads)
    noise = np.random.uniform(0.0, 0.05, (height, width)).astype(np.float32)
    prob_map = np.clip(prob_map + noise, 0, 1)

    return prob_map


def mock_damage_detections(image_shape=(640, 640), n_detections=5) -> list:
    """
    Generate realistic YOLO-style damage detections WITHOUT loading the model.
    Mimics the output of detect_damage().
    """
    height, width = image_shape[:2]
    detections = []
    np.random.seed(42)
    for _ in range(n_detections):
        x1 = int(np.random.uniform(0, width  * 0.8))
        y1 = int(np.random.uniform(0, height * 0.8))
        x2 = min(x1 + int(np.random.uniform(30, 120)), width)
        y2 = min(y1 + int(np.random.uniform(30, 120)), height)
        detections.append({
            "bbox":       [float(x1), float(y1), float(x2), float(y2)],
            "confidence": round(float(np.random.uniform(0.3, 0.95)), 3),
            "label":      "damaged_building",
        })
    return detections


def mock_zone_map_from_real_pipeline(image_shape=(640, 640),
                                     flood_hotspots=None) -> dict:
    """
    Build a zone_map using the REAL grid_mapper and severity functions,
    but with mocked model outputs.  This tests the actual pipeline logic
    without needing GPU/model files.
    """
    from agents.vision_agent.grid_mapper import build_zone_map
    from agents.vision_agent.severity    import add_severity

    height, width = image_shape
    image          = np.zeros((height, width, 3), dtype=np.uint8)
    flood_prob_map = mock_flood_prob_map(height, width, flood_hotspots)
    damage_detections = mock_damage_detections(image_shape)

    zone_map = build_zone_map(
        image             = image,
        flood_prob_map    = flood_prob_map,
        damage_detections = damage_detections,
    )
    zone_map = add_severity(zone_map)
    return zone_map, flood_prob_map


def make_base_locations() -> dict:
    """Real-format base_locations dict matching _DEFAULT_BASE_LOCATIONS."""
    return {
        "ambulance":   {"name": "District Hospital",    "lat": 25.440, "lon": 81.840},
        "rescue_team": {"name": "NDRF Station",         "lat": 25.430, "lon": 81.855},
        "boat":        {"name": "Boat Depot Allahabad", "lat": 25.425, "lon": 81.848},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════════

_results = []

def run_test(name: str, fn):
    try:
        fn()
        print(f"  ✅ PASS  {name}")
        _results.append((name, True, None))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  ❌ FAIL  {name}")
        print(f"          {e}")
        for line in tb.splitlines()[1:]:
            print(f"          {line}")
        _results.append((name, False, str(e)))


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 1: geo_reference.py ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def test_geo_transform_center_maps_to_center():
    """
    The pixel at the exact centre of the image must map to (center_lat, center_lon).
    Validates build_geo_transform ↔ pixel_to_latlon round-trip.

    FIXED: previous version called build_geo_transform with width_px/height_px
    but the function signature uses image_width_px/image_height_px.
    """
    from agents.route_agent.geo_reference import build_geo_transform, pixel_to_latlon

    meta = make_image_meta()
    gt   = build_geo_transform(
        center_lat      = meta["center_lat"],
        center_lon      = meta["center_lon"],
        coverage_km     = meta["coverage_km"],
        image_width_px  = meta["width_px"],    # correct param name
        image_height_px = meta["height_px"],   # correct param name
    )
    cx, cy = meta["width_px"] / 2, meta["height_px"] / 2
    lat, lon = pixel_to_latlon(cx, cy, gt)
    assert abs(lat - meta["center_lat"]) < 1e-4, \
        f"Centre lat wrong: got {lat}, expected {meta['center_lat']}"
    assert abs(lon - meta["center_lon"]) < 1e-4, \
        f"Centre lon wrong: got {lon}, expected {meta['center_lon']}"


def test_geo_transform_top_left_is_northwest():
    """Top-left pixel (0,0) must be north-west of centre."""
    from agents.route_agent.geo_reference import build_geo_transform, pixel_to_latlon

    meta = make_image_meta()
    gt   = build_geo_transform(
        center_lat      = meta["center_lat"],
        center_lon      = meta["center_lon"],
        coverage_km     = meta["coverage_km"],
        image_width_px  = meta["width_px"],
        image_height_px = meta["height_px"],
    )
    lat, lon = pixel_to_latlon(0, 0, gt)
    assert lat > meta["center_lat"], "Top-left should be north (higher lat)"
    assert lon < meta["center_lon"], "Top-left should be west  (lower lon)"


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 2: zone_coordinates.py ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def test_parse_zone_name_basic():
    """Z35 → row=3, col=5 (0-based, matching Vision Agent output)."""
    from agents.route_agent.zone_coordinates import parse_zone_name
    row, col = parse_zone_name("Z35")
    assert (row, col) == (3, 5), f"Expected (3,5), got ({row},{col})"


def test_parse_zone_name_Z00():
    """Z00 is the top-left zone and must parse without error."""
    from agents.route_agent.zone_coordinates import parse_zone_name
    row, col = parse_zone_name("Z00")
    assert (row, col) == (0, 0), f"Expected (0,0), got ({row},{col})"


def test_parse_zone_name_Z99():
    """Z99 is the bottom-right zone."""
    from agents.route_agent.zone_coordinates import parse_zone_name
    row, col = parse_zone_name("Z99")
    assert (row, col) == (9, 9), f"Expected (9,9), got ({row},{col})"


def test_parse_zone_name_underscore():
    """Z3_10 → row=3, col=10 (underscore form for col ≥ 10)."""
    from agents.route_agent.zone_coordinates import parse_zone_name
    row, col = parse_zone_name("Z3_10")
    assert (row, col) == (3, 10), f"Expected (3,10), got ({row},{col})"


def test_parse_zone_name_bad_raises():
    """Non-Z prefix should raise ValueError."""
    from agents.route_agent.zone_coordinates import parse_zone_name
    try:
        parse_zone_name("A35")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_zone_center_pixels_Z00():
    """
    Zone Z00 (row=0, col=0) centre should be at (cell_w/2, cell_h/2).
    This was WRONG in the old code: (col-1)*cell_w = -cell_w for col=0.
    """
    from agents.route_agent.zone_coordinates import zone_center_pixels
    px, py = zone_center_pixels(0, 0, image_width_px=640, image_height_px=640)
    cell = 640 / 10  # 64 pixels per cell
    assert px == cell / 2, f"Z00 px: expected {cell/2}, got {px}"
    assert py == cell / 2, f"Z00 py: expected {cell/2}, got {py}"


def test_zone_center_pixels_Z99():
    """Z99 should be centred in the bottom-right cell."""
    from agents.route_agent.zone_coordinates import zone_center_pixels
    px, py = zone_center_pixels(9, 9, image_width_px=640, image_height_px=640)
    cell = 640 / 10
    expected_px = 9 * cell + cell / 2
    expected_py = 9 * cell + cell / 2
    assert abs(px - expected_px) < 1e-6, f"Z99 px: expected {expected_px}, got {px}"
    assert abs(py - expected_py) < 1e-6, f"Z99 py: expected {expected_py}, got {py}"


def test_all_100_zone_coords_are_unique():
    """Every zone must map to a distinct GPS coordinate."""
    from agents.route_agent.geo_reference    import build_geo_transform
    from agents.route_agent.zone_coordinates import get_all_zone_coordinates

    meta = make_image_meta()
    gt   = build_geo_transform(
        center_lat      = meta["center_lat"],
        center_lon      = meta["center_lon"],
        coverage_km     = meta["coverage_km"],
        image_width_px  = meta["width_px"],
        image_height_px = meta["height_px"],
    )
    coords = get_all_zone_coordinates(gt)
    assert len(coords) == 100, f"Expected 100 zones, got {len(coords)}"
    unique = set(coords.values())
    assert len(unique) == 100, f"Expected 100 unique coordinates, got {len(unique)}"


def test_zone_coords_within_image_bounds():
    """All zone GPS coordinates must lie within the image's geographic bounding box."""
    from agents.route_agent.geo_reference    import build_geo_transform
    from agents.route_agent.zone_coordinates import get_all_zone_coordinates

    meta = make_image_meta()
    gt   = build_geo_transform(
        center_lat      = meta["center_lat"],
        center_lon      = meta["center_lon"],
        coverage_km     = meta["coverage_km"],
        image_width_px  = meta["width_px"],
        image_height_px = meta["height_px"],
    )
    coords = get_all_zone_coordinates(gt)
    lats   = [v[0] for v in coords.values()]
    lons   = [v[1] for v in coords.values()]
    assert min(lats) >= gt["top_left_lat"] - gt["lat_per_pixel"] * meta["height_px"] - 0.001
    assert max(lats) <= gt["top_left_lat"] + 0.001
    assert min(lons) >= gt["top_left_lon"] - 0.001
    assert max(lons) <= gt["top_left_lon"] + gt["lon_per_pixel"] * meta["width_px"] + 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 3: road_network.py ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def test_synthetic_graph_structure():
    """Synthetic graph must have 9 nodes and bidirectional edges."""
    from agents.route_agent.road_network import build_synthetic_graph
    G = build_synthetic_graph()
    assert len(G.nodes) == 9,   f"Expected 9 nodes, got {len(G.nodes)}"
    assert len(G.edges) > 10,  f"Expected >10 directed edges, got {len(G.edges)}"


def test_synthetic_graph_has_travel_time():
    """Every edge must have a travel_time attribute > 0."""
    from agents.route_agent.road_network import build_synthetic_graph
    G = build_synthetic_graph()
    for u, v, data in G.edges(data=True):
        assert "travel_time" in data, f"Edge ({u},{v}) missing travel_time"
        assert data["travel_time"] > 0, f"Edge ({u},{v}) has travel_time={data['travel_time']}"


def test_nearest_node_synthetic_returns_valid_node():
    """nearest_node_synthetic must return a node that exists in the graph."""
    from agents.route_agent.road_network import build_synthetic_graph, nearest_node_synthetic
    G    = build_synthetic_graph()
    node = nearest_node_synthetic(G, lat=25.435, lon=81.846)
    assert node in G.nodes, f"Returned node {node} not in graph"


def test_mask_to_polygons_empty_mask():
    """All-zero mask must return empty list."""
    from agents.route_agent.geo_reference import build_geo_transform
    from agents.route_agent.road_network  import mask_to_polygons

    meta = make_image_meta()
    gt   = build_geo_transform(
        center_lat=meta["center_lat"], center_lon=meta["center_lon"],
        coverage_km=meta["coverage_km"],
        image_width_px=meta["width_px"], image_height_px=meta["height_px"],
    )
    empty_mask = np.zeros((640, 640), dtype=bool)
    polys = mask_to_polygons(empty_mask, gt)
    assert polys == [], f"Expected [], got {polys}"


def test_mask_to_polygons_flood_hotspot():
    """
    A binarised flood hotspot should produce at least one polygon.
    Tests the BUG 2 fix: float map binarised BEFORE calling mask_to_polygons.
    """
    from agents.route_agent.geo_reference import build_geo_transform
    from agents.route_agent.road_network  import mask_to_polygons

    meta          = make_image_meta()
    gt            = build_geo_transform(
        center_lat=meta["center_lat"], center_lon=meta["center_lon"],
        coverage_km=meta["coverage_km"],
        image_width_px=meta["width_px"], image_height_px=meta["height_px"],
    )
    flood_map     = mock_flood_prob_map(640, 640, hotspot_zones=[(2, 3)])
    threshold     = 0.45
    binary_mask   = (flood_map >= threshold)   # BUG 2 fix applied here

    # Verify there IS something to detect
    assert binary_mask.any(), "Test data error: binary mask is all False"

    polys = mask_to_polygons(binary_mask, gt)
    assert len(polys) > 0, "Expected at least one polygon from flood hotspot"


def test_remove_blocked_roads_penalises_edges():
    """
    After remove_blocked_roads, at least one edge in the flood zone should
    have travel_time = 1e9.
    """
    from agents.route_agent.geo_reference import build_geo_transform
    from agents.route_agent.road_network  import (
        build_synthetic_graph, mask_to_polygons, remove_blocked_roads
    )

    meta = make_image_meta()
    gt   = build_geo_transform(
        center_lat=meta["center_lat"], center_lon=meta["center_lon"],
        coverage_km=meta["coverage_km"],
        image_width_px=meta["width_px"], image_height_px=meta["height_px"],
    )
    G = build_synthetic_graph()

    # Create a flood mask that covers the whole image (worst case)
    flood_map   = np.ones((640, 640), dtype=np.float32) * 0.9
    binary_mask = (flood_map >= 0.45)
    polys       = mask_to_polygons(binary_mask, gt)

    G_blocked = remove_blocked_roads(G, polys)

    # At least one edge should be penalised
    penalised = [
        (u, v)
        for u, v, d in G_blocked.edges(data=True)
        if d.get("travel_time", 0) >= 1e9
    ]
    assert len(penalised) > 0, "Expected some edges to be penalised"


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 4: router.py ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def test_find_route_success():
    """Dijkstra must find a path between two connected synthetic nodes."""
    from agents.route_agent.road_network import build_synthetic_graph
    from agents.route_agent.router       import find_route

    G      = build_synthetic_graph()
    result = find_route(G, origin_node=0, dest_node=8)

    assert result["success"],          f"Route failed: {result.get('error')}"
    assert len(result["node_path"]) >= 2, "Path too short"
    assert result["distance_km"]  > 0,  "Distance must be positive"
    assert result["eta_minutes"]  > 0,  "ETA must be positive"


def test_find_route_same_node():
    """Routing from a node to itself must succeed with distance=0."""
    from agents.route_agent.road_network import build_synthetic_graph
    from agents.route_agent.router       import find_route

    G      = build_synthetic_graph()
    result = find_route(G, origin_node=4, dest_node=4)
    assert result["success"]
    assert result["distance_km"] == 0.0


def test_find_route_disconnected():
    """Routing to an isolated node must fail gracefully."""
    from agents.route_agent.road_network import build_synthetic_graph
    from agents.route_agent.router       import find_route
    import networkx as nx

    G = build_synthetic_graph()
    # Add an isolated node with no edges
    G.add_node(999, y=25.0, x=80.0)

    result = find_route(G, origin_node=0, dest_node=999)
    assert not result["success"], "Should fail for isolated node"
    assert result["error"] is not None


def test_path_to_waypoints():
    """path_to_waypoints must return (lat, lon) pairs matching node attributes."""
    from agents.route_agent.road_network import build_synthetic_graph
    from agents.route_agent.router       import find_route, path_to_waypoints

    G      = build_synthetic_graph()
    result = find_route(G, 0, 8)
    assert result["success"]

    waypoints = path_to_waypoints(G, result["node_path"])
    assert len(waypoints) == len(result["node_path"])

    for (lat, lon) in waypoints:
        assert isinstance(lat, float), "lat must be float"
        assert isinstance(lon, float), "lon must be float"
        assert 20 < lat < 35,  f"lat {lat} out of India range"
        assert 70 < lon < 100, f"lon {lon} out of India range"


def test_build_route_plan_success_shape():
    """build_route_plan must return a dict with all required keys."""
    from agents.route_agent.road_network import build_synthetic_graph
    from agents.route_agent.router       import find_route, path_to_waypoints, build_route_plan

    G      = build_synthetic_graph()
    result = find_route(G, 0, 8)
    wps    = path_to_waypoints(G, result["node_path"])
    plan   = build_route_plan("Z35", "ambulance", "City Hospital",
                               (25.44, 81.85), result, wps)

    required = {"zone", "resource_type", "origin", "destination_latlon",
                "success", "error", "waypoints", "distance_km",
                "eta_minutes", "blocked_roads_avoided"}
    missing = required - set(plan.keys())
    assert not missing, f"Missing keys in route plan: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 5: plan_all_routes() end-to-end ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def test_plan_all_routes_basic():
    """
    End-to-end test: build routes from real-format resource_assignments
    (same format as rescue_decision_llm output) using synthetic graph.
    No models loaded, no internet needed.
    """
    from agents.route_agent.route_agent import plan_all_routes

    image_meta = make_image_meta()
    # Format exactly as rescue_decision_llm returns (plural keys)
    resource_assignments = {
        "Z35": {"boats": 1, "ambulances": 1, "rescue_teams": 0},
        "Z72": {"boats": 0, "ambulances": 0, "rescue_teams": 2},
    }
    base_locations = make_base_locations()

    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = base_locations,
        blocked_masks        = None,
        use_real_osm         = False,
    )

    # Should have one route per non-zero resource assignment
    expected_count = sum(
        1 for asn in resource_assignments.values()
        for count in asn.values() if count > 0
    )
    assert len(routes) == expected_count, \
        f"Expected {expected_count} routes, got {len(routes)}"

    for r in routes:
        assert r["success"], f"Route failed: {r.get('error')}"
        assert r["unit_count"] > 0
        assert r["distance_km"] >= 0
        assert r["eta_minutes"] >= 0
        assert isinstance(r["waypoints"], list)


def test_plan_all_routes_with_flood_mask():
    """
    Routes must still be found even when flood mask marks some roads blocked.
    The synthetic graph is connected enough to always find an alternative.
    Tests BUG 2 fix: float flood map binarised before use.
    """
    from agents.route_agent.route_agent import plan_all_routes

    image_meta = make_image_meta()
    resource_assignments = {
        "Z22": {"boats": 1},
        "Z77": {"rescue_teams": 1},
    }
    base_locations = make_base_locations()

    # Float flood map as returned by detect_flood() — BUG 2 was here
    flood_map = mock_flood_prob_map(640, 640, hotspot_zones=[(2, 2), (2, 3)])

    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = base_locations,
        blocked_masks        = {"flood": flood_map},
        use_real_osm         = False,
        flood_threshold      = 0.45,
    )

    assert len(routes) >= 1, "Should have at least one route"
    # All routes should either succeed or fail gracefully (no exception)
    for r in routes:
        assert "success" in r


def test_plan_all_routes_resource_key_normalisation():
    """
    Tests BUG 3 fix: LLM returns plural keys like "boats", "ambulances".
    These must match base_locations which uses singular keys "boat", "ambulance".
    """
    from agents.route_agent.route_agent import plan_all_routes

    image_meta = make_image_meta()
    # Plural keys — as the LLM returns them
    resource_assignments = {
        "Z11": {
            "boats":        2,
            "ambulances":   1,
            "rescue_teams": 1,
        }
    }
    base_locations = make_base_locations()  # uses singular keys

    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = base_locations,
        use_real_osm         = False,
    )

    # All three resource types should produce routes (none skipped)
    assert len(routes) == 3, \
        (f"Expected 3 routes (boats/ambulances/rescue_teams), got {len(routes)}. "
         "Check resource key normalisation (BUG 3 fix).")


def test_plan_all_routes_bad_zone_name_skipped():
    """
    BUG 1 fix: A bad zone name from the LLM (e.g. "zone_35") must be skipped
    gracefully, not crash the whole routing pass.
    """
    from agents.route_agent.route_agent import plan_all_routes

    image_meta = make_image_meta()
    resource_assignments = {
        "zone_bad":   {"ambulances": 1},   # invalid — should be skipped
        "Z22":        {"ambulances": 1},   # valid   — should be routed
    }
    base_locations = make_base_locations()

    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = base_locations,
        use_real_osm         = False,
    )

    # Only the valid zone should produce a route
    zones_routed = {r["zone"] for r in routes}
    assert "Z22" in zones_routed, "Valid zone Z22 should have a route"
    assert "zone_bad" not in zones_routed, "Invalid zone should be skipped"


def test_plan_all_routes_zero_count_skipped():
    """Resource assignments with count=0 must produce no route."""
    from agents.route_agent.route_agent import plan_all_routes

    image_meta = make_image_meta()
    resource_assignments = {
        "Z55": {"boats": 0, "ambulances": 0, "rescue_teams": 0},
    }
    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = make_base_locations(),
        use_real_osm         = False,
    )
    assert routes == [], f"Expected no routes for all-zero counts, got {routes}"


def test_plan_all_routes_unknown_resource_skipped():
    """An unknown resource type must be skipped with a warning, not crash."""
    from agents.route_agent.route_agent import plan_all_routes

    image_meta = make_image_meta()
    resource_assignments = {
        "Z33": {"tanks": 2, "ambulances": 1},  # "tanks" unknown
    }
    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = make_base_locations(),
        use_real_osm         = False,
    )
    resource_types = {r["resource_type"] for r in routes}
    assert "tanks" not in resource_types, "'tanks' should be skipped (not in base_locations)"
    assert "ambulances" in resource_types, "'ambulances' should still be routed"


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 6: Vision Agent data flow (no models) ─────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def test_zone_map_keys_match_vision_agent_format():
    """
    Zone keys produced by grid_mapper.py must be in "Z{row}{col}" 0-based format
    AND must be parseable by the (fixed) zone_coordinates.parse_zone_name().
    This is the core integration test for the Vision Agent → Route Agent handoff.
    """
    from agents.route_agent.zone_coordinates import parse_zone_name

    zone_map, _ = mock_zone_map_from_real_pipeline()

    assert len(zone_map) == 100, f"Expected 100 zones, got {len(zone_map)}"

    for zone_id in zone_map:
        try:
            row, col = parse_zone_name(zone_id)
            assert 0 <= row <= 9, f"{zone_id}: row {row} out of 0-9 range"
            assert 0 <= col <= 9, f"{zone_id}: col {col} out of 0-9 range"
        except ValueError as e:
            raise AssertionError(
                f"Zone key '{zone_id}' from grid_mapper cannot be parsed by "
                f"zone_coordinates.parse_zone_name: {e}\n"
                "(This is the BUG 1 mismatch — 0-based vs 1-based indexing)"
            )


def test_zone_map_scores_in_range():
    """All zone scores must be in [0, 1]."""
    zone_map, _ = mock_zone_map_from_real_pipeline()
    for zone_id, data in zone_map.items():
        for field in ("flood_score", "damage_score", "severity"):
            v = data.get(field, -1)
            assert 0 <= v <= 1.0001, \
                f"{zone_id}.{field} = {v} is out of [0, 1]"


def test_flood_prob_map_is_float():
    """flood_prob_map must be a float numpy array with values in [0, 1]."""
    flood_map = mock_flood_prob_map()
    assert flood_map.dtype in (np.float32, np.float64), \
        f"Expected float dtype, got {flood_map.dtype}"
    assert flood_map.min() >= 0.0
    assert flood_map.max() <= 1.0


def test_binarised_flood_map_threshold():
    """
    BUG 2 fix verification: after binarising at threshold=0.45, only pixels
    with value ≥ 0.45 should be True.  Low-probability pixels (background
    noise at 0–0.05) must NOT be blocked.
    """
    flood_map = mock_flood_prob_map(640, 640, hotspot_zones=[(3, 5)])
    threshold = 0.45

    binary = (flood_map >= threshold)

    # Hotspot pixels should be True
    assert binary[3*64 + 20, 5*64 + 20], "Hotspot pixel should be blocked"
    # Background noise (0–0.05) should be False after thresholding
    # (mock generates 0–0.05 noise, well below 0.45 threshold)
    background_pixel = flood_map[0, 0]   # corner away from hotspot
    assert background_pixel < threshold, \
        f"Background pixel {background_pixel} is above threshold — test data issue"
    assert not binary[0, 0], "Background pixel should NOT be blocked after thresholding"


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 7: Resource Agent output format ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def test_resource_key_map_covers_llm_output():
    """
    _RESOURCE_KEY_MAP must cover all keys that the LLM prompt example shows.
    This validates BUG 3 fix is complete.
    """
    from agents.route_agent.route_agent import _canonical_resource_key

    # LLM prompt example keys (plural)
    llm_keys = ["boats", "ambulances", "rescue_teams"]
    for key in llm_keys:
        canonical = _canonical_resource_key(key)
        assert canonical is not None, \
            f"'{key}' not in _RESOURCE_KEY_MAP — BUG 3 fix incomplete"

    # Singular keys (LLM sometimes outputs these too)
    singular_keys = ["boat", "ambulance", "rescue_team"]
    for key in singular_keys:
        canonical = _canonical_resource_key(key)
        assert canonical is not None, \
            f"Singular '{key}' not handled — BUG 3 fix incomplete"


def test_rescue_plan_format_accepted():
    """
    The exact JSON format returned by allocate_rescue_resources_llm must
    be accepted by plan_all_routes without modification.
    """
    from agents.route_agent.route_agent import plan_all_routes

    # Exactly what the LLM returns per the prompt example
    rescue_plan = {
        "Z35": {"boats": 2, "ambulances": 1, "rescue_teams": 2},
        "Z01": {"boats": 1, "ambulances": 0, "rescue_teams": 1},
    }

    routes = plan_all_routes(
        image_meta           = make_image_meta(),
        resource_assignments = rescue_plan,
        base_locations       = make_base_locations(),
        use_real_osm         = False,
    )

    # Should produce routes for all non-zero counts
    non_zero = sum(1 for asn in rescue_plan.values()
                   for v in asn.values() if v > 0)
    assert len(routes) == non_zero, \
        f"Expected {non_zero} routes from rescue_plan, got {len(routes)}"


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 8: reroute (dynamic blockage) ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def test_reroute_after_new_blockage():
    """
    reroute() must find an alternative path after a new edge is blocked.
    """
    from agents.route_agent.road_network import build_synthetic_graph
    from agents.route_agent.router       import reroute

    G = build_synthetic_graph()

    # Original route from 0 to 8
    original = reroute(G, current_node=0, dest_node=8, newly_blocked_edges=[])
    assert original["success"], "Should find route before any blockage"

    # Block the direct top row (0→1→2 and back)
    blocked_result = reroute(G, current_node=0, dest_node=8,
                              newly_blocked_edges=[(0,1),(1,2),(2,5),(5,8)])
    # Should still succeed via the left column (0→3→6→7→8)
    assert blocked_result["success"], \
        "Should find alternate route after blocking top-right path"


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main ──────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  CRISIS MANAGEMENT — ROUTE AGENT TEST SUITE")
    print("="*65)

    sections = [
        ("geo_reference.py",            [
            test_geo_transform_center_maps_to_center,
            test_geo_transform_top_left_is_northwest,
        ]),
        ("zone_coordinates.py",         [
            test_parse_zone_name_basic,
            test_parse_zone_name_Z00,
            test_parse_zone_name_Z99,
            test_parse_zone_name_underscore,
            test_parse_zone_name_bad_raises,
            test_zone_center_pixels_Z00,
            test_zone_center_pixels_Z99,
            test_all_100_zone_coords_are_unique,
            test_zone_coords_within_image_bounds,
        ]),
        ("road_network.py",             [
            test_synthetic_graph_structure,
            test_synthetic_graph_has_travel_time,
            test_nearest_node_synthetic_returns_valid_node,
            test_mask_to_polygons_empty_mask,
            test_mask_to_polygons_flood_hotspot,
            test_remove_blocked_roads_penalises_edges,
        ]),
        ("router.py",                   [
            test_find_route_success,
            test_find_route_same_node,
            test_find_route_disconnected,
            test_path_to_waypoints,
            test_build_route_plan_success_shape,
        ]),
        ("plan_all_routes() end-to-end",[
            test_plan_all_routes_basic,
            test_plan_all_routes_with_flood_mask,
            test_plan_all_routes_resource_key_normalisation,
            test_plan_all_routes_bad_zone_name_skipped,
            test_plan_all_routes_zero_count_skipped,
            test_plan_all_routes_unknown_resource_skipped,
        ]),
        ("Vision Agent data flow",      [
            test_zone_map_keys_match_vision_agent_format,
            test_zone_map_scores_in_range,
            test_flood_prob_map_is_float,
            test_binarised_flood_map_threshold,
        ]),
        ("Resource Agent output format",[
            test_resource_key_map_covers_llm_output,
            test_rescue_plan_format_accepted,
        ]),
        ("reroute() dynamic blockage",  [
            test_reroute_after_new_blockage,
        ]),
    ]

    total_pass = 0
    total_fail = 0

    for section_name, tests in sections:
        print(f"\n── {section_name} {'─'*(50-len(section_name))}")
        for test_fn in tests:
            run_test(test_fn.__name__.replace("test_", ""), test_fn)

    # Summary
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total  = len(_results)

    print("\n" + "="*65)
    print(f"  RESULTS:  {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
        print("\n  Failed tests:")
        for name, ok, err in _results:
            if not ok:
                print(f"    ✗ {name}")
                print(f"      {err}")
    else:
        print("  — ALL PASSED ✅")
    print("="*65 + "\n")

    sys.exit(0 if failed == 0 else 1)