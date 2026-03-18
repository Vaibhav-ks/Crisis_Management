"""
master_nodes.py
---------------
All LangGraph node functions for the master crisis-response pipeline.

Each function:
  • Receives the current MasterState
  • Does one well-defined job
  • Returns a dict of updated state fields

Pipeline order:
  vision → store_zone → drone_analysis → drone_decision → drone_dispatch
    → drone_vision → update_people → rescue_decision → admin_resource
    → route_planner → admin_route → communication → END
"""

from agents.vision_agent.vision_agent               import analyze_image
from agents.resource_agent.drone_analysis           import get_most_affected_zones
from agents.drone_agent.drone_nodes                 import drone_decision_node, drone_dispatch_node
from agents.drone_agent.drone_vision                import drone_vision_node
from agents.resource_agent.rescue_decision_llm      import allocate_rescue_resources_llm
from agents.route_agent.route_agent                 import plan_all_routes, print_routes
from agents.communication_agent.communication_agent import dispatch_all

from db.update_from_vision  import update_zones_from_vision
from db.update_people_count import update_people_count
from utils.admin_interface  import admin_approval
from generate_route_map     import generate_route_map


# ── Default image metadata (Prayagraj area) ───────────────────────────────────
# Used when image_meta is not provided in the initial invoke() call.

_DEFAULT_IMAGE_META = {
    "center_lat":  25.502483,
    "center_lon":  81.857394,
    "coverage_km": 0.5,
    "width_px":    1024,
    "height_px":   554,
}

# Default base locations for rescue resources (Prayagraj area).
# Override by including "base_locations" in the initial invoke() state.
_DEFAULT_BASE_LOCATIONS = {
    "ambulance":   {"name": "Hospital",   "lat": 19.06546856543151, "lon":  72.86100899070198},
    "rescue_team": {"name": "Rescue Center", "lat": 19.06847079812735, "lon": 72.85793995490616},
    "boat":        {"name": "Boat Depot",  "lat": 19.063380373548366, "lon": 72.85538649195271},
}


# ── Vision Node ───────────────────────────────────────────────────────────────

def vision_node(state):
    """
    Run the Vision Agent on the satellite image.

    Reads : state["satellite_image"]  — file path to the image
            state["image_meta"]       — optional GPS metadata dict

    Writes: state["zone_map"]    — 100-zone severity/flood/damage map
            state["image_meta"]  — GPS metadata (defaults if not supplied)
            state["flood_mask"]  — raw float flood probability array (H×W)
    """
    print("\n[MASTER] ── Vision Agent ─────────────────────────────────")

    result     = analyze_image(state["satellite_image"])
    
    flood_mask = result.get("flood_prob_map")
    height, width = flood_mask.shape[:2]

    # ✅ Step 1: get user-provided or default metadata
    image_meta = state.get("image_meta") or _DEFAULT_IMAGE_META.copy()

    if not state.get("image_meta"):
        print(f"[VISION] No image_meta in state — using defaults: {image_meta}")

    # ✅ Step 2: update with computed values
    image_meta.update({
        "width_px": width,
        "height_px": height
    })

    zone_count   = len(result.get("zone_map", {}))
    flood_zones  = sum(
        1 for z in result.get("zone_map", {}).values()
        if z.get("flood_score", 0) >= 0.45
    )
    damage_zones = sum(
        1 for z in result.get("zone_map", {}).values()
        if z.get("damage_score", 0) >= 0.45
    )

    print(f"[VISION] Complete — {zone_count} zones analysed  "
          f"| flood zones: {flood_zones}  | damage zones: {damage_zones}")

    return {
        "zone_map":   result["zone_map"],
        "flood_mask": result.get("flood_prob_map"),
        "image_meta": image_meta,
    }


# ── Store Zone Node ───────────────────────────────────────────────────────────

def store_zone_node(state):
    """Persist vision results to the SQLite database."""
    print("\n[DB] ── Storing zone data ───────────────────────────────")
    update_zones_from_vision(state["zone_map"])
    print("[DB] Zone data written to crisis.db")
    return {}


# ── Drone Analysis Node ───────────────────────────────────────────────────────

def drone_analysis_node(state):
    """Query DB for the top-N most affected zones."""
    print("\n[RESOURCE AGENT] ── Drone Analysis ─────────────────────")
    affected_zones = get_most_affected_zones(top_n=5)
    print(f"[RESOURCE AGENT] Top affected zones: {affected_zones}")
    return {"most_affected_zones": affected_zones}


# ── Update People Node ────────────────────────────────────────────────────────

def update_people_node(state):
    """Write drone-detected people counts to the DB."""
    print("\n[DB] ── Updating people counts ──────────────────────────")
    update_people_count(state["people_counts"])
    total = sum(state["people_counts"].values()) if state.get("people_counts") else 0
    print(f"[DB] People counts saved — total detected: {total}")
    return {}


# ── Rescue Decision Node ──────────────────────────────────────────────────────

def rescue_decision_node(state):
    """Ask the Gemini LLM to allocate rescue resources across zones."""
    print("\n[LLM] ── Rescue Resource Allocation ─────────────────────")

    rescue_plan = allocate_rescue_resources_llm(
        zone_map      = state["zone_map"],
        people_counts = state.get("people_counts", {}),
        zones         = state.get("most_affected_zones", []),
    )

    print("\n[LLM] Proposed rescue plan:")
    for zone, plan in rescue_plan.items():
        resources = ", ".join(f"{v}× {k}" for k, v in plan.items() if v)
        print(f"  {zone}  →  {resources}")

    return {"rescue_plan": rescue_plan}


# ── Admin Resource Approval Node ──────────────────────────────────────────────

def admin_resource_node(state):
    """Ask the admin to approve or reject the rescue plan."""
    print("\n[ADMIN] ── Resource Allocation Approval ─────────────────")
    approved = admin_approval("Approve rescue resource allocation?")
    print("[ADMIN] Resources " + ("APPROVED ✓" if approved else "REJECTED ✗"))
    return {"resource_approved": approved}


def resource_approval_router(state):
    """Conditional edge: approved → route_planner, rejected → rescue_decision."""
    return "approved" if state.get("resource_approved") else "rejected"


# ── Route Planner Node ────────────────────────────────────────────────────────

def route_planner_node(state):
    """
    Call plan_all_routes() and auto-generate the HTML route map.

    Reads : state["rescue_plan"]     — zone → resource allocations
            state["image_meta"]      — GPS coverage for geo-transform
            state["flood_mask"]      — optional flood probability array
            state["base_locations"]  — optional override for resource origins

    Writes: state["route_plan"]      — list of route dicts
            state["route_map_path"]  — absolute path to the generated HTML map
    """
    print("\n[ROUTE PLANNER] ── Planning Routes ──────────────────────")

    resource_assignments = state.get("rescue_plan", {})
    if not resource_assignments:
        print("[ROUTE PLANNER] WARNING: rescue_plan is empty — no routes to plan.")
        return {"route_plan": [], "route_map_path": None}

    image_meta     = state.get("image_meta")     or _DEFAULT_IMAGE_META
    base_locations = state.get("base_locations") or _DEFAULT_BASE_LOCATIONS
    flood_mask     = state.get("flood_mask")

    blocked_masks = {}
    if flood_mask is not None:
        blocked_masks["flood"] = flood_mask

    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = base_locations,
        blocked_masks        = blocked_masks if blocked_masks else None,
        use_real_osm         = True,
        flood_threshold      = 0.45,
    )

    print_routes(routes)

    # Auto-generate HTML route map immediately after planning
    print("\n[ROUTE MAP] Generating HTML map ...")
    try:
        map_path = generate_route_map(
            route_plans    = routes,
            image_meta     = image_meta,
            base_locations = base_locations,
            zone_map       = state.get("zone_map"),
        )
        print(f"[ROUTE MAP] Saved → {map_path}")
    except Exception as e:
        print(f"[ROUTE MAP] WARNING: Could not generate map: {e}")
        map_path = None

    return {"route_plan": routes, "route_map_path": map_path}


# ── Admin Route Approval Node ─────────────────────────────────────────────────

def admin_route_node(state):
    """
    Ask the operator to approve the route plan.
    Approved → communication node.
    Rejected → loops back to route_planner for re-planning.
    """
    print("\n[ADMIN] ── Route Plan Approval ──────────────────────────")

    map_path = state.get("route_map_path")
    if map_path:
        print(f"[ADMIN] Route map: {map_path}")

    approved = admin_approval("Approve rescue routes?")
    print("[ADMIN] Routes " + ("APPROVED ✓" if approved else "REJECTED ✗ — re-running planner"))
    return {"route_approved": approved}


def route_approval_router(state):
    """Conditional edge: approved → communication, rejected → route_planner."""
    return "approved" if state.get("route_approved") else "rejected"


# ── Communication Node ────────────────────────────────────────────────────────

def communication_node(state):
    """Generate dispatch instructions and send via SMS / audio."""
    print("\n[COMMUNICATION AGENT] ── Dispatch Pipeline ──────────────")

    route_plan      = state.get("route_plan")      or []
    zone_map        = state.get("zone_map")         or {}
    people_counts   = state.get("people_counts")    or {}
    field_reports   = state.get("field_reports")    or []
    dispatch_config = state.get("dispatch_config")  or {}

    if not route_plan:
        print("[COMMUNICATION AGENT] WARNING: No route_plan — nothing to dispatch.")
        return {
            "dispatch_result":  {
                "instructions": {}, "sms_results": [],
                "audio_files": [], "summary": "No routes planned.",
            },
            "dispatch_message": "No routes planned.",
        }

    zone_metadata = {}
    for zone_id, data in zone_map.items():
        raw_severity = data.get("severity", 0)
        if isinstance(raw_severity, float):
            severity_label = (
                "Critical" if raw_severity >= 0.7 else
                "Moderate" if raw_severity >= 0.4 else
                "Low"
            )
        else:
            severity_label = str(raw_severity)
        zone_metadata[zone_id] = {
            "severity":     severity_label,
            "victim_count": people_counts.get(zone_id, 0),
        }

    result = dispatch_all(
        route_plans     = route_plan,
        zone_metadata   = zone_metadata,
        field_reports   = field_reports,
        dispatch_config = dispatch_config,
    )

    n_instructions = len(result.get("instructions", {}))
    n_sms          = len(result.get("sms_results", []))
    print(f"[COMMUNICATION AGENT] Complete — "
          f"{n_instructions} instruction(s)  |  {n_sms} SMS(es) sent")

    return {
        "dispatch_result":  result,
        "dispatch_message": result.get("summary", ""),
    }


# ── Re-exports ────────────────────────────────────────────────────────────────

__all__ = [
    "vision_node",
    "store_zone_node",
    "drone_analysis_node",
    "drone_decision_node",
    "drone_dispatch_node",
    "drone_vision_node",
    "update_people_node",
    "rescue_decision_node",
    "admin_resource_node",
    "resource_approval_router",
    "route_planner_node",
    "admin_route_node",
    "route_approval_router",
    "communication_node",
]