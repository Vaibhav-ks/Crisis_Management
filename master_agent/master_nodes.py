# from agents.vision_agent.vision_agent import analyze_image
# from agents.resource_agent.drone_analysis import get_most_affected_zones
# from agents.drone_agent.drone_nodes import drone_decision_node, drone_dispatch_node
# from agents.drone_agent.drone_vision import drone_vision_node
 
# from agents.resource_agent.rescue_decision_llm import allocate_rescue_resources_llm
# # from agents.resource_agent.resource_agent import allocate_drones
# # from agents.resource_agent.resource_agent import allocate_rescue_resources
# # from agents.route_planner.route_planner import plan_routes
# # from agents.communication_agent.communication_agent import send_dispatch
# # from utils.llm_message import generate_dispatch_message


# from db.update_from_vision import update_zones_from_vision
# from db.update_people_count import update_people_count


# from utils.admin_interface import admin_approval


# # -----------------------------------
# # Vision Agent
# # -----------------------------------

# def vision_node(state):

#     print("\n[MASTER] Running Vision Agent")

#     result = analyze_image(state["satellite_image"])

#     print("[VISION] Zone analysis complete")

#     return {"zone_map": result["zone_map"]}


# # -----------------------------------
# # Store Zone Data
# # -----------------------------------

# def store_zone_node(state):

#     print("[DB] Updating zone database")

#     update_zones_from_vision(state["zone_map"])

#     return {}



#  # -----------------------------------
# # Drone Analysis → Most Affected Zones
# # -----------------------------------
 
# def drone_analysis_node(state):
 
#     print("\n[RESOURCE AGENT] Running Drone Analysis (Gemini-powered)")
 
#     affected_zones = get_most_affected_zones(top_n=5)
 
#     print(f"[RESOURCE AGENT] Most affected zones: {affected_zones}")
 
#     return {"most_affected_zones": affected_zones}
 
 
# # -----------------------------------
# # Update People Count
# # -----------------------------------
 
# def update_people_node(state):
 
#     print("\n[DB] Updating people counts in database")
 
#     update_people_count(state["people_counts"])
 
#     print("[DB] ✅ People counts saved to crisis.db")
 
#     return {}
 
 
# # Re-export drone nodes so master_graph can import everything from here
# __all__ = [
#     "vision_node",
#     "store_zone_node",
#     "drone_analysis_node",
#     "drone_decision_node",
#     "drone_dispatch_node",
#     "drone_vision_node",
#     "update_people_node",
# ]

# # -----------------------------------
# # Rescue Resource Allocation
# # -----------------------------------

# def rescue_decision_node(state):

#     print("\n[LLM RESCUE DECISION] Asking Gemini to allocate resources")

#     zone_map = state["zone_map"]
#     people_counts = state.get("people_counts", {})
#     zones = state.get("most_affected_zones", [])

#     rescue_plan = allocate_rescue_resources_llm(
#         zone_map,
#         people_counts,
#         zones
#     )

#     print("\n[LLM RESCUE DECISION] Proposed rescue plan:")

#     for zone, plan in rescue_plan.items():
#         print(f"  {zone} → {plan}")

#     return {"rescue_plan": rescue_plan}
# # -----------------------------------
# # Admin Resource Approval
# # -----------------------------------

# def admin_resource_node(state):

#     approved = admin_approval("Approve rescue resource allocation?")

#     if approved:
#         print("[ADMIN] Resources approved")
#     else:
#         print("[ADMIN] Resources rejected")

#     return {"resource_approved": approved}

# def resource_approval_router(state):

#     if state.get("resource_approved"):
#         return "approved"

#     else:
#         return "rejected"


# # -----------------------------------
# # Route Planning
# # -----------------------------------

# # def route_planner_node(state):

# #     print("\n[ROUTE PLANNER] Planning routes")

# #     routes = plan_routes(state["rescue_plan"])

# #     print("[ROUTE PLANNER] Routes planned")

# #     return {"route_plan": routes}


# # -----------------------------------
# # Admin Route Approval
# # -----------------------------------

# # def admin_route_node(state):

# #     approved = admin_approval("Approve rescue routes?")

# #     if not approved:
# #         raise Exception("Admin rejected routes")

# #     print("[ADMIN] Routes approved")

# #     return {}



# # def llm_message_node(state):

# #     print("\n[LLM] Generating human-friendly dispatch message")

# #     message = generate_dispatch_message(
# #         state["route_plan"],
# #         state["rescue_plan"]
# #     )

# #     print("\n[LLM MESSAGE]")
# #     print(message)

# #     return {"dispatch_message": message}


# # -----------------------------------
# # Communication Agent
# # -----------------------------------

# # def communication_node(state):

# #     print("\n[COMMUNICATION AGENT] Sending dispatch")

# #     send_dispatch(state["dispatch_message"])

# #     print("[COMMUNICATION AGENT] Dispatch delivered")

# #     return {}






# # from agents.vision_agent.vision_agent import analyze_image
# # from agents.resource_agent.drone_analysis import get_most_affected_zones
# # from agents.drone_agent.drone_nodes import drone_decision_node, drone_dispatch_node
# # from agents.drone_agent.drone_vision import drone_vision_node
 
# # # from agents.resource_agent.resource_agent import allocate_drones
# # # from agents.resource_agent.resource_agent import allocate_rescue_resources
# # # from agents.route_planner.route_planner import plan_routes
# # # from agents.communication_agent.communication_agent import send_dispatch
# # # from utils.llm_message import generate_dispatch_message


# # from db.update_from_vision import update_zones_from_vision
# # from db.update_people_count import update_people_count

# # # from utils.admin_interface import admin_approval


# # # -----------------------------------
# # # Vision Agent
# # # -----------------------------------

# # def vision_node(state):

# #     print("\n[MASTER] Running Vision Agent")

# #     result = analyze_image(state["satellite_image"])

# #     print("[VISION] Zone analysis complete")

# #     return {"zone_map": result["zone_map"]}


# # # -----------------------------------
# # # Store Zone Data
# # # -----------------------------------

# # def store_zone_node(state):

# #     print("[DB] Updating zone database")

# #     update_zones_from_vision(state["zone_map"])

# #     return {}



# #  # -----------------------------------
# # # Drone Analysis → Most Affected Zones
# # # -----------------------------------
 
# # def drone_analysis_node(state):
 
# #     print("\n[RESOURCE AGENT] Running Drone Analysis (Gemini-powered)")
 
# #     affected_zones = get_most_affected_zones(db_path="crisis.db", top_n=5)
 
# #     print(f"[RESOURCE AGENT] Most affected zones: {affected_zones}")
 
# #     return {"most_affected_zones": affected_zones}
 
 
# # # -----------------------------------
# # # Update People Count
# # # -----------------------------------
 
# # def update_people_node(state):
 
# #     print("\n[DB] Updating people counts in database")
 
# #     update_people_count(state["people_counts"])
 
# #     print("[DB] ✅ People counts saved to crisis.db")
 
# #     return {}
 
 
# # # Re-export drone nodes so master_graph can import everything from here
# # __all__ = [
# #     "vision_node",
# #     "store_zone_node",
# #     "drone_analysis_node",
# #     "drone_decision_node",
# #     "drone_dispatch_node",
# #     "drone_vision_node",
# #     "update_people_node",
# # ]
 

# # # -----------------------------------
# # # Drone Decision
# # # -----------------------------------

# # # def drone_decision_node(state):

# # #     print("\n[RESOURCE AGENT] Deciding drone deployment")

# # #     decisions = allocate_drones(state["zone_map"])

# # #     drone_zones = [
# # #         zone
# # #         for zone, data in decisions.items()
# # #         if data["deploy_drone"]
# # #     ]

# # #     print("[RESOURCE AGENT] Drone zones:", drone_zones)

# # #     return {"drone_zones": drone_zones}


# # # -----------------------------------
# # # Drone Dispatch
# # # -----------------------------------

# # # def drone_dispatch_node(state):

# # #     print("\n[MASTER] Dispatching drones")

# # #     for z in state["drone_zones"]:
# # #         print("Drone sent to zone:", z)

# # #     return {}


# # # -----------------------------------
# # # Drone Vision Analysis
# # # -----------------------------------

# # # def drone_vision_node(state):

# # #     print("\n[DRONE VISION] Detecting people")

# # #     # demo simulation
# # #     people_counts = {}

# # #     for zone in state["drone_zones"]:
# # #         people_counts[zone] = 5

# # #     print("[DRONE VISION] People detected:", people_counts)

# # #     return {"people_counts": people_counts}


# # # -----------------------------------
# # # Update People Count
# # # -----------------------------------

# # # def update_people_node(state):

# # #     print("\n[DB] Updating people counts")

# # #     update_people_count(state["people_counts"])

# # #     return {}


# # # -----------------------------------
# # # Rescue Resource Allocation
# # # -----------------------------------

# # # def rescue_decision_node(state):

# # #     print("\n[RESOURCE AGENT] Deciding rescue resources")

# # #     rescue_plan = allocate_rescue_resources()

# # #     print("[RESOURCE AGENT] Proposed rescue plan:")
# # #     print(rescue_plan)

# # #     return {"rescue_plan": rescue_plan}


# # # -----------------------------------
# # # Admin Resource Approval
# # # -----------------------------------

# # # def admin_resource_node(state):

# # #     approved = admin_approval("Approve rescue resource allocation?")

# # #     if not approved:
# # #         raise Exception("Admin rejected resource allocation")

# # #     print("[ADMIN] Resources approved")

# # #     return {}


# # # -----------------------------------
# # # Route Planning
# # # -----------------------------------

# # # def route_planner_node(state):

# # #     print("\n[ROUTE PLANNER] Planning routes")

# # #     routes = plan_routes(state["rescue_plan"])

# # #     print("[ROUTE PLANNER] Routes planned")

# # #     return {"route_plan": routes}


# # # -----------------------------------
# # # Admin Route Approval
# # # -----------------------------------

# # # def admin_route_node(state):

# # #     approved = admin_approval("Approve rescue routes?")

# # #     if not approved:
# # #         raise Exception("Admin rejected routes")

# # #     print("[ADMIN] Routes approved")

# # #     return {}



# # # def llm_message_node(state):

# # #     print("\n[LLM] Generating human-friendly dispatch message")

# # #     message = generate_dispatch_message(
# # #         state["route_plan"],
# # #         state["rescue_plan"]
# # #     )

# # #     print("\n[LLM MESSAGE]")
# # #     print(message)

# # #     return {"dispatch_message": message}


# # # -----------------------------------
# # # Communication Agent
# # # -----------------------------------

# # # def communication_node(state):

# # #     print("\n[COMMUNICATION AGENT] Sending dispatch")

# # #     send_dispatch(state["dispatch_message"])

# # #     print("[COMMUNICATION AGENT] Dispatch delivered")

# # #     return {}

"""
master_nodes.py
---------------
All LangGraph node functions for the master pipeline.

Pipeline flow:
  vision → store_zone → drone_analysis → drone_decision → drone_dispatch
  → drone_vision → update_people → rescue_decision → admin_resource
  → [approved] route_planner → admin_route → END
  → [rejected]  rescue_decision  (loop)

═══════════════════════════════════════════════════════════════════════════════
BUGS FOUND AND FIXED
═══════════════════════════════════════════════════════════════════════════════

BUG 1 ─ route_planner_node was commented out everywhere
  After admin approves resources, the graph jumped straight to END.
  The route_plan field in MasterState was NEVER populated.
  Fix: route_planner_node and admin_route_node are now implemented and active.

BUG 2 ─ image_meta not stored in state
  plan_all_routes() needs image GPS metadata (center_lat, center_lon, etc.)
  but vision_node only stored zone_map. The Route Agent had no way to build
  the geo-transform that maps zone names to GPS coordinates.
  Fix: vision_node now also stores image_meta in state.
       The initial invoke() call must include image_meta (see run_system.py).

BUG 3 ─ base_locations not configurable
  The old placeholder had base_locations hard-coded inside a comment.
  plan_all_routes() requires this dict. It is now read from MasterState
  (set by the caller) with sensible Prayagraj defaults as fallback.

BUG 4 ─ blocked_masks (flood prob map) never forwarded to Route Agent
  vision_agent.analyze_image() only returns zone_map, not the raw flood
  probability array from detect_flood(). The Route Agent needs this array
  to mark flooded roads as impassable.
  Fix: vision_agent.py is updated to also return flood_prob_map in its
       result dict. vision_node stores it as state["flood_mask"].
       route_planner_node passes it as blocked_masks={"flood": flood_mask}.
       (See also the updated vision_agent.py.)
"""

from agents.vision_agent.vision_agent        import analyze_image
from agents.resource_agent.drone_analysis    import get_most_affected_zones
from agents.drone_agent.drone_nodes          import drone_decision_node, drone_dispatch_node
from agents.drone_agent.drone_vision         import drone_vision_node
from agents.resource_agent.rescue_decision_llm import allocate_rescue_resources_llm
from agents.route_agent.route_agent          import plan_all_routes, print_routes

from db.update_from_vision  import update_zones_from_vision
from db.update_people_count import update_people_count
from utils.admin_interface  import admin_approval


# ── Default image metadata ────────────────────────────────────────────────────
# Used when image_meta is not provided in the initial invoke() call.
# These correspond to the Prayagraj area (the project's test region).

_DEFAULT_IMAGE_META = {
    "center_lat":  25.435,
    "center_lon":  81.846,
    "coverage_km": 5.0,
    "width_px":    640,
    "height_px":   640,
}

# Default base locations for rescue resources (Prayagraj area).
# Override by including "base_locations" in the initial invoke() state.
_DEFAULT_BASE_LOCATIONS = {
    "ambulance":   {"name": "District Hospital",   "lat": 25.440, "lon": 81.840},
    "rescue_team": {"name": "NDRF Station",        "lat": 25.430, "lon": 81.855},
    "boat":        {"name": "Boat Depot Allahabad", "lat": 25.425, "lon": 81.848},
}


# ── Vision Node ───────────────────────────────────────────────────────────────

def vision_node(state):
    """
    Run the Vision Agent on the satellite image.

    Reads : state["satellite_image"]  — file path to the image
            state["image_meta"]       — optional GPS metadata dict

    Writes: state["zone_map"]    — 100-zone severity/flood/damage map
            state["image_meta"]  — GPS metadata (populated with defaults if missing)
            state["flood_mask"]  — raw float flood probability map (H×W numpy array)
                                   forwarded to Route Agent for road blocking
    """
    print("\n[MASTER] Running Vision Agent")

    result = analyze_image(state["satellite_image"])

    # BUG 2 fix: ensure image_meta is in state for downstream use by Route Agent
    image_meta = state.get("image_meta") or _DEFAULT_IMAGE_META
    if not state.get("image_meta"):
        print(f"[VISION] No image_meta in state — using defaults: {image_meta}")

    print("[VISION] Zone analysis complete")

    return {
        "zone_map":   result["zone_map"],
        "flood_mask": result.get("flood_prob_map"),   # BUG 4 fix: forward flood map
        "image_meta": image_meta,
    }


# ── Store Zone Node ───────────────────────────────────────────────────────────

def store_zone_node(state):
    """Persist vision results to the SQLite database."""
    print("[DB] Updating zone database")
    update_zones_from_vision(state["zone_map"])
    return {}


# ── Drone Analysis Node ───────────────────────────────────────────────────────

def drone_analysis_node(state):
    """Query DB for the top-N most affected zones."""
    print("\n[RESOURCE AGENT] Running Drone Analysis")
    affected_zones = get_most_affected_zones(top_n=5)
    print(f"[RESOURCE AGENT] Most affected zones: {affected_zones}")
    return {"most_affected_zones": affected_zones}


# ── Update People Node ────────────────────────────────────────────────────────

def update_people_node(state):
    """Write drone-detected people counts to the DB."""
    print("\n[DB] Updating people counts in database")
    update_people_count(state["people_counts"])
    print("[DB] ✅ People counts saved to crisis.db")
    return {}


# ── Rescue Decision Node ──────────────────────────────────────────────────────

def rescue_decision_node(state):
    """Ask the Gemini LLM to allocate rescue resources across zones."""
    print("\n[LLM RESCUE DECISION] Asking Gemini to allocate resources")

    rescue_plan = allocate_rescue_resources_llm(
        zone_map      = state["zone_map"],
        people_counts = state.get("people_counts", {}),
        zones         = state.get("most_affected_zones", []),
    )

    print("\n[LLM RESCUE DECISION] Proposed rescue plan:")
    for zone, plan in rescue_plan.items():
        print(f"  {zone} → {plan}")

    return {"rescue_plan": rescue_plan}


# ── Admin Resource Approval Node ──────────────────────────────────────────────

def admin_resource_node(state):
    """Ask the human operator to approve or reject the rescue plan."""
    approved = admin_approval("Approve rescue resource allocation?")
    print("[ADMIN] Resources " + ("approved" if approved else "rejected"))
    return {"resource_approved": approved}


def resource_approval_router(state):
    """Conditional edge: approved → route_planner, rejected → rescue_decision."""
    return "approved" if state.get("resource_approved") else "rejected"


# ── Route Planner Node  (BUG 1 fix — was fully commented out) ─────────────────

def route_planner_node(state):
    """
    Call plan_all_routes() with:
      • rescue_plan  from the Resource Agent
      • image_meta   from state (set by vision_node, BUG 2 fix)
      • flood_mask   from Vision Agent (BUG 4 fix)
      • base_locations from state or defaults (BUG 3 fix)

    Writes: state["route_plan"] — list of route dicts
    """
    print("\n[ROUTE PLANNER] Planning routes for approved rescue plan")

    resource_assignments = state.get("rescue_plan", {})
    if not resource_assignments:
        print("[ROUTE PLANNER] WARNING: rescue_plan is empty — no routes to plan.")
        return {"route_plan": []}

    image_meta     = state.get("image_meta")   or _DEFAULT_IMAGE_META
    base_locations = state.get("base_locations") or _DEFAULT_BASE_LOCATIONS
    flood_mask     = state.get("flood_mask")    # may be None — that's fine

    # Build blocked_masks dict for route_agent (only flood for now;
    # a debris mask could be derived from damage_detections in future)
    blocked_masks = {}
    if flood_mask is not None:
        blocked_masks["flood"] = flood_mask

    # use_real_osm=False → synthetic graph (safe default; set True in production
    # when internet is available and osmnx is installed)
    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = base_locations,
        blocked_masks        = blocked_masks if blocked_masks else None,
        use_real_osm         = False,
        flood_threshold      = 0.45,
    )

    print_routes(routes)
    return {"route_plan": routes}


# ── Admin Route Approval Node ─────────────────────────────────────────────────

def admin_route_node(state):
    """
    Ask the operator to approve the route plan.
    If rejected, raises an exception to halt the pipeline (routes must be
    re-planned after fixing the rescue_plan — future improvement: loop back).
    """
    approved = admin_approval("Approve rescue routes?")
    if not approved:
        # For now, reject = hard stop. A future improvement would loop back
        # to rescue_decision or route_planner with modified parameters.
        raise RuntimeError("[ADMIN] Operator rejected the route plan. "
                           "Rerun with adjusted parameters.")
    print("[ADMIN] Routes approved")
    return {}


# ── Re-exports so master_graph.py can import everything from one place ─────────

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
]