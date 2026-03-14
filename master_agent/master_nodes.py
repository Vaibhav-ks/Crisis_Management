from agents.vision_agent.vision_agent import analyze_image
from agents.resource_agent.drone_analysis import get_most_affected_zones
from agents.drone_agent.drone_nodes import drone_decision_node, drone_dispatch_node
from agents.drone_agent.drone_vision import drone_vision_node
 
from agents.resource_agent.rescue_decision_llm import allocate_rescue_resources_llm
# from agents.resource_agent.resource_agent import allocate_drones
# from agents.resource_agent.resource_agent import allocate_rescue_resources
# from agents.route_planner.route_planner import plan_routes
# from agents.communication_agent.communication_agent import send_dispatch
# from utils.llm_message import generate_dispatch_message


from db.update_from_vision import update_zones_from_vision
from db.update_people_count import update_people_count


from utils.admin_interface import admin_approval


# -----------------------------------
# Vision Agent
# -----------------------------------

def vision_node(state):

    print("\n[MASTER] Running Vision Agent")

    result = analyze_image(state["satellite_image"])

    print("[VISION] Zone analysis complete")

    return {"zone_map": result["zone_map"]}


# -----------------------------------
# Store Zone Data
# -----------------------------------

def store_zone_node(state):

    print("[DB] Updating zone database")

    update_zones_from_vision(state["zone_map"])

    return {}



 # -----------------------------------
# Drone Analysis → Most Affected Zones
# -----------------------------------
 
def drone_analysis_node(state):
 
    print("\n[RESOURCE AGENT] Running Drone Analysis (Gemini-powered)")
 
    affected_zones = get_most_affected_zones(top_n=5)
 
    print(f"[RESOURCE AGENT] Most affected zones: {affected_zones}")
 
    return {"most_affected_zones": affected_zones}
 
 
# -----------------------------------
# Update People Count
# -----------------------------------
 
def update_people_node(state):
 
    print("\n[DB] Updating people counts in database")
 
    update_people_count(state["people_counts"])
 
    print("[DB] ✅ People counts saved to crisis.db")
 
    return {}
 
 
# Re-export drone nodes so master_graph can import everything from here
__all__ = [
    "vision_node",
    "store_zone_node",
    "drone_analysis_node",
    "drone_decision_node",
    "drone_dispatch_node",
    "drone_vision_node",
    "update_people_node",
]

# -----------------------------------
# Rescue Resource Allocation
# -----------------------------------

def rescue_decision_node(state):

    print("\n[LLM RESCUE DECISION] Asking Gemini to allocate resources")

    zone_map = state["zone_map"]
    people_counts = state.get("people_counts", {})
    zones = state.get("most_affected_zones", [])

    rescue_plan = allocate_rescue_resources_llm(
        zone_map,
        people_counts,
        zones
    )

    print("\n[LLM RESCUE DECISION] Proposed rescue plan:")

    for zone, plan in rescue_plan.items():
        print(f"  {zone} → {plan}")

    return {"rescue_plan": rescue_plan}
# -----------------------------------
# Admin Resource Approval
# -----------------------------------

def admin_resource_node(state):

    approved = admin_approval("Approve rescue resource allocation?")

    if approved:
        print("[ADMIN] Resources approved")
    else:
        print("[ADMIN] Resources rejected")

    return {"resource_approved": approved}

def resource_approval_router(state):

    if state.get("resource_approved"):
        return "approved"

    else:
        return "rejected"


# -----------------------------------
# Route Planning
# -----------------------------------

# def route_planner_node(state):

#     print("\n[ROUTE PLANNER] Planning routes")

#     routes = plan_routes(state["rescue_plan"])

#     print("[ROUTE PLANNER] Routes planned")

#     return {"route_plan": routes}


# -----------------------------------
# Admin Route Approval
# -----------------------------------

# def admin_route_node(state):

#     approved = admin_approval("Approve rescue routes?")

#     if not approved:
#         raise Exception("Admin rejected routes")

#     print("[ADMIN] Routes approved")

#     return {}



# def llm_message_node(state):

#     print("\n[LLM] Generating human-friendly dispatch message")

#     message = generate_dispatch_message(
#         state["route_plan"],
#         state["rescue_plan"]
#     )

#     print("\n[LLM MESSAGE]")
#     print(message)

#     return {"dispatch_message": message}


# -----------------------------------
# Communication Agent
# -----------------------------------

# def communication_node(state):

#     print("\n[COMMUNICATION AGENT] Sending dispatch")

#     send_dispatch(state["dispatch_message"])

#     print("[COMMUNICATION AGENT] Dispatch delivered")

#     return {}






# from agents.vision_agent.vision_agent import analyze_image
# from agents.resource_agent.drone_analysis import get_most_affected_zones
# from agents.drone_agent.drone_nodes import drone_decision_node, drone_dispatch_node
# from agents.drone_agent.drone_vision import drone_vision_node
 
# # from agents.resource_agent.resource_agent import allocate_drones
# # from agents.resource_agent.resource_agent import allocate_rescue_resources
# # from agents.route_planner.route_planner import plan_routes
# # from agents.communication_agent.communication_agent import send_dispatch
# # from utils.llm_message import generate_dispatch_message


# from db.update_from_vision import update_zones_from_vision
# from db.update_people_count import update_people_count

# # from utils.admin_interface import admin_approval


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
 
#     affected_zones = get_most_affected_zones(db_path="crisis.db", top_n=5)
 
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
# # Drone Decision
# # -----------------------------------

# # def drone_decision_node(state):

# #     print("\n[RESOURCE AGENT] Deciding drone deployment")

# #     decisions = allocate_drones(state["zone_map"])

# #     drone_zones = [
# #         zone
# #         for zone, data in decisions.items()
# #         if data["deploy_drone"]
# #     ]

# #     print("[RESOURCE AGENT] Drone zones:", drone_zones)

# #     return {"drone_zones": drone_zones}


# # -----------------------------------
# # Drone Dispatch
# # -----------------------------------

# # def drone_dispatch_node(state):

# #     print("\n[MASTER] Dispatching drones")

# #     for z in state["drone_zones"]:
# #         print("Drone sent to zone:", z)

# #     return {}


# # -----------------------------------
# # Drone Vision Analysis
# # -----------------------------------

# # def drone_vision_node(state):

# #     print("\n[DRONE VISION] Detecting people")

# #     # demo simulation
# #     people_counts = {}

# #     for zone in state["drone_zones"]:
# #         people_counts[zone] = 5

# #     print("[DRONE VISION] People detected:", people_counts)

# #     return {"people_counts": people_counts}


# # -----------------------------------
# # Update People Count
# # -----------------------------------

# # def update_people_node(state):

# #     print("\n[DB] Updating people counts")

# #     update_people_count(state["people_counts"])

# #     return {}


# # -----------------------------------
# # Rescue Resource Allocation
# # -----------------------------------

# # def rescue_decision_node(state):

# #     print("\n[RESOURCE AGENT] Deciding rescue resources")

# #     rescue_plan = allocate_rescue_resources()

# #     print("[RESOURCE AGENT] Proposed rescue plan:")
# #     print(rescue_plan)

# #     return {"rescue_plan": rescue_plan}


# # -----------------------------------
# # Admin Resource Approval
# # -----------------------------------

# # def admin_resource_node(state):

# #     approved = admin_approval("Approve rescue resource allocation?")

# #     if not approved:
# #         raise Exception("Admin rejected resource allocation")

# #     print("[ADMIN] Resources approved")

# #     return {}


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