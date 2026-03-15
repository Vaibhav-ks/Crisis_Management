# from langgraph.graph import StateGraph, END

# from .master_state import MasterState
 
# from .master_state import MasterState
# from .master_nodes import (
#     vision_node,
#     store_zone_node,
#     drone_analysis_node,
#     drone_decision_node,
#     drone_dispatch_node,
#     drone_vision_node,
#     update_people_node,
# )
 

# # create graph
# builder = StateGraph(MasterState)

# # add nodes
# builder.add_node("vision", vision_node)
# builder.add_node("store_zone", store_zone_node)

# # entry point
# builder.set_entry_point("vision")

# # edges
# builder.add_edge("vision", "store_zone")
# builder.add_edge("store_zone", END)

# # compile graph
# master_graph = builder.compile()

# # from langgraph.graph import StateGraph, END

# # from .master_state import MasterState

# from .master_nodes import (
#     vision_node,
#     store_zone_node,
#     drone_analysis_node,
#     drone_decision_node,
#     drone_dispatch_node,
#     drone_vision_node,
#     update_people_node,
#     rescue_decision_node,
#     admin_resource_node,
#     resource_approval_router
#     # route_planner_node,
#     # admin_route_node,
#     # llm_message_node,
#     # communication_node
# )

# # # ---------------------------------------------------
# # # Build LangGraph Workflow
# # # ---------------------------------------------------

# builder = StateGraph(MasterState)

# # # ---------------------------------------------------
# # # Register Nodes
# # # ---------------------------------------------------

# builder.add_node("vision", vision_node)

# builder.add_node("store_zone", store_zone_node)


# builder.add_node("drone_analysis", drone_analysis_node)       # Gemini: find most affected zones


# builder.add_node("drone_decision", drone_decision_node)

# builder.add_node("drone_dispatch", drone_dispatch_node)

# builder.add_node("drone_vision", drone_vision_node)

# builder.add_node("update_people", update_people_node)

# builder.add_node("rescue_decision", rescue_decision_node)

# builder.add_node("admin_resource", admin_resource_node)

# # builder.add_node("route_planner", route_planner_node)

# # builder.add_node("admin_route", admin_route_node)

# # builder.add_node("llm_message", llm_message_node)

# # builder.add_node("communication", communication_node)

# # # ---------------------------------------------------
# # # Entry Point
# # # ---------------------------------------------------

# builder.set_entry_point("vision")

# # # ---------------------------------------------------
# # # Workflow Edges
# # # ---------------------------------------------------

# builder.add_edge("vision", "store_zone")


# builder.add_edge("store_zone", "drone_analysis")

# builder.add_edge("drone_analysis", "drone_decision")


# builder.add_edge("drone_decision", "drone_dispatch")

# builder.add_edge("drone_dispatch", "drone_vision")

# builder.add_edge("drone_vision", "update_people")


# builder.add_edge("update_people", "rescue_decision")

# builder.add_edge("rescue_decision", "admin_resource")

# builder.add_conditional_edges(
#     "admin_resource",
#     resource_approval_router,
#     {
#         "approved": END,
#         "rejected": "rescue_decision"
#     }
# )


# # builder.add_edge("update_people", "rescue_decision")

# # builder.add_edge("rescue_decision", "admin_resource")

# # builder.add_edge("admin_resource", "route_planner")

# # builder.add_edge("route_planner", "admin_route")

# # builder.add_edge("admin_route", "llm_message")

# # builder.add_edge("llm_message", "communication")

# # builder.add_edge("communication", END)

# # # ---------------------------------------------------
# # # Compile Graph
# # # ---------------------------------------------------

# master_graph = builder.compile()

"""
master_graph.py
---------------
Builds and compiles the LangGraph StateGraph for the full crisis pipeline.

Flow:
  vision
    → store_zone
      → drone_analysis
        → drone_decision
          → drone_dispatch
            → drone_vision
              → update_people
                → rescue_decision
                  → admin_resource
                    → [approved] route_planner
                                   → admin_route → END
                    → [rejected]  rescue_decision  (loop until approved)

"""

from langgraph.graph import StateGraph, END

from .master_state import MasterState
from .master_nodes import (
    vision_node,
    store_zone_node,
    drone_analysis_node,
    drone_decision_node,
    drone_dispatch_node,
    drone_vision_node,
    update_people_node,
    rescue_decision_node,
    admin_resource_node,
    resource_approval_router,
    route_planner_node,    # BUG FIX: was commented out
    admin_route_node,      # BUG FIX: was commented out
)

# ── Build the graph ───────────────────────────────────────────────────────────

builder = StateGraph(MasterState)

# ── Register all nodes ────────────────────────────────────────────────────────

builder.add_node("vision",          vision_node)
builder.add_node("store_zone",      store_zone_node)
builder.add_node("drone_analysis",  drone_analysis_node)
builder.add_node("drone_decision",  drone_decision_node)
builder.add_node("drone_dispatch",  drone_dispatch_node)
builder.add_node("drone_vision",    drone_vision_node)
builder.add_node("update_people",   update_people_node)
builder.add_node("rescue_decision", rescue_decision_node)
builder.add_node("admin_resource",  admin_resource_node)
builder.add_node("route_planner",   route_planner_node)   # BUG FIX: active now
builder.add_node("admin_route",     admin_route_node)     # BUG FIX: active now

# ── Entry point ───────────────────────────────────────────────────────────────

builder.set_entry_point("vision")

# ── Edges ─────────────────────────────────────────────────────────────────────

builder.add_edge("vision",          "store_zone")
builder.add_edge("store_zone",      "drone_analysis")
builder.add_edge("drone_analysis",  "drone_decision")
builder.add_edge("drone_decision",  "drone_dispatch")
builder.add_edge("drone_dispatch",  "drone_vision")
builder.add_edge("drone_vision",    "update_people")
builder.add_edge("update_people",   "rescue_decision")
builder.add_edge("rescue_decision", "admin_resource")

# Conditional: admin approves → route_planner, rejects → re-run rescue_decision
builder.add_conditional_edges(
    "admin_resource",
    resource_approval_router,
    {
        "approved": "route_planner",   # BUG FIX: was END
        "rejected": "rescue_decision",
    }
)

builder.add_edge("route_planner",   "admin_route")   # BUG FIX: active now
builder.add_edge("admin_route",     END)

# ── Compile ───────────────────────────────────────────────────────────────────

master_graph = builder.compile()