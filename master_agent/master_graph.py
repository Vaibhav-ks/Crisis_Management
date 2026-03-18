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
                  → admin_resource ──[approved]──→ route_planner
                  │                                     ↓
                  └──[rejected]──→ rescue_decision  admin_route ──[approved]──→ communication → END
                                                         │
                                                    [rejected]──→ route_planner  (re-plan loop)
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
    route_planner_node,
    admin_route_node,
    route_approval_router,
    communication_node,
)

# ── Build the graph ───────────────────────────────────────────────────────────

builder = StateGraph(MasterState)

# ── Register nodes ────────────────────────────────────────────────────────────

builder.add_node("vision",          vision_node)
builder.add_node("store_zone",      store_zone_node)
builder.add_node("drone_analysis",  drone_analysis_node)
builder.add_node("drone_decision",  drone_decision_node)
builder.add_node("drone_dispatch",  drone_dispatch_node)
builder.add_node("drone_vision",    drone_vision_node)
builder.add_node("update_people",   update_people_node)
builder.add_node("rescue_decision", rescue_decision_node)
builder.add_node("admin_resource",  admin_resource_node)
builder.add_node("route_planner",   route_planner_node)
builder.add_node("admin_route",     admin_route_node)
builder.add_node("communication",   communication_node)

# ── Entry point ───────────────────────────────────────────────────────────────

builder.set_entry_point("vision")

# ── Linear edges ─────────────────────────────────────────────────────────────

builder.add_edge("vision",          "store_zone")
builder.add_edge("store_zone",      "drone_analysis")
builder.add_edge("drone_analysis",  "drone_decision")
builder.add_edge("drone_decision",  "drone_dispatch")
builder.add_edge("drone_dispatch",  "drone_vision")
builder.add_edge("drone_vision",    "update_people")
builder.add_edge("update_people",   "rescue_decision")
builder.add_edge("rescue_decision", "admin_resource")

# ── Conditional: admin approves resources ─────────────────────────────────────

builder.add_conditional_edges(
    "admin_resource",
    resource_approval_router,
    {
        "approved": "route_planner",
        "rejected": "rescue_decision",   # loop back, re-allocate resources
    }
)

builder.add_edge("route_planner", "admin_route")

# ── Conditional: admin approves routes ───────────────────────────────────────

builder.add_conditional_edges(
    "admin_route",
    route_approval_router,
    {
        "approved": "communication",
        "rejected": "route_planner",     # re-plan routes with same rescue_plan
    }
)

builder.add_edge("communication", END)

# ── Compile ───────────────────────────────────────────────────────────────────

master_graph = builder.compile()