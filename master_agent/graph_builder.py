"""
graph_builder.py
----------------
THE single source of truth for the crisis-pipeline LangGraph topology.

WHY THIS FILE EXISTS
---------------------
Before this fix, the same 12 nodes + 2 conditional edges were defined
THREE separate times:
    1. master_agent/master_graph.py          (used by run_system.py — CLI, blocking input())
    2. master_agent/enhanced_master_graph.py  (dead code — imported nowhere)
    3. streamlit_app.py:_get_graph()          (Streamlit — needs a checkpointer + interrupts,
                                                because Streamlit reruns the whole script on every
                                                click and can't block on input())

That's a classic DRY violation: every time a node was renamed or an edge changed, someone had
to remember to update it in up to three places. They had already silently drifted — see e.g.
enhanced_master_graph.py's "pending" branch, which the router functions could never actually
return.

THE FIX
-------
Build the graph ONCE, parameterised by the two things that legitimately differ between the
CLI and the UI: whether a checkpointer is attached, and which nodes to interrupt before.

    CLI  (run_system.py):  build_master_graph()
                            → no checkpointer, no interrupts. admin_resource_node /
                              admin_route_node call admin_approval() which blocks on input().

    UI   (streamlit_app.py): build_master_graph(
                                  checkpointer=MemorySaver(),
                                  interrupt_before=["admin_resource", "admin_route"],
                              )
                            → the graph pauses BEFORE those nodes; Streamlit injects the
                              human decision via graph.update_state() and resumes with
                              graph.invoke(None, config=...).

Both call sites now import nodes/edges from exactly one place. Change the topology once,
both consumers update automatically.
"""

from typing import Optional, Sequence

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

# ── Node registry ──────────────────────────────────────────────────────────────

NODES = [
    ("vision",          vision_node),
    ("store_zone",      store_zone_node),
    ("drone_analysis",  drone_analysis_node),
    ("drone_decision",  drone_decision_node),
    ("drone_dispatch",  drone_dispatch_node),
    ("drone_vision",    drone_vision_node),
    ("update_people",   update_people_node),
    ("rescue_decision", rescue_decision_node),
    ("admin_resource",  admin_resource_node),
    ("route_planner",   route_planner_node),
    ("admin_route",     admin_route_node),
    ("communication",   communication_node),
]

# ── Linear edges (everything that isn't a conditional branch) ─────────────────

LINEAR_EDGES = [
    ("vision",          "store_zone"),
    ("store_zone",      "drone_analysis"),
    ("drone_analysis",  "drone_decision"),
    ("drone_decision",  "drone_dispatch"),
    ("drone_dispatch",  "drone_vision"),
    ("drone_vision",    "update_people"),
    ("update_people",   "rescue_decision"),
    ("rescue_decision", "admin_resource"),
    ("route_planner",   "admin_route"),
]

ENTRY_POINT = "vision"


def build_master_graph(checkpointer=None, interrupt_before: Optional[Sequence[str]] = None):
    """
    Build and compile the crisis-management StateGraph.

    Parameters
    ----------
    checkpointer : a LangGraph checkpointer (e.g. MemorySaver()), or None.
        None  → CLI mode. admin_resource_node / admin_route_node block on input().
        Given → required for `interrupt_before` to work, and for graph.get_state()/
                update_state() to function across separate invoke() calls (e.g. one
                per Streamlit rerun).

    interrupt_before : list of node names to pause before, or None.
        Typically ["admin_resource", "admin_route"] for any UI that can't block on
        a blocking input() call (Streamlit, FastAPI, etc).

    Returns
    -------
    A compiled LangGraph graph, ready for .invoke() / .get_state() / .update_state().
    """
    builder = StateGraph(MasterState)

    for name, fn in NODES:
        builder.add_node(name, fn)

    builder.set_entry_point(ENTRY_POINT)

    for src, dst in LINEAR_EDGES:
        builder.add_edge(src, dst)

    builder.add_conditional_edges(
        "admin_resource",
        resource_approval_router,
        {
            "approved": "route_planner",
            "rejected": "rescue_decision",   # loop back, re-allocate resources
        },
    )

    builder.add_conditional_edges(
        "admin_route",
        route_approval_router,
        {
            "approved": "communication",
            "rejected": "route_planner",     # re-plan routes with same rescue_plan
        },
    )

    builder.add_edge("communication", END)

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = list(interrupt_before)

    return builder.compile(**compile_kwargs)
