"""
master_state.py
---------------
Defines the shared TypedDict that flows through the LangGraph pipeline.

Fields are populated progressively as each node runs.
"""

from typing import TypedDict, Dict, List, Optional, Any


class MasterState(TypedDict):

    # ── Input (provided by caller in invoke()) ────────────────────────────────
    satellite_image: str                # file path to the satellite/aerial image

    image_meta: Optional[Dict]          # GPS coverage metadata:
                                        # {center_lat, center_lon, coverage_km,
                                        #  width_px, height_px}

    base_locations: Optional[Dict]      # override default resource origins:
                                        # {"ambulance": {name, lat, lon}, ...}

    # ── Vision Agent output ───────────────────────────────────────────────────
    zone_map: Dict                      # {"Z00": {flood_score, damage_score, severity}, ...}
    flood_mask: Optional[Any]           # float H×W numpy array (0–1) from detect_flood()

    # ── Drone Analysis output ─────────────────────────────────────────────────
    most_affected_zones: List[str]      # e.g. ["Z35", "Z01", "Z72"]

    # ── Drone Decision & Dispatch output ─────────────────────────────────────
    drone_zones:      List[str]
    drone_allocation: Dict              # {"drone_1": "Z35", "drone_2": "Z01", ...}
    zone_image_map:   Dict              # {"Z35": "zone_images/img1.jpg", ...}

    # ── Drone Vision output ───────────────────────────────────────────────────
    people_counts: Dict                 # {"Z35": 12, "Z01": 5, ...}

    # ── Rescue Decision output ────────────────────────────────────────────────
    rescue_plan: Optional[Dict]         # {"Z35": {"boats": 2, "ambulances": 1}, ...}

    # ── Admin Resource Approval output ───────────────────────────────────────
    resource_approved: Optional[bool]

    # ── Route Planner output ──────────────────────────────────────────────────
    route_plan: Optional[List]          # list of route dicts from plan_all_routes()
    route_map_path: Optional[str]       # absolute path to the generated HTML map

    # ── Admin Route Approval output ───────────────────────────────────────────
    route_approved: Optional[bool]

    # ── Communication Agent output ────────────────────────────────────────────
    dispatch_message: Optional[str]
    dispatch_result:  Optional[Dict]    # {instructions, sms_results, audio_files, summary}
    dispatch_config:  Optional[Dict]    # {language, send_sms, generate_audio, to_number}
    field_reports:    Optional[List[str]]