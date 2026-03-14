from typing import TypedDict, Dict, List, Optional


class MasterState(TypedDict):

    # Input
    satellite_image: str

    # Vision Agent output
    zone_map: Dict

    # Drone Analysis output (resource agent)
    most_affected_zones: List[str]

    # Drone Decision & Dispatch output
    drone_zones: List[str]
    drone_allocation: Dict         # { "drone_1": "Z23", ... }
    zone_image_map: Dict           # { "Z23": "zone_images/img1.jpg", ... }

    # Drone Vision output
    people_counts: Dict            # { "Z23": 5, "Z45": 12, ... }

    # (Future nodes)
    rescue_plan: Optional[Dict]
    route_plan: Optional[Dict]
    dispatch_message: Optional[str]