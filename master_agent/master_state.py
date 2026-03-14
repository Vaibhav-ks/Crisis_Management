from typing import TypedDict, Dict, List


class MasterState(TypedDict):

    satellite_image: str

    zone_map: Dict

    drone_zones: List[str]

    people_counts: Dict

    rescue_plan: Dict

    route_plan: Dict

    dispatch_message: str