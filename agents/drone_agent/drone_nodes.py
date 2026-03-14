import os


# Map zone IDs to image filenames in the zone_images/ folder
# Images must be named to match zones: Z00.jpg, Z12.png, etc.
ZONE_IMAGES_DIR = "zone_images"


def allocate_drones(affected_zones):
    """
    Allocate one drone per affected zone.
    
    Args:
        affected_zones: List[str] — e.g. ["Z23", "Z45", "Z12"]
    
    Returns:
        Dict: { "drone_1": "Z23", "drone_2": "Z45", ... }
    """
    allocation = {}

    for i, zone_id in enumerate(affected_zones, start=1):
        drone_name = f"drone_{i}"
        allocation[drone_name] = zone_id

    return allocation


def drone_decision_node(state):
    """
    LangGraph node: Decides which zones need drone deployment.
    Reads most_affected_zones from state.
    
    Returns:
        { "drone_zones": List[str], "drone_allocation": Dict }
    """
    from agents.resource_agent.drone_analysis import get_most_affected_zones

    print("\n[DRONE DECISION] Identifying most affected zones for drone deployment")

    # Get from state if already computed, else re-compute
    affected_zones = state.get("most_affected_zones")

    if not affected_zones:
        affected_zones = get_most_affected_zones()

    allocation = allocate_drones(affected_zones)

    print(f"[DRONE DECISION] Zones selected: {affected_zones}")
    print(f"[DRONE DECISION] Drone allocation: {allocation}")

    for drone, zone in allocation.items():
        print(f"  ✈ {drone} → {zone}")

    return {
        "drone_zones": affected_zones,
        "drone_allocation": allocation
    }


def drone_dispatch_node(state):
    """
    LangGraph node: Dispatches drones to their allocated zones.
    Confirms allocation and maps each drone to a zone image.
    
    Returns:
        { "zone_image_map": Dict }  — maps zone_id → image_path
    """
    allocation = state.get("drone_allocation", {})
    drone_zones = state.get("drone_zones", [])

    print("\n[DRONE DISPATCH] Dispatching drones to assigned zones")

    zone_image_map = {}

    # Gather all available images in zone_images folder
    available_images = []
    if os.path.exists(ZONE_IMAGES_DIR):
        available_images = sorted([
            f for f in os.listdir(ZONE_IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
    else:
        print(f"[DRONE DISPATCH] ⚠️  Folder '{ZONE_IMAGES_DIR}' not found!")

    print(f"[DRONE DISPATCH] Found {len(available_images)} images in '{ZONE_IMAGES_DIR}/'")

    # Map each affected zone to one image (in order)
    for i, zone_id in enumerate(drone_zones):
        if i < len(available_images):
            image_path = os.path.join(ZONE_IMAGES_DIR, available_images[i])
            zone_image_map[zone_id] = image_path
            print(f"  📷 {zone_id} → {image_path}")
        else:
            print(f"  ⚠️  No image available for {zone_id} (only {len(available_images)} images provided)")

    # Log drone dispatch confirmations
    for drone, zone in allocation.items():
        status = "✅ DISPATCHED" if zone in zone_image_map else "⚠️  NO IMAGE"
        print(f"  ✈ {drone} → {zone} [{status}]")

    return {"zone_image_map": zone_image_map}