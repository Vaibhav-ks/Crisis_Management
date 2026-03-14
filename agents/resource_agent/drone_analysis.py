from db.load_zone_state import load_zone_state as load_zones_from_db


def get_most_affected_zones(top_n=5):
    """
    Load zones from crisis.db and return top N most affected
    sorted by: severity (primary), flood_score (tiebreaker)

    Returns:
        List[str]: e.g. ["Z23", "Z45", "Z12", "Z67", "Z89"]
    """
    print("[Drone Analysis] Loading zone data from database...")
    zone_map = load_zones_from_db()

    if not zone_map:
        print("[Drone Analysis] No zones found in database!")
        return []

    print(f"[Drone Analysis] Loaded {len(zone_map)} zones")

    # Sort by severity descending, flood_score as tiebreaker
    sorted_zones = sorted(
        zone_map.items(),
        key=lambda x: (x[1]["severity"], x[1]["flood_score"]),
        reverse=True
    )

    top_zones = [zone_id for zone_id, _ in sorted_zones[:top_n]]

    print(f"[Drone Analysis] Top {top_n} most affected zones:")
    for zone_id, data in sorted_zones[:top_n]:
        print(f"  {zone_id} -> severity={data['severity']:.3f}, "
              f"flood={data['flood_score']:.3f}, "
              f"damage={data['damage_score']:.3f}")

    return top_zones