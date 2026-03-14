import sqlite3


def load_zones_from_db(db_path="crisis.db"):
    """
    Load zone data from crisis.db
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT zone_id, flood_score, damage_score, severity, people_count
        FROM zones
    """)

    rows = cursor.fetchall()
    conn.close()

    zone_map = {}
    for row in rows:
        zone_id, flood, damage, severity, people = row
        zone_map[zone_id] = {
            "flood_score": flood or 0.0,
            "damage_score": damage or 0.0,
            "severity": severity or 0.0,
            "people_count": people or 0
        }

    return zone_map


def get_most_affected_zones(db_path="crisis.db", top_n=5):
    """
    Load zones from crisis.db and return top N most affected
    sorted by: severity (primary), flood_score (tiebreaker)

    Returns:
        List[str]: e.g. ["Z23", "Z45", "Z12", "Z67", "Z89"]
    """
    print("[Drone Analysis] Loading zone data from database...")
    zone_map = load_zones_from_db(db_path)

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