import sqlite3
from datetime import datetime


def update_zones_from_vision(zone_map):

    conn = sqlite3.connect("crisis.db")
    cursor = conn.cursor()

    for zone_id, data in zone_map.items():

        cursor.execute("""
        INSERT INTO zones
        (zone_id, flood_score, damage_score, severity, people_count, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(zone_id) DO UPDATE SET
            flood_score = excluded.flood_score,
            damage_score = excluded.damage_score,
            severity = excluded.severity,
            last_updated = excluded.last_updated
        """, (
            zone_id,
            data["flood_score"],
            data["damage_score"],
            data["severity"],
            None,
            datetime.utcnow().isoformat()
        ))

    conn.commit()
    conn.close()