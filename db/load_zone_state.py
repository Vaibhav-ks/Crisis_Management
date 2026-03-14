import sqlite3


def load_zone_state():

    conn = sqlite3.connect("crisis.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT zone_id, flood_score, damage_score, severity, people_count
    FROM zones
    """)

    rows = cursor.fetchall()

    zone_map = {}

    for row in rows:

        zone_id, flood, damage, severity, people = row

        zone_map[zone_id] = {
            "flood_score": flood,
            "damage_score": damage,
            "severity": severity,
            "people_count": people
        }

    conn.close()

    return zone_map