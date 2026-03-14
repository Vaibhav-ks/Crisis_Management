import sqlite3
from datetime import datetime


def update_people_count(people_counts):

    conn = sqlite3.connect("crisis.db")
    cursor = conn.cursor()

    for zone_id, count in people_counts.items():

        cursor.execute("""
        UPDATE zones
        SET people_count = ?, last_updated = ?
        WHERE zone_id = ?
        """, (
            count,
            datetime.utcnow().isoformat(),
            zone_id
        ))

    conn.commit()
    conn.close()