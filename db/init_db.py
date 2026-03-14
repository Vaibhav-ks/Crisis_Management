import sqlite3


def init_database():

    conn = sqlite3.connect("crisis.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS zones (
        zone_id TEXT PRIMARY KEY,
        flood_score REAL,
        damage_score REAL,
        severity REAL,
        people_count INTEGER,
        drone_deployed INTEGER,
        last_updated TEXT
    )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_database()