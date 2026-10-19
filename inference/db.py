import sqlite3
import datetime


def init_db(path="potholes.db"):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS potholes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER,
        severity TEXT,
        confidence REAL,
        timestamp TEXT,
        image_path TEXT)""")
    conn.commit()
    return conn


def log_pothole(conn, track_id, severity, confidence, image_path):
    conn.execute(
        "INSERT INTO potholes (track_id, severity, confidence, timestamp, image_path) VALUES (?,?,?,?,?)",
        (track_id, severity, confidence,
         datetime.datetime.now().isoformat(timespec="seconds"), image_path))
    conn.commit()


def export_to_excel(db_path="potholes.db", xlsx_path="potholes_export.xlsx"):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM potholes", conn)
    conn.close()
    df.to_excel(xlsx_path, index=False)
    return len(df)