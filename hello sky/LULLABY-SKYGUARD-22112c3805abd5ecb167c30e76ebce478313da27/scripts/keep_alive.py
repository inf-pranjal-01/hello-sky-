"""
SkyGuard AI — TimescaleDB / Tiger Cloud Keep-Alive Utility
Pings the database with a lightweight heartbeat query to reset the inactivity timer.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

db_url = os.environ.get("DATABASE_URL") or os.environ.get("TIMESCALE_SERVICE_URL")

if not db_url:
    print("[ERROR] Database connection string (DATABASE_URL) is missing in .env file.")
    sys.exit(1)

try:
    import psycopg2

    # Mask password for safe display
    host_display = db_url.split("@")[-1] if "@" in db_url else "configured host"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Connecting to Tiger Cloud ({host_display})...")

    conn = psycopg2.connect(db_url, connect_timeout=10)
    cursor = conn.cursor()
    cursor.execute("SELECT NOW(), version();")
    row = cursor.fetchone()
    
    # Check readings count
    cursor.execute("SELECT COUNT(*) FROM sensor_readings;")
    count = cursor.fetchone()[0]

    # Check stations count
    cursor.execute("SELECT COUNT(DISTINCT station_id) FROM sensor_readings;")
    stations_count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    print("=" * 60)
    print("[SUCCESS] TIGER CLOUD TIMESCALEDB IS AWAKE & HEALTHY!")
    print(f"Server Timestamp : {row[0]}")
    print(f"Active Stations  : {stations_count}")
    print(f"Stored Readings  : {count} rows")
    print("Inactivity Timer : Successfully reset. Service remains active.")
    print("=" * 60)

except Exception as e:
    print(f"[ERROR] Connection error: {e}")
    sys.exit(1)
