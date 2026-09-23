"""
SkyGuard AI — View Database Telemetry
Prints the actual rows stored in Tiger Cloud TimescaleDB hypertable in a clean table format.
"""

import os
import sys
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

db_url = os.environ.get("DATABASE_URL") or os.environ.get("TIMESCALE_SERVICE_URL")

if not db_url:
    print("[ERROR] DATABASE_URL is missing in .env")
    sys.exit(1)

try:
    import psycopg2
    conn = psycopg2.connect(db_url)
    
    query = """
    SELECT 
        time,
        station_id,
        temperature_c,
        pressure_hpa,
        humidity_pct,
        is_anomaly,
        fault_type,
        severity,
        health_status,
        source
    FROM sensor_readings
    ORDER BY time DESC
    LIMIT 25;
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    print("=" * 95)
    print(" 📡 TIGER CLOUD TIMESCALEDB — LATEST 25 READINGS (HYPERTABLE: sensor_readings)")
    print("=" * 95)
    if df.empty:
        print(" (Table is currently empty. Run the simulator or fetch live data to see readings.)")
    else:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
    print("=" * 95)

except Exception as e:
    print(f"[ERROR] Failed to query database: {e}")
