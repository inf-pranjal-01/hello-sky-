"""
SkyGuard AI — Export TimescaleDB Telemetry to CSV
Dumps the entire cloud hypertable into a CSV file for viewing in Microsoft Excel or VS Code.
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
    print("Connecting to Tiger Cloud TimescaleDB...")
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
        anomaly_score_pct,
        decision_basis,
        suggested_temperature_c,
        suggested_pressure_hpa,
        suggested_humidity_pct,
        health_status,
        source
    FROM sensor_readings
    ORDER BY time DESC;
    """
    
    print("Exporting readings to CSV...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    output_file = "cloud_telemetry_export.csv"
    df.to_csv(output_file, index=False)

    print("=" * 65)
    print(f"[SUCCESS] Exported {len(df)} rows to {output_file}")
    print("You can now open 'cloud_telemetry_export.csv' in Excel or VS Code!")
    print("=" * 65)

except Exception as e:
    print(f"[ERROR] Failed to export: {e}")
