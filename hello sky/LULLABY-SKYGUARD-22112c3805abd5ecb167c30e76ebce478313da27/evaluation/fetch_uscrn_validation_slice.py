import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

# Base URL for USCRN hourly data
BASE_URL = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/{year}/CRNH0203-{year}-{station}.txt"

# Edit this list with actual station names from the USCRN directory
STATIONS = [
    "CO_Boulder_14_W",
    # Add more stations here
]
YEAR = "2024"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "uscrn_validation_slice.csv"

# Columns as per NOAA readme
COLUMNS = [
    "WBAN", "LST_DATE", "LST_TIME", "CRX_VN", "LONGITUDE", "LATITUDE",
    "T_CALC", "T_HR_AVG", "T_MAX", "T_MIN", "P_CALC",
    "SOLARAD", "SOLARAD_FLAG", "SOLARAD_MAX", "SOLARAD_MAX_FLAG",
    "SOLARAD_MIN", "SOLARAD_MIN_FLAG", "SUR_TEMP_TYPE", "SUR_TEMP",
    "SUR_TEMP_FLAG", "SUR_TEMP_MAX", "SUR_TEMP_MAX_FLAG", "SUR_TEMP_MIN",
    "SUR_TEMP_MIN_FLAG", "RH_HR_AVG", "RH_HR_AVG_FLAG", 
    "SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "SOIL_MOISTURE_20", 
    "SOIL_MOISTURE_50", "SOIL_MOISTURE_100", 
    "SOIL_TEMP_5", "SOIL_TEMP_10", "SOIL_TEMP_20", 
    "SOIL_TEMP_50", "SOIL_TEMP_100"
]

def fetch_and_parse(station, year):
    url = BASE_URL.format(year=year, station=station)
    print(f"Fetching {station}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            df = pd.read_csv(response, sep=r'\s+', header=None, names=COLUMNS, na_values=["-99.000", "-9999.0"])
            
            # Construct timestamp
            # LST_DATE is YYYYMMDD, LST_TIME is HHMM
            df['timestamp'] = pd.to_datetime(
                df['LST_DATE'].astype(str) + df['LST_TIME'].astype(str).str.zfill(4), 
                format='%Y%m%d%H%M'
            )
            
            df['station_id'] = station
            
            # Map columns
            df['temperature_c'] = df['T_HR_AVG']
            df['humidity_pct'] = df['RH_HR_AVG']
            df['pressure_hpa'] = np.nan
            
            # Ground truth:
            # Temperature: "anomaly" means USCRN reported it missing
            temp_missing = df['temperature_c'].isna()
            
            # Humidity: RH_HR_AVG_FLAG == 3 means erroneous
            # But missing is also an anomaly
            humidity_bad = (df['RH_HR_AVG_FLAG'] == 3) | df['humidity_pct'].isna()
            
            df['is_anomaly'] = temp_missing | humidity_bad
            
            # Assign a fault type for evaluation breakdown
            def get_fault_type(row):
                if not row['is_anomaly']:
                    return 'none'
                if pd.isna(row['temperature_c']) or pd.isna(row['humidity_pct']):
                    return 'dropout'
                if row['RH_HR_AVG_FLAG'] == 3:
                    return 'uscrn_qc_flag'
                return 'unknown'
                
            df['fault_type'] = df.apply(get_fault_type, axis=1)
            
            return df[['station_id', 'timestamp', 'temperature_c', 'pressure_hpa', 'humidity_pct', 'is_anomaly', 'fault_type']]
    except Exception as e:
        print(f"Failed to fetch {station}: {e}")
        return None

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for station in STATIONS:
        df = fetch_and_parse(station, YEAR)
        if df is not None:
            frames.append(df)
            
    if frames:
        final_df = pd.concat(frames, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Wrote {len(final_df)} rows to {OUTPUT_FILE}")
    else:
        print("No data fetched.")

if __name__ == '__main__':
    main()
