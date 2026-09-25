import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from data.anomaly_injector import generate_network_benchmark

data = generate_network_benchmark(regime='benchmark_b', seed=20260924, save_to_disk=False)
clean_bho = pd.read_csv('data/AWS-BHO-030.csv', parse_dates=['timestamp'])
inj_bho = data['AWS-BHO-030']
merged = inj_bho.merge(clean_bho, on='timestamp', suffixes=('_inj', '_clean'))

for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
    merged[f'{param}_diff'] = merged[f'{param}_inj'] - merged[f'{param}_clean']

# Find first non-zero drift episode for temperature
t_drift = merged[merged['temperature_c_diff'].abs() > 1e-4]
if len(t_drift) > 0:
    t_start = t_drift['timestamp'].iloc[0]
    ep = merged[(merged['timestamp'] >= t_start) & (merged['fault_type'] == 'drift')].head(12)
    print(f"--- TEMPERATURE DRIFT EPISODE (Starting {t_start}) ---")
    print(ep[['timestamp', 'temperature_c_clean', 'temperature_c_inj', 'temperature_c_diff']])

p_drift = merged[merged['pressure_hpa_diff'].abs() > 1e-4]
if len(p_drift) > 0:
    p_start = p_drift['timestamp'].iloc[0]
    ep_p = merged[(merged['timestamp'] >= p_start) & (merged['fault_type'] == 'drift')].head(12)
    print(f"\n--- PRESSURE DRIFT EPISODE (Starting {p_start}) ---")
    print(ep_p[['timestamp', 'pressure_hpa_clean', 'pressure_hpa_inj', 'pressure_hpa_diff']])
