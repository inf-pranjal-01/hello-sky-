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

# Filter strictly for fault_type == 'drift'
drifts_only = merged[merged['fault_type'] == 'drift']
for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
    diff_mask = drifts_only[f'{param}_diff'].abs() > 1e-4
    if diff_mask.sum() > 0:
        ep = drifts_only[diff_mask]
        print(f"\n==========================================")
        print(f"STRICT FAULT_TYPE == 'DRIFT' FOR {param.upper()}")
        print(f"Total timesteps: {len(ep)}")
        print(f"First 10 timesteps of episode:")
        print(ep[['timestamp', f'{param}_clean', f'{param}_inj', f'{param}_diff']].head(10))
