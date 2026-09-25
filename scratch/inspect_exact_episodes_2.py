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

# Find non-zero drift episodes
for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
    diff_mask = merged[f'{param}_diff'].abs() > 1e-4
    if diff_mask.sum() > 0:
        ep = merged[diff_mask].head(12)
        print(f"\n--- ACTIVE {param.upper()} DRIFT EPISODE ---")
        print(ep[['timestamp', f'{param}_clean', f'{param}_inj', f'{param}_diff']])
