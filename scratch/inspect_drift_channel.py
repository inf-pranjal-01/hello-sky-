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
drifts = merged[merged['fault_type'] == 'drift']

print("Drift episodes on AWS-BHO-030:")
print(f"Total drift rows: {len(drifts)}")
for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
    diff = drifts[f'{param}_inj'] - drifts[f'{param}_clean']
    changed_rows = (diff.abs() > 1e-4).sum()
    print(f"Parameter: {param:<15} | Changed rows: {changed_rows:<5} | Max Delta: {diff.abs().max():.2f}")

# Check first 10 rows of the actual drifting parameter
active_param = [p for p in ['temperature_c', 'pressure_hpa', 'humidity_pct'] if (drifts[f'{p}_inj'] - drifts[f'{p}_clean']).abs().sum() > 0][0]
print(f"\nTracing active drift channel: {active_param}")
print(drifts[['timestamp', f'{active_param}_clean', f'{active_param}_inj']].head(10))
