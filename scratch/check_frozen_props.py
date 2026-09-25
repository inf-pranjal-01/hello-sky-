import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.anomaly_injector import generate_network_benchmark

data = generate_network_benchmark(regime='operational_v1', seed=42, save_to_disk=False)

frozen_eps = []
for sid, df in data.items():
    if 'fault_type' not in df.columns:
        continue
    f_rows = df[df['fault_type'] == 'frozen_value']
    if len(f_rows) > 0:
        for p in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            diffs = f_rows[p].diff().abs()
            print(f"Station {sid} frozen {p}: vals={f_rows[p].values[:5]}, diffs={diffs.values[:5]}, std={f_rows[p].std():.4f}")
        break
