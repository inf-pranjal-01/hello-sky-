import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import _featurize
from config import MULTIVARIATE_TEMP_DEVIATION_THRESHOLD, MULTIVARIATE_HUMIDITY_DEVIATION_THRESHOLD, MULTIVARIATE_VAPOR_CONSISTENCY_THRESHOLD

data = generate_network_benchmark(regime='operational_v1', seed=42, save_to_disk=False)

for sid, df in data.items():
    if 'fault_type' not in df.columns:
        continue
    mv_rows = df[df['fault_type'] == 'multivariate_inconsistency']
    if len(mv_rows) > 0:
        feat, _ = _featurize(df.drop(columns=['is_anomaly', 'fault_type', 'injected_delta', 'injected_start', 'injected_end'], errors='ignore'))
        feat['timestamp'] = pd.to_datetime(feat['timestamp']).dt.tz_localize(None)
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp']).dt.tz_localize(None)
        merged = feat.merge(df_copy[['timestamp', 'fault_type']], on='timestamp')
        sub = merged[merged['fault_type'] == 'multivariate_inconsistency']
        print(f"Station {sid} Multivariate fault rows ({len(sub)}):")
        for i in range(min(5, len(sub))):
            r = sub.iloc[i]
            t_dev = r['temp_deviation']
            h_dev = r['humidity_deviation']
            p_dev = r['pressure_deviation']
            v_dev = r['vapor_pressure_consistency_dev']
            print(f"  {r['timestamp']}: T_dev={t_dev:.2f}, H_dev={h_dev:.2f}, P_dev={p_dev:.2f}, V_dev={v_dev:.2f}")
        break
