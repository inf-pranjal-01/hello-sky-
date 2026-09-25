import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import joblib

from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import ARTIFACTS_PATH, _featurize
from model.spike_tracker import init_spike_state, step_spike_state
from model.detect import graduated_confidence_spike

data = generate_network_benchmark(regime='operational_v1', seed=42, save_to_disk=False)
artifact = joblib.load(ARTIFACTS_PATH)
thresholds = artifact["rule_thresholds"]

# Find 5 spike episodes across stations
spikes_traced = 0
for sid, df in data.items():
    if 'fault_type' not in df.columns:
        continue
    spike_rows = df[df['fault_type'] == 'spike']
    if len(spike_rows) == 0:
        continue
    
    # Trace through the station
    st_t = init_spike_state()
    st_p = init_spike_state()
    st_h = init_spike_state()
    
    # Featurize
    feat, _ = _featurize(df.drop(columns=['is_anomaly', 'fault_type', 'injected_delta', 'injected_start', 'injected_end'], errors='ignore'))
    
    # Get thresholds
    th_t = thresholds['spike']['temp'].get(sid, thresholds['spike']['temp']['__global__'])
    th_p = thresholds['spike']['pressure'].get(sid, thresholds['spike']['pressure']['__global__'])
    th_h = thresholds['spike']['humidity'].get(sid, thresholds['spike']['humidity']['__global__'])
    
    for idx, row in df.iterrows():
        if row['fault_type'] == 'spike':
            t_val = row['temperature_c']
            p_val = row['pressure_hpa']
            h_val = row['humidity_pct']
            
            # Find matching featurized row
            f_row = feat[feat['timestamp'] == row['timestamp']]
            t_dev = f_row['temp_deviation'].values[0] if len(f_row) > 0 and 'temp_deviation' in f_row.columns else 0.0
            p_dev = f_row['pressure_deviation'].values[0] if len(f_row) > 0 and 'pressure_deviation' in f_row.columns else 0.0
            h_dev = f_row['humidity_deviation'].values[0] if len(f_row) > 0 and 'humidity_deviation' in f_row.columns else 0.0
            
            prev_t = df.iloc[idx-1]['temperature_c'] if idx > 0 else t_val
            diff_t = abs(t_val - prev_t)
            prev_p = df.iloc[idx-1]['pressure_hpa'] if idx > 0 else p_val
            diff_p = abs(p_val - prev_p)
            prev_h = df.iloc[idx-1]['humidity_pct'] if idx > 0 else h_val
            diff_h = abs(h_val - prev_h)
            
            conf_t, stat_t, r_t = step_spike_state(t_val, t_dev, th_t, 1.1, st_t, graduated_confidence_spike)
            conf_p, stat_p, r_p = step_spike_state(p_val, p_dev, th_p, 1.1, st_p, graduated_confidence_spike)
            conf_h, stat_h, r_h = step_spike_state(h_val, h_dev, th_h, 1.1, st_h, graduated_confidence_spike)
            
            print(f"Spike at {sid} {row['timestamp']}:")
            print(f"  T: val={t_val:.1f}, prev={prev_t:.1f}, diff={diff_t:.1f}, dev={t_dev:.1f}, th={th_t:.1f} -> conf={conf_t}, stat={stat_t}")
            print(f"  P: val={p_val:.1f}, prev={prev_p:.1f}, diff={diff_p:.1f}, dev={p_dev:.1f}, th={th_p:.1f} -> conf={conf_p}, stat={stat_p}")
            print(f"  H: val={h_val:.1f}, prev={prev_h:.1f}, diff={diff_h:.1f}, dev={h_dev:.1f}, th={th_h:.1f} -> conf={conf_h}, stat={stat_h}")
            spikes_traced += 1
            if spikes_traced >= 5:
                break
    if spikes_traced >= 5:
        break
