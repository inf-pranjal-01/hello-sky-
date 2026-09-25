import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark

STATION_TO_CLUSTER = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    for sid in [center] + neighbors:
        STATION_TO_CLUSTER[sid] = cid

def test_peer_trajectory(seed=42):
    benchmark_b = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    # Calculate peer divergence for each station
    all_frames = []
    for sid, df in benchmark_b.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        
        d = df.copy()
        d['station_id'] = sid
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            peer_vals = pd.DataFrame({p: benchmark_b[p][param] for p in peer_ids if p in benchmark_b})
            peer_med = peer_vals.median(axis=1)
            peer_diff = d[param] - peer_med
            
            # Causal rolling baseline of peer difference (excluding outliers)
            rolling_base = peer_diff.rolling(72, min_periods=12).median().shift(1)
            residual_peer = peer_diff - rolling_base
            
            d[f'{param}_peer_diff'] = peer_diff
            d[f'{param}_peer_res'] = residual_peer
        all_frames.append(d)
        
    df_all = pd.concat(all_frames, ignore_index=True)
    
    # Check drift detection using peer residual
    drift_mask = df_all['fault_type'] == 'drift'
    clean_mask = ~df_all['is_anomaly']
    
    print(f"Seed {seed}:")
    print(f"  Drift rows total: {drift_mask.sum()}")
    for thresh in [2.0, 2.5, 3.0, 3.5]:
        t_hit = (df_all['temperature_c_peer_res'].abs() > thresh) | (df_all['pressure_hpa_peer_res'].abs() > thresh*0.4) | (df_all['humidity_pct_peer_res'].abs() > thresh*3.0)
        tp_dr = (drift_mask & t_hit).sum()
        fp_dr = (clean_mask & t_hit).sum()
        print(f"  Thresh {thresh:.1f} -> Drift TP: {tp_dr}/{drift_mask.sum()} ({tp_dr/drift_mask.sum():.1%}), Clean FP: {fp_dr} ({fp_dr/clean_mask.sum():.2%})")

if __name__ == '__main__':
    for s in [42, 101, 202]:
        test_peer_trajectory(s)
