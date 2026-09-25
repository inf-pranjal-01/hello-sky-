import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from data.anomaly_injector import generate_network_benchmark
from config import CLUSTERS, CUSUM_DRIFT_ALLOWANCE

# Map stations to clusters
STATION_TO_CLUSTER = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    for sid in [center] + neighbors:
        STATION_TO_CLUSTER[sid] = cid

def analyze_seed(seed=42):
    benchmark_b = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    # Analyze drift rows vs clean rows across all stations
    results = []
    for sid, df in benchmark_b.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        
        # Peer median for each param
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            peer_vals = pd.DataFrame({p: benchmark_b[p][param] for p in peer_ids if p in benchmark_b})
            peer_med = peer_vals.median(axis=1)
            peer_diff = df[param] - peer_med
            
            # Rate of change
            roc = df[param].diff()
            df[f'{param}_peer_diff'] = peer_diff
            df[f'{param}_roc'] = roc
            
        is_drift = (df['fault_type'] == 'drift')
        is_clean = (~df['is_anomaly'])
        
        # Check temperature peer diff on drift vs clean
        if is_drift.any():
            print(f"Station {sid} (Cluster {cid}) - Drift active:")
            print(f"  Drift temp peer diff mean: {df.loc[is_drift, 'temperature_c_peer_diff'].mean():.2f}, max: {df.loc[is_drift, 'temperature_c_peer_diff'].abs().max():.2f}")
            print(f"  Clean temp peer diff mean: {df.loc[is_clean, 'temperature_c_peer_diff'].mean():.2f}, max: {df.loc[is_clean, 'temperature_c_peer_diff'].abs().max():.2f}")

if __name__ == '__main__':
    analyze_seed(42)
