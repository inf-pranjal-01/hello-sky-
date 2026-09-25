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

def precompute_clean_profiles():
    profiles = {}
    for sid in STATION_TO_CLUSTER:
        cid = STATION_TO_CLUSTER[sid]
        df_s = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        hours = df_s['timestamp'].dt.hour
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            peer_dfs = [pd.read_csv(f'data/{p}.csv', parse_dates=['timestamp'])[param] for p in peer_ids]
            peer_med = pd.concat(peer_dfs, axis=1).median(axis=1)
            diff = df_s[param] - peer_med
            
            p_mean = diff.groupby(hours).mean()
            p_std = diff.groupby(hours).std().clip(lower=0.3)
            
            profiles[(sid, param)] = {
                'peer_mean': p_mean,
                'peer_std': p_std,
            }
    return profiles

PROFILES = precompute_clean_profiles()

def test_consensus_drift(seed=42):
    benchmark_b = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    station_alerts = {}
    for sid, df in benchmark_b.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        hours = pd.to_datetime(df['timestamp']).dt.hour
        n_rows = len(df)
        drift_alert = np.zeros(n_rows, dtype=bool)
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            prof = PROFILES[(sid, param)]
            peer_vals = pd.DataFrame({p: benchmark_b[p][param] for p in peer_ids if p in benchmark_b})
            peer_med = peer_vals.median(axis=1)
            peer_std = peer_vals.std(axis=1).fillna(1.0)
            
            raw_diff = df[param] - peer_med
            exp_diff = hours.map(prof['peer_mean'])
            exp_std = hours.map(prof['peer_std']).fillna(1.0).clip(lower=0.3)
            
            # Isolated peer divergence (target is far from peer median, while peers agree with each other)
            z_peer = (raw_diff - exp_diff) / exp_std
            
            # Peer consensus ratio: how much target deviates compared to peer internal spread
            peer_internal_spread = peer_std.clip(lower=0.2)
            isolation_ratio = raw_diff.abs() / peer_internal_spread
            
            # CUSUM accumulation
            splus, sminus = 0.0, 0.0
            in_drift = False
            drift_dir = 0
            clean_count = 0
            
            p_alert = np.zeros(n_rows, dtype=bool)
            k = 0.8
            h_trigger = 8.0
            
            for i in range(n_rows):
                z = z_peer.iloc[i]
                iso = isolation_ratio.iloc[i]
                p_s = peer_internal_spread.iloc[i]
                
                if np.isnan(z):
                    splus, sminus = 0.0, 0.0
                    in_drift = False
                    continue
                
                # Only accumulate if peers are mutually consistent (internal spread is reasonable)
                # and target is diverging
                if p_s < 3.0:
                    splus = max(0.0, splus + z - k) if z > 0.5 else max(0.0, splus - 1.0)
                    sminus = max(0.0, sminus - z - k) if z < -0.5 else max(0.0, sminus - 1.0)
                else:
                    # Weather front passing through entire cluster: drain CUSUM
                    splus = max(0.0, splus - 1.5)
                    sminus = max(0.0, sminus - 1.5)
                
                if not in_drift:
                    if (splus >= h_trigger and z >= 2.5 and iso >= 2.0):
                        in_drift = True
                        drift_dir = 1
                        clean_count = 0
                    elif (sminus >= h_trigger and z <= -2.5 and iso >= 2.0):
                        in_drift = True
                        drift_dir = -1
                        clean_count = 0
                else:
                    # Maintain alert as long as deviation persists
                    if (drift_dir == 1 and z >= 1.5) or (drift_dir == -1 and z <= -1.5):
                        clean_count = 0
                    else:
                        clean_count += 1
                        if clean_count >= 3 or (drift_dir == 1 and z < 0.5) or (drift_dir == -1 and z > -0.5):
                            in_drift = False
                            splus, sminus = 0.0, 0.0
                            
                if in_drift:
                    p_alert[i] = True
                    
            drift_alert |= p_alert
        station_alerts[sid] = drift_alert
        
    all_gt = []
    all_pred = []
    all_ft = []
    for sid, df in benchmark_b.items():
        all_gt.extend(df['is_anomaly'].tolist())
        all_pred.extend(station_alerts[sid].tolist())
        all_ft.extend(df['fault_type'].fillna('none').tolist())
        
    gt = np.array(all_gt, dtype=bool)
    pred = np.array(all_pred, dtype=bool)
    ft = np.array(all_ft)
    
    drift_gt = (ft == 'drift')
    clean_gt = (~gt)
    
    tp_drift = (drift_gt & pred).sum()
    fp_drift = (clean_gt & pred).sum()
    
    print(f"Seed {seed}:")
    print(f"  Drift TP={tp_drift}/{drift_gt.sum()} (Recall: {tp_drift/drift_gt.sum()*100:.2f}%), Clean FP={fp_drift} (Clean FP Rate: {fp_drift/clean_gt.sum()*100:.2f}%)")
    print(f"  Drift Precision: {tp_drift/(tp_drift+fp_drift)*100:.2f}%")

if __name__ == '__main__':
    for s in [42, 101, 202, 2024, 8888, 20260924, 45456231412727229999]:
        test_consensus_drift(s)
