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

# Precompute clean diurnal peer profile
clean_peer_diurnal = {}
for sid in STATION_TO_CLUSTER:
    cid = STATION_TO_CLUSTER[sid]
    df_s = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
    peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
    
    for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
        peer_dfs = [pd.read_csv(f'data/{p}.csv', parse_dates=['timestamp'])[param] for p in peer_ids]
        peer_med = pd.concat(peer_dfs, axis=1).median(axis=1)
        diff = df_s[param] - peer_med
        hours = df_s['timestamp'].dt.hour
        hourly_profile = diff.groupby(hours).mean()
        hourly_std = diff.groupby(hours).std()
        clean_peer_diurnal[(sid, param)] = (hourly_profile, hourly_std)

def test_refined_drift(seed=42):
    benchmark_b = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    all_frames = []
    for sid, df in benchmark_b.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        
        d = df.copy()
        d['station_id'] = sid
        d_hours = pd.to_datetime(d['timestamp']).dt.hour
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            peer_vals = pd.DataFrame({p: benchmark_b[p][param] for p in peer_ids if p in benchmark_b})
            peer_med = peer_vals.median(axis=1)
            raw_peer_diff = d[param] - peer_med
            
            exp_profile, exp_std = clean_peer_diurnal[(sid, param)]
            exp_diff = d_hours.map(exp_profile)
            std_diff = d_hours.map(exp_std).fillna(1.0).clip(lower=0.4)
            
            res_peer = (raw_peer_diff - exp_diff) / std_diff
            
            # CUSUM on persistent separation with slow leak (0.2)
            splus, sminus = 0.0, 0.0
            splus_arr, sminus_arr = [], []
            for v in res_peer:
                if np.isnan(v):
                    splus, sminus = 0.0, 0.0
                else:
                    # Decay towards 0 when returning towards normal
                    splus = max(0.0, splus * 0.95 + v - 0.2) if v > 0 else max(0.0, splus * 0.8)
                    sminus = max(0.0, sminus * 0.95 - v - 0.2) if v < 0 else max(0.0, sminus * 0.8)
                splus_arr.append(splus)
                sminus_arr.append(sminus)
                
            d[f'{param}_psplus'] = splus_arr
            d[f'{param}_psminus'] = sminus_arr
            d[f'{param}_pz'] = res_peer
            
        all_frames.append(d)
        
    df_all = pd.concat(all_frames, ignore_index=True)
    drift_mask = df_all['fault_type'] == 'drift'
    clean_mask = ~df_all['is_anomaly']
    
    print(f"\n--- REFINED DRIFT (SEED {seed}) ---")
    for t_cusum in [5.0, 8.0, 10.0, 12.0, 15.0]:
        for t_z in [1.5, 2.0, 2.5, 3.0]:
            hit = pd.Series(False, index=df_all.index)
            for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
                p_hit = (
                    ((df_all[f'{param}_psplus'] > t_cusum) | (df_all[f'{param}_psminus'] > t_cusum)) &
                    (df_all[f'{param}_pz'].abs() > t_z)
                )
                hit = hit | p_hit
                
            tp_dr = (drift_mask & hit).sum()
            fp_dr = (clean_mask & hit).sum()
            prec_dr = tp_dr / (tp_dr + fp_dr) if (tp_dr + fp_dr) > 0 else 0
            rec_dr = tp_dr / drift_mask.sum()
            if fp_dr < 2000:
                print(f"CUSUM>{t_cusum:<4.1f}, Z>{t_z:.1f} -> Drift TP: {tp_dr:<4}/{drift_mask.sum()} (Rec: {rec_dr*100:<5.1f}%), Clean FP: {fp_dr:<4} (Drift Prec: {prec_dr*100:<5.1f}%)")

if __name__ == '__main__':
    for s in [42, 101, 202]:
        test_refined_drift(s)
