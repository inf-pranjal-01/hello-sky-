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

# First, precompute clean diurnal peer difference profile for all stations from clean CSVs
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
        # Mean diff per hour
        hourly_profile = diff.groupby(hours).mean()
        hourly_std = diff.groupby(hours).std()
        clean_peer_diurnal[(sid, param)] = (hourly_profile, hourly_std)

def test_diurnal_peer_cusum(seed=42):
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
            std_diff = d_hours.map(exp_std).fillna(1.0).clip(lower=0.3)
            
            # True normalized diurnal-subtracted peer residual
            res_peer = (raw_peer_diff - exp_diff) / std_diff
            
            # CUSUM on diurnal peer residual
            splus, sminus = 0.0, 0.0
            splus_arr, sminus_arr = [], []
            streak = 0
            streak_arr = []
            allowance = 0.5
            
            for v in res_peer:
                if np.isnan(v):
                    splus, sminus, streak = 0.0, 0.0, 0
                else:
                    splus = max(0.0, splus + v - allowance)
                    sminus = max(0.0, sminus - v - allowance)
                    if abs(v) > 1.5:
                        streak += 1
                    else:
                        streak = max(0, streak - 1)
                        
                splus_arr.append(splus)
                sminus_arr.append(sminus)
                streak_arr.append(streak)
                
            d[f'{param}_peer_splus'] = splus_arr
            d[f'{param}_peer_sminus'] = sminus_arr
            d[f'{param}_peer_streak'] = streak_arr
            d[f'{param}_peer_z'] = res_peer
            
        all_frames.append(d)
        
    df_all = pd.concat(all_frames, ignore_index=True)
    drift_mask = df_all['fault_type'] == 'drift'
    clean_mask = ~df_all['is_anomaly']
    
    print(f"\n--- DIURNAL PEER CUSUM (SEED {seed}) ---")
    for cusum_thresh in [8.0, 10.0, 12.0, 15.0, 18.0, 20.0]:
        hit = pd.Series(False, index=df_all.index)
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            p_hit = (
                ((df_all[f'{param}_peer_splus'] > cusum_thresh) | (df_all[f'{param}_peer_sminus'] > cusum_thresh)) &
                (df_all[f'{param}_peer_z'].abs() > 2.0)
            )
            hit = hit | p_hit
            
        tp_dr = (drift_mask & hit).sum()
        fp_dr = (clean_mask & hit).sum()
        prec_dr = tp_dr / (tp_dr + fp_dr) if (tp_dr + fp_dr) > 0 else 0
        rec_dr = tp_dr / drift_mask.sum()
        print(f"CUSUM>{cusum_thresh:<4.1f} -> Drift TP: {tp_dr:<4}/{drift_mask.sum()} (Rec: {rec_dr*100:<5.1f}%), Clean FP: {fp_dr:<4} (Drift Prec: {prec_dr*100:<5.1f}%)")

if __name__ == '__main__':
    for s in [42, 101, 202]:
        test_diurnal_peer_cusum(s)
