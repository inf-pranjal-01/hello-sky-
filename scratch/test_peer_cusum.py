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

def test_peer_cusum(seed=42):
    benchmark_b = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    all_frames = []
    for sid, df in benchmark_b.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        
        d = df.copy()
        d['station_id'] = sid
        
        # We track peer divergence CUSUM for each parameter
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            peer_vals = pd.DataFrame({p: benchmark_b[p][param] for p in peer_ids if p in benchmark_b})
            peer_med = peer_vals.median(axis=1)
            peer_diff = d[param] - peer_med
            
            # Baseline expected offset (e.g. 72h median)
            base_offset = peer_diff.rolling(96, min_periods=24).median().shift(1).bfill()
            divergence = peer_diff - base_offset
            
            # Normalization scale
            scale = 1.0 if param == 'temperature_c' else (0.5 if param == 'pressure_hpa' else 3.0)
            norm_div = divergence / scale
            
            # Causal persistence: CUSUM on divergence
            splus, sminus = 0.0, 0.0
            splus_arr, sminus_arr = [], []
            streak = 0
            streak_arr = []
            
            for v in norm_div:
                if np.isnan(v):
                    splus_arr.append(0.0)
                    sminus_arr.append(0.0)
                    streak_arr.append(0)
                    continue
                
                # CUSUM accumulation
                splus = max(0.0, splus + (v - 0.5))
                sminus = max(0.0, sminus + (-v - 0.5))
                
                # Persistence streak
                if abs(v) > 1.0:
                    streak += 1
                else:
                    streak = max(0, streak - 1)
                    
                splus_arr.append(splus)
                sminus_arr.append(sminus)
                streak_arr.append(streak)
                
            d[f'{param}_splus'] = splus_arr
            d[f'{param}_sminus'] = sminus_arr
            d[f'{param}_streak'] = streak_arr
            d[f'{param}_div'] = norm_div
            
        all_frames.append(d)
        
    df_all = pd.concat(all_frames, ignore_index=True)
    drift_mask = df_all['fault_type'] == 'drift'
    clean_mask = ~df_all['is_anomaly']
    
    print(f"\n--- SEED {seed} PEER CUSUM EVALUATION ---")
    for streak_thresh in [3, 4, 5]:
        for cusum_thresh in [3.0, 4.0, 5.0, 6.0]:
            hit = pd.Series(False, index=df_all.index)
            for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
                p_hit = (
                    (df_all[f'{param}_streak'] >= streak_thresh) &
                    ((df_all[f'{param}_splus'] > cusum_thresh) | (df_all[f'{param}_sminus'] > cusum_thresh)) &
                    (df_all[f'{param}_div'].abs() > 1.2)
                )
                hit = hit | p_hit
            
            tp_dr = (drift_mask & hit).sum()
            fp_dr = (clean_mask & hit).sum()
            prec_dr = tp_dr / (tp_dr + fp_dr) if (tp_dr + fp_dr) > 0 else 0
            rec_dr = tp_dr / drift_mask.sum()
            print(f"Streak>={streak_thresh}, CUSUM>{cusum_thresh:.1f} -> Drift TP: {tp_dr}/{drift_mask.sum()} (Rec: {rec_dr:.1%}), Clean FP: {fp_dr} (Drift Prec: {prec_dr:.1%})")

if __name__ == '__main__':
    for s in [42, 101, 202]:
        test_peer_cusum(s)
