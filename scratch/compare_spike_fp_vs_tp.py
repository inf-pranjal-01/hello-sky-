import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel,
    STATION_TO_CLUSTER
)

def compare_spike_fp_vs_tp(seed=42):
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()

    data = generate_network_benchmark(regime='observable_v1', seed=seed, save_to_disk=False)
    
    records = []
    for sid in STATION_TO_CLUSTER:
        df = data[sid].sort_values('timestamp').reset_index(drop=True)
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: data[pid].sort_values('timestamp').reset_index(drop=True) for pid in peer_ids}

        for p in ['temperature_c', 'humidity_pct', 'pressure_hpa']:
            y_hat, _ = normal_model.predict_target_robust(sid, p, df, peer_dfs)
            vals = df[p].values
            sig = uncertainty_model.predict_sigma(sid, p, y_hat, df['timestamp'].values)
            step_diff = np.abs(np.diff(vals, prepend=vals[0]))
            spatial_resid = np.abs(vals - y_hat)
            z_spatial = spatial_resid / np.maximum(0.1, sig)
            
            is_ano = df['is_anomaly'].values
            ft = df['fault_type'].values

            for i in range(len(df)):
                # If step_diff is large enough to trigger spike rule (> 3.5 C, > 10% RH, > 5 hPa)
                thresh = 3.67 if p == 'temperature_c' else (10.92 if p == 'humidity_pct' else 5.60)
                if step_diff[i] >= thresh:
                    records.append({
                        'station_id': sid,
                        'param': p,
                        'timestamp': df.at[i, 'timestamp'],
                        'hour': pd.to_datetime(df.at[i, 'timestamp']).hour,
                        'step_diff': step_diff[i],
                        'spatial_resid': spatial_resid[i],
                        'z_spatial': z_spatial[i],
                        'is_anomaly': is_ano[i],
                        'fault_type': ft[i],
                        'is_spike_tp': is_ano[i] and (ft[i] == 'spike'),
                        'is_fp': not is_ano[i]
                    })

    df_spikes = pd.DataFrame(records)
    print(f"Total candidate spike triggers: {len(df_spikes)}")
    tps = df_spikes[df_spikes['is_spike_tp']]
    fps = df_spikes[df_spikes['is_fp']]
    print(f"Spike TPs: {len(tps)}, Spike FPs: {len(fps)}")

    print("\n--- STATISTICAL COMPARISON: SPIKE TP vs SPIKE FP ---")
    for col in ['step_diff', 'spatial_resid', 'z_spatial']:
        print(f"\nMetric: {col}")
        print(f"  TP Mean: {tps[col].mean():.3f} (Median: {tps[col].median():.3f}, 10th-pct: {tps[col].quantile(0.10):.3f}, Min: {tps[col].min():.3f})")
        print(f"  FP Mean: {fps[col].mean():.3f} (Median: {fps[col].median():.3f}, 90th-pct: {fps[col].quantile(0.90):.3f}, Max: {fps[col].max():.3f})")

    # Evaluate simple spatial corroboration gate on spike
    # A true spike must have spatial deviation as well as step jump!
    print("\n--- TESTING SPATIAL RESIDUAL DISCRIMINATOR FOR SPIKE ---")
    for z_cut in [1.5, 2.0, 2.5, 3.0, 3.5]:
        tp_kept = (tps['z_spatial'] >= z_cut).sum()
        fp_kept = (fps['z_spatial'] >= z_cut).sum()
        print(f"  Cutoff z_spatial >= {z_cut:.1f} -> TP Kept: {tp_kept}/{len(tps)} ({tp_kept/len(tps)*100:.2f}%) | FP Filtered: {len(fps)-fp_kept}/{len(fps)} ({(len(fps)-fp_kept)/len(fps)*100:.2f}%)")

if __name__ == '__main__':
    compare_spike_fp_vs_tp(seed=42)
