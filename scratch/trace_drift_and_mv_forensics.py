import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel, ConditionalMultivariateDriftDetector,
    STATION_TO_CLUSTER
)

def trace_drift_and_mv_fps(seed=42):
    print("=== TRACING DRIFT, MULTIVARIATE & FROZEN FPS ON SEED 42 ===")
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    theta_glrt = 30.0
    theta_cusum = 24.0
    detector = ConditionalMultivariateDriftDetector(glrt_thresh=theta_glrt, cusum_thresh=theta_cusum, allowance=0.45, persist_req=2)

    data = generate_network_benchmark(regime='observable_v1', seed=seed, save_to_disk=False)

    drift_records = []
    for sid in STATION_TO_CLUSTER:
        df = data[sid].sort_values('timestamp').reset_index(drop=True)
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: data[pid].sort_values('timestamp').reset_index(drop=True) for pid in peer_ids}
        hr = uncertainty_model.get_hour_regime(pd.to_datetime(df['timestamp']).dt.hour.values)

        Z_list = []
        for p in joint_model.channels:
            y_hat, _ = normal_model.predict_target_robust(sid, p, df, peer_dfs)
            r = df[p].values - y_hat
            sig = uncertainty_model.predict_sigma(sid, p, y_hat, df['timestamp'].values)
            Z_list.append(np.nan_to_num(r / np.maximum(0.1, sig), nan=0.0))
        Z = np.column_stack(Z_list)
        n = len(Z)

        u_T = np.zeros(n)
        sig_cond_T = np.ones(n)
        for i in range(n):
            u, s = joint_model.get_conditional_innovation(sid, hr[i], Z[i], 0)
            u_T[i] = u
            sig_cond_T[i] = s

        res = detector.process_station_stream(u_T, sig_cond_T, df['timestamp'].values)
        preds = res['online_predictions']
        is_ano = df['is_anomaly'].values
        ft = df['fault_type'].values

        for i in range(n):
            if preds[i]:
                drift_records.append({
                    'station_id': sid,
                    'timestamp': df.at[i, 'timestamp'],
                    'glrt_lambda': res['glrt_lambdas'][i],
                    'glrt_slope': res['glrt_slopes'][i],
                    'c_plus': res['c_plus'][i],
                    'c_minus': res['c_minus'][i],
                    'u_T': u_T[i],
                    'z_T': Z[i, 0],
                    'z_RH': Z[i, 1],
                    'z_P': Z[i, 2],
                    'coherence': res['coherence'][i],
                    'is_anomaly': is_ano[i],
                    'fault_type': ft[i],
                    'is_drift_tp': is_ano[i] and (ft[i] == 'drift'),
                    'is_fp': not is_ano[i]
                })

    df_d = pd.DataFrame(drift_records)
    print(f"Total Drift Stream Detections: {len(df_d)}")
    d_tps = df_d[df_d['is_drift_tp']]
    d_fps = df_d[df_d['is_fp']]
    print(f"Drift TPs: {len(d_tps)}, Drift FPs: {len(d_fps)}")

    print("\n--- STATISTICAL COMPARISON: DRIFT STREAM TP vs FP ---")
    for col in ['glrt_lambda', 'glrt_slope', 'u_T', 'z_T', 'coherence']:
        print(f"\nMetric: {col}")
        print(f"  TP Mean: {d_tps[col].mean():.3f} (Median: {d_tps[col].median():.3f}, 10th-pct: {d_tps[col].quantile(0.10):.3f})")
        print(f"  FP Mean: {d_fps[col].mean():.3f} (Median: {d_fps[col].median():.3f}, 90th-pct: {d_fps[col].quantile(0.90):.3f})")

    # Check coherence and persistent slope as discriminators
    print("\n--- TESTING DRIFT DISCRIMINATOR GATES ---")
    for min_lam in [20.0, 30.0, 40.0, 50.0, 60.0]:
        tp_k = (d_tps['glrt_lambda'] >= min_lam).sum()
        fp_k = (d_fps['glrt_lambda'] >= min_lam).sum()
        print(f"  GLRT >= {min_lam:.1f} -> TP Kept: {tp_k}/{len(d_tps)} ({tp_k/len(d_tps)*100:.2f}%) | FP Kept: {fp_k}/{len(d_fps)} ({fp_k/len(d_fps)*100:.2f}%)")

if __name__ == '__main__':
    trace_drift_and_mv_fps(seed=42)
