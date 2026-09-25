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
    ConditionalMultivariateJointModel, STATION_TO_CLUSTER
)

def compute_psychrometric_features(df):
    """
    Computes exact thermodynamic and psychrometric variables:
    - Saturation vapor pressure: e_sat(T) = 6.112 * exp(17.67 * T / (T + 243.5)) in hPa (Tetens formula)
    - Actual vapor pressure: e_act = (RH / 100) * e_sat(T)
    - Dew point: T_dew = (243.5 * ln(e_act/6.112)) / (17.67 - ln(e_act/6.112))
    - Psychrometric Defect: d(e_act)/dt vs d(T)/dt consistency
    """
    T = df['temperature_c'].values
    RH = np.clip(df['humidity_pct'].values, 1.0, 100.0)
    P = df['pressure_hpa'].values

    e_sat = 6.112 * np.exp(17.67 * T / (T + 243.5))
    e_act = (RH / 100.0) * e_sat
    
    # Specific humidity q = 0.622 * e_act / (P - 0.378 * e_act) * 1000 (g/kg)
    q = 622.0 * e_act / np.maximum(100.0, P - 0.378 * e_act)

    # Dew point
    ln_term = np.log(np.maximum(1e-4, e_act / 6.112))
    T_dew = (243.5 * ln_term) / np.maximum(0.1, 17.67 - ln_term)

    # Psychrometric coupled residual:
    # Under constant moisture, if T rises by dT, RH must drop by dRH = - (RH / (T+243.5)) * (17.67 * 243.5 / (T+243.5)) * dT
    # If BOTH T rises AND RH rises, delta_CC = dRH - expected_dRH is strongly positive!
    dT = np.diff(T, prepend=T[0])
    dRH = np.diff(RH, prepend=RH[0])
    expected_dRH_iso = - (RH / (T + 243.5)) * (4302.6 / (T + 243.5)) * dT
    cc_violation_step = dRH - expected_dRH_iso

    return pd.DataFrame({
        'e_sat': e_sat,
        'e_act': e_act,
        'q': q,
        'T_dew': T_dew,
        'dT': dT,
        'dRH': dRH,
        'cc_violation_step': cc_violation_step
    })

def analyze_mv_and_drift_signatures(seed=42):
    print("=== PHASE 2 & 6: DEEP MULTIVARIATE & DRIFT FORENSIC TRACE ===")
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    data = generate_network_benchmark(regime='observable_v1', seed=seed, save_to_disk=False)

    records = []
    for sid in STATION_TO_CLUSTER:
        df = data[sid].sort_values('timestamp').reset_index(drop=True)
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: data[pid].sort_values('timestamp').reset_index(drop=True) for pid in peer_ids}

        # Spatial residuals
        y_hat_T, _ = normal_model.predict_target_robust(sid, 'temperature_c', df, peer_dfs)
        y_hat_RH, _ = normal_model.predict_target_robust(sid, 'humidity_pct', df, peer_dfs)
        y_hat_P, _ = normal_model.predict_target_robust(sid, 'pressure_hpa', df, peer_dfs)

        sig_T = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat_T, df['timestamp'].values)
        sig_RH = uncertainty_model.predict_sigma(sid, 'humidity_pct', y_hat_RH, df['timestamp'].values)
        sig_P = uncertainty_model.predict_sigma(sid, 'pressure_hpa', y_hat_P, df['timestamp'].values)

        z_T = (df['temperature_c'].values - y_hat_T) / np.maximum(0.1, sig_T)
        z_RH = (df['humidity_pct'].values - y_hat_RH) / np.maximum(0.1, sig_RH)
        z_P = (df['pressure_hpa'].values - y_hat_P) / np.maximum(0.1, sig_P)

        psych = compute_psychrometric_features(df)
        
        # Cross-channel joint physical residual:
        # In MV fault, both z_T and z_RH are simultaneously large and positive!
        mv_coupled_z = z_T * z_RH  # Strongly positive for MV fault, near zero/negative for clean weather
        
        # Specific humidity defect against spatial peers:
        # In MV fault, specific humidity jumps by 40-80% above peer expected moisture
        df_clean = df.copy()
        for p in ['temperature_c', 'humidity_pct', 'pressure_hpa']:
            df_clean[p] = df_clean[p].ffill().bfill()

        # Build records
        for i in range(len(df)):
            records.append({
                'station_id': sid,
                'timestamp': df.at[i, 'timestamp'],
                'fault_type': df.at[i, 'fault_type'] if 'fault_type' in df.columns else 'none',
                'is_anomaly': df.at[i, 'is_anomaly'] if 'is_anomaly' in df.columns else False,
                'z_T': z_T[i],
                'z_RH': z_RH[i],
                'z_P': z_P[i],
                'mv_coupled_z': mv_coupled_z[i],
                'cc_violation_step': psych.at[i, 'cc_violation_step'],
                'e_act': psych.at[i, 'e_act'],
                'q': psych.at[i, 'q'],
            })

    res_df = pd.DataFrame(records)
    print(f"Total evaluated rows: {len(res_df)}")
    
    mv_rows = res_df[res_df['fault_type'] == 'multivariate_inconsistency']
    drift_rows = res_df[res_df['fault_type'] == 'drift']
    norm_rows = res_df[res_df['fault_type'] == 'none']

    print(f"\n--- 1. MULTIVARIATE FAULT (N={len(mv_rows)}) vs CLEAN WEATHER (N={len(norm_rows)}) ---")
    for col in ['z_T', 'z_RH', 'mv_coupled_z', 'cc_violation_step', 'q']:
        print(f"\nFeature: {col}")
        print(f"  MV Fault: Mean={mv_rows[col].mean():.3f} | Med={mv_rows[col].median():.3f} | 10th-pct={mv_rows[col].quantile(0.10):.3f}")
        print(f"  Clean Weather: Mean={norm_rows[col].mean():.3f} | Med={norm_rows[col].median():.3f} | 90th-pct={norm_rows[col].quantile(0.90):.3f} | 99th-pct={norm_rows[col].quantile(0.99):.3f}")

    print("\n--- 2. DRIFT FAULT (N={len(drift_rows)}) vs CLEAN WEATHER ---")
    for col in ['z_T', 'z_RH', 'z_P']:
        print(f"\nFeature: {col}")
        print(f"  Drift Fault: Mean={drift_rows[col].abs().mean():.3f} | Med={drift_rows[col].abs().median():.3f} | 10th-pct={drift_rows[col].abs().quantile(0.10):.3f}")
        print(f"  Clean Weather: Mean={norm_rows[col].abs().mean():.3f} | Med={norm_rows[col].abs().median():.3f} | 90th-pct={norm_rows[col].abs().quantile(0.90):.3f}")

    # Test spatial coupled discriminator for MV
    print("\n--- TESTING MULTIVARIATE DISCRIMINATOR: mv_coupled_z >= threshold ---")
    for cut in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        mv_k = (mv_rows['mv_coupled_z'] >= cut).sum()
        norm_k = (norm_rows['mv_coupled_z'] >= cut).sum()
        print(f"  mv_coupled_z >= {cut:.1f} -> MV Recall: {mv_k}/{len(mv_rows)} ({mv_k/len(mv_rows)*100:.1f}%) | Clean FPs: {norm_k}/{len(norm_rows)} ({norm_k/len(norm_rows)*100:.2f}%)")

if __name__ == '__main__':
    analyze_mv_and_drift_signatures(seed=42)
