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

class NormalStationBehaviorModel:
    """
    Robust Causal Spatial-Diurnal Model:
    For each target station s and parameter p:
      y_s(t) = sum_j w_j [ alpha_j + beta_j * y_j(t) + gamma_1 * sin(2pi*h/24) + gamma_2 * cos(2pi*h/24) ]
    Where weights w_j are proportional to 1 / sigma_j^2.
    """
    def __init__(self):
        self.models = {}
        self.stats = {}

    def fit_clean_history(self):
        # Load clean historical data for all stations
        clean_dfs = {}
        for sid in STATION_TO_CLUSTER:
            df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            clean_dfs[sid] = df

        params = ['temperature_c', 'pressure_hpa', 'humidity_pct']

        for sid, cid in STATION_TO_CLUSTER.items():
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            df_target = clean_dfs[sid]
            hours = df_target['timestamp'].dt.hour
            sin_h = np.sin(2 * np.pi * hours / 24.0)
            cos_h = np.cos(2 * np.pi * hours / 24.0)

            for p in params:
                peer_regressors = {}
                for pid in peer_ids:
                    df_peer = clean_dfs[pid]
                    # Design matrix: [1, y_peer, sin_h, cos_h]
                    X = np.column_stack([np.ones(len(df_target)), df_peer[p].values, sin_h.values, cos_h.values])
                    y = df_target[p].values

                    # Robust OLS
                    # beta = (X^T X)^{-1} X^T y
                    try:
                        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                        pred = X @ coeffs
                        err = y - pred
                        sigma = np.std(err)
                        mad = np.median(np.abs(err - np.median(err)))
                        peer_regressors[pid] = {
                            "coeffs": coeffs,
                            "sigma": max(0.1, sigma),
                            "mad": max(0.1, mad)
                        }
                    except Exception as e:
                        pass

                # Inverse variance weighting
                if peer_regressors:
                    inv_vars = {pid: 1.0 / (reg["sigma"]**2) for pid, reg in peer_regressors.items()}
                    total_inv = sum(inv_vars.values())
                    weights = {pid: inv_vars[pid] / total_inv for pid in peer_regressors}
                    comb_sigma = 1.0 / np.sqrt(total_inv)
                else:
                    weights = {}
                    comb_sigma = 1.0

                self.models[(sid, p)] = {
                    "peer_regressors": peer_regressors,
                    "weights": weights,
                    "comb_sigma": comb_sigma
                }

    def predict(self, sid, p, target_df, peer_dfs_dict):
        """
        Causally predicts expected target reading y_hat and uncertainty sigma at each timestep.
        """
        model_info = self.models.get((sid, p))
        if not model_info or not model_info["weights"]:
            # fallback
            return target_df[p].values, np.ones(len(target_df))

        hours = pd.to_datetime(target_df['timestamp']).dt.hour
        sin_h = np.sin(2 * np.pi * hours / 24.0).values
        cos_h = np.cos(2 * np.pi * hours / 24.0).values
        n = len(target_df)

        y_hat_comb = np.zeros(n, dtype=float)
        weights = model_info["weights"]

        for pid, w in weights.items():
            reg = model_info["peer_regressors"][pid]
            coeffs = reg["coeffs"]
            peer_val = peer_dfs_dict[pid][p].values
            # X: [1, peer_val, sin_h, cos_h]
            X = np.column_stack([np.ones(n), peer_val, sin_h, cos_h])
            y_hat_p = X @ coeffs
            y_hat_comb += w * y_hat_p

        comb_sigma = model_info["comb_sigma"] * np.ones(n, dtype=float)
        return y_hat_comb, comb_sigma

def evaluate_representation_across_seeds(seeds=[42, 101, 202, 2024, 8888, 20260924]):
    model = NormalStationBehaviorModel()
    model.fit_clean_history()

    print("="*100)
    print("SECTIONS E, F, G: EVALUATING NORMAL STATION-BEHAVIOR MODEL ACROSS SEEDS")
    print("="*100)

    summary_rows = []

    for seed in seeds:
        data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
        
        old_clean_residuals = []
        new_clean_residuals = []
        old_drift_residuals = []
        new_drift_residuals = []

        for sid, df in data.items():
            cid = STATION_TO_CLUSTER[sid]
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            peer_dfs = {pid: data[pid] for pid in peer_ids if pid in data}
            
            for p in ['temperature_c']: # focus on temperature as primary drift parameter
                y_hat, sigma = model.predict(sid, p, df, peer_dfs)
                
                # Raw peer median
                peer_vals = [peer_dfs[pid][p].values for pid in peer_ids if pid in peer_dfs]
                if peer_vals:
                    p_med = np.median(np.column_stack(peer_vals), axis=1)
                else:
                    p_med = df[p].values

                raw_res = df[p].values - p_med
                new_res = df[p].values - y_hat
                
                is_anom = df["is_anomaly"].fillna(False).values.astype(bool)
                is_drift = (df["fault_type"] == "drift").fillna(False).values.astype(bool)
                is_clean = (~is_anom) & (~np.isnan(raw_res)) & (~np.isnan(new_res))

                old_clean_residuals.extend(raw_res[is_clean])
                new_clean_residuals.extend(new_res[is_clean])
                old_drift_residuals.extend(raw_res[is_drift & (~np.isnan(raw_res))])
                new_drift_residuals.extend(new_res[is_drift & (~np.isnan(new_res))])

        old_c = np.array(old_clean_residuals)
        new_c = np.array(new_clean_residuals)
        old_d = np.array(old_drift_residuals)
        new_d = np.array(new_drift_residuals)

        # Statistics
        c_old_std = np.std(old_c)
        c_new_std = np.std(new_c)
        c_old_p95 = np.percentile(np.abs(old_c), 95)
        c_new_p95 = np.percentile(np.abs(new_c), 95)

        d_old_p50 = np.percentile(np.abs(old_d), 50) if len(old_d) > 0 else 0
        d_new_p50 = np.percentile(np.abs(new_d), 50) if len(new_d) > 0 else 0

        summary_rows.append({
            "Seed": seed,
            "Clean Old Std": c_old_std,
            "Clean New Std": c_new_std,
            "Clean Std Reduction": (c_old_std - c_new_std) / c_old_std * 100,
            "Clean Old 95th": c_old_p95,
            "Clean New 95th": c_new_p95,
            "Drift Old Med": d_old_p50,
            "Drift New Med": d_new_p50
        })

    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))
    print("="*100)

def replay_bho_clean_case():
    print("\n" + "="*100)
    print("SECTION H: EXPLICIT REPLAY OF AWS-BHO-030 CLEAN DAYTIME CASE (2025-01-02)")
    print("="*100)
    
    model = NormalStationBehaviorModel()
    model.fit_clean_history()
    
    bho_stations = ['AWS-BHO-030', 'AWS-BHO-101', 'AWS-BHO-102', 'AWS-BHO-103']
    dfs = {s: pd.read_csv(f'data/{s}.csv', parse_dates=['timestamp']) for s in bho_stations}
    for s in dfs:
        dfs[s]['timestamp'] = pd.to_datetime(dfs[s]['timestamp']).dt.tz_localize(None)

    df_target = dfs['AWS-BHO-030']
    peer_dfs = {s: dfs[s] for s in bho_stations if s != 'AWS-BHO-030'}
    
    y_hat, sigma = model.predict('AWS-BHO-030', 'temperature_c', df_target, peer_dfs)
    
    peer_t_vals = [peer_dfs[s]['temperature_c'].values for s in peer_dfs]
    p_med = np.median(np.column_stack(peer_t_vals), axis=1)
    
    df_eval = pd.DataFrame({
        "timestamp": df_target["timestamp"],
        "target_actual": df_target["temperature_c"],
        "peer_median": p_med,
        "old_raw_residual": df_target["temperature_c"] - p_med,
        "new_predicted": y_hat,
        "new_residual": df_target["temperature_c"] - y_hat,
        "standardized_z": (df_target["temperature_c"] - y_hat) / sigma
    })
    
    df_eval = df_eval.set_index("timestamp")
    ts_range = pd.date_range('2025-01-02 08:00:00', '2025-01-02 16:00:00', freq='h')
    print(df_eval.loc[ts_range].to_string())
    print("="*100)

if __name__ == '__main__':
    evaluate_representation_across_seeds()
    replay_bho_clean_case()
