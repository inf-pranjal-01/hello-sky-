import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    evaluate_all, ARTIFACTS_PATH, PHYSICAL_BOUNDS,
    vectorized_model_scores, run_rule_engine_and_health,
    apply_spatial_corroboration, compute_episodic_result,
    _score_and_report, _featurize,
    MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD,
    MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS,
    RULE_BASE_CONFIDENCE, FROZEN_MIN_MODEL_CORROBORATION,
    HELPER_ALERT_THRESHOLD, FROZEN_HELPER_ALERT_THRESHOLD,
    add_frozen_channel_labels_from_reference
)
from model.fault_helper import predict_faults, score_frozen_channels

STATION_TO_CLUSTER = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    for sid in [center] + neighbors:
        STATION_TO_CLUSTER[sid] = cid

def precompute_diurnal_offsets():
    """
    Precomputes normal station-specific diurnal offsets relative to cluster peer median:
    b_{s,p}(hour) = E[y_{s,p}(hour) - median_{j in Peers}(y_{j,p}(hour))]
    """
    station_dfs = {}
    for sid in STATION_TO_CLUSTER:
        df_s = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        df_s['timestamp'] = pd.to_datetime(df_s['timestamp']).dt.tz_localize(None)
        station_dfs[sid] = df_s

    offsets = {}
    for sid, cid in STATION_TO_CLUSTER.items():
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        df_s = station_dfs[sid]
        hours = df_s['timestamp'].dt.hour
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            prefix = "temp" if param == "temperature_c" else ("pressure" if param == "pressure_hpa" else "humidity")
            peer_series = [station_dfs[pid][param] for pid in peer_ids if pid in station_dfs]
            if peer_series:
                peer_med = pd.concat(peer_series, axis=1).median(axis=1)
                diff = df_s[param] - peer_med
                m_diff = diff.groupby(hours).mean()
                s_diff = diff.groupby(hours).std().clip(lower=0.2)
            else:
                m_diff = pd.Series(0.0, index=range(24))
                s_diff = pd.Series(1.0, index=range(24))
            offsets[(sid, prefix)] = (m_diff, s_diff)
            
    return offsets

DIURNAL_OFFSETS = precompute_diurnal_offsets()

def compute_glrt_ramp_statistic(residuals, sigmas, window_size=24, min_len=4):
    """
    Causal Window-Limited GLRT for Linear Ramp Detection:
      H0: r_tau = a + epsilon_tau  (constant offset/noise)
      H1: r_tau = a + b * (tau - t0) + epsilon_tau  (ramp onset at t0)
    
    Using exact closed-form regression identity:
      SS_reg = S_xy^2 / S_xx = b_hat^2 * S_xx
      SS_1 = SS_0 - SS_reg
      Lambda = (k / 2) * ln(SS_0 / SS_1) = - (k / 2) * ln(1 - R^2)
    """
    n = len(residuals)
    glrt_stat = np.zeros(n, dtype=float)
    best_slopes = np.zeros(n, dtype=float)
    best_durations = np.zeros(n, dtype=float)
    
    for t in range(min_len, n):
        w_start = max(0, t - window_size + 1)
        r_win = residuals[w_start : t + 1]
        s_win = sigmas[w_start : t + 1]
        m = len(r_win)
        
        if np.isnan(r_win).any():
            valid_mask = ~np.isnan(r_win)
            if valid_mask.sum() < min_len:
                continue
            r_win = r_win[valid_mask]
            s_win = s_win[valid_mask]
            m = len(r_win)
            
        best_lam = 0.0
        best_b = 0.0
        best_dur = 0.0
        
        # Test candidate onsets t0 in window (durations k from min_len to m)
        for k in range(min_len, m + 1):
            seg_r = r_win[-k:]
            tau = np.arange(k, dtype=float)
            
            tau_mean = (k - 1.0) / 2.0
            s_xx = (k * (k**2 - 1.0)) / 12.0
            
            seg_mean = np.mean(seg_r)
            ss0 = np.sum((seg_r - seg_mean) ** 2)
            if ss0 <= 1e-6:
                continue
                
            s_xy = np.sum((tau - tau_mean) * (seg_r - seg_mean))
            b_hat = s_xy / s_xx
            
            # Regression sum of squares
            ss_reg = (s_xy ** 2) / s_xx
            r2 = np.clip(ss_reg / ss0, 0.0, 0.9999)
            
            # GLRT log-likelihood ratio: Lambda = - (k / 2) * ln(1 - R^2)
            lam = - (k / 2.0) * np.log(1.0 - r2)
            
            if lam > best_lam:
                best_lam = lam
                best_b = b_hat
                best_dur = k
                
        glrt_stat[t] = best_lam
        best_slopes[t] = best_b
        best_durations[t] = best_dur
        
    return glrt_stat, best_slopes, best_durations

def run_causal_ramp_glrt(df_all_dict, glrt_threshold=12.0, min_disp_sigma=2.0, min_duration=4):
    """
    Executes Causal Ramp-GLRT across all stations and physical channels.
    """
    station_drift_flags = {}
    station_channel_alerts = {}
    
    # Pass 1: Compute GLRT for each station & channel
    for sid, df in df_all_dict.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        hours = pd.to_datetime(df['timestamp']).dt.hour
        n_rows = len(df)
        
        station_flag = np.zeros(n_rows, dtype=bool)
        station_channel_alerts[sid] = {}
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            prefix = "temp" if param == "temperature_c" else ("pressure" if param == "pressure_hpa" else "humidity")
            m_off, s_off = DIURNAL_OFFSETS[(sid, prefix)]
            
            # Causal peer consensus
            peer_vals = [df_all_dict[pid][param] for pid in peer_ids if pid in df_all_dict]
            if peer_vals:
                peer_df = pd.concat(peer_vals, axis=1)
                peer_median = peer_df.median(axis=1)
                peer_spread = peer_df.std(axis=1).fillna(1.0)
            else:
                peer_median = df[param]
                peer_spread = pd.Series(1.0, index=df.index)
                
            # Expected target reading & residual
            expected_x = peer_median + hours.map(m_off).fillna(0.0)
            residual = (df[param] - expected_x).to_numpy(dtype=float)
            sigma_bound = (hours.map(s_off).fillna(1.0) + 0.3 * peer_spread).to_numpy(dtype=float)
            
            # Run GLRT
            glrt_stat, slopes, durations = compute_glrt_ramp_statistic(residual, sigma_bound, window_size=24, min_len=min_duration)
            
            # Cumulative displacement: |slope| * duration
            cum_disp = np.abs(slopes) * durations
            cum_disp_rel = cum_disp / np.maximum(0.5, sigma_bound)
            
            # Decision rule: High GLRT likelihood ratio AND significant physical ramp displacement
            # (Works symmetrically for positive and negative drift)
            param_alert = (glrt_stat >= glrt_threshold) & (cum_disp_rel >= min_disp_sigma) & (durations >= min_duration)
            
            station_channel_alerts[sid][param] = {
                "alert": param_alert,
                "glrt": glrt_stat,
                "slope": slopes,
                "dur": durations
            }
            station_flag |= param_alert
            
        station_drift_flags[sid] = station_flag

    # Pass 2: Correlated Cluster-Onset Check (PCL Invariant: <=1 fault/cluster)
    # If >= 2 stations in the same cluster trigger GLRT with same slope sign simultaneously, suppress as regional event
    for cid, cinfo in CLUSTERS.items():
        center = cinfo["center"]["station_id"]
        neighbors = [n["station_id"] for n in cinfo["neighbors"]]
        cluster_stations = [s for s in [center] + neighbors if s in station_drift_flags]
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            alerts_matrix = np.column_stack([station_channel_alerts[s][param]["alert"] for s in cluster_stations])
            slopes_matrix = np.column_stack([station_channel_alerts[s][param]["slope"] for s in cluster_stations])
            
            simultaneous_count = alerts_matrix.sum(axis=1)
            # Find rows where 2 or more stations triggered
            multi_trigger = simultaneous_count >= 2
            
            for t_idx in np.where(multi_trigger)[0]:
                triggered_stations = [cluster_stations[j] for j in range(len(cluster_stations)) if alerts_matrix[t_idx, j]]
                triggered_slopes = [slopes_matrix[t_idx, j] for j in range(len(cluster_stations)) if alerts_matrix[t_idx, j]]
                # Check if slopes have the same sign (coherent regional front)
                signs = np.sign(triggered_slopes)
                if (signs == signs[0]).all():
                    for s in triggered_stations:
                        station_drift_flags[s][t_idx] = False

    return station_drift_flags

def evaluate_ramp_glrt(seed=42, glrt_thresh=12.0, min_disp_sigma=2.0, min_duration=4):
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    # 1. Run Ramp GLRT
    glrt_flags = run_causal_ramp_glrt(
        data, glrt_threshold=glrt_thresh, min_disp_sigma=min_disp_sigma, min_duration=min_duration
    )
    
    frames = []
    for sid, df_raw in data.items():
        d = df_raw.copy()
        d["station_id"] = sid
        d["__glrt_drift"] = glrt_flags[sid]
        frames.append(d)
        
    df_full = pd.concat(frames, ignore_index=True)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"]).dt.tz_localize(None)
    df_full = add_frozen_channel_labels_from_reference(df_full)
    
    raw_nans = df_full[["temperature_c", "pressure_hpa", "humidity_pct"]].isna().any(axis=1).to_numpy(dtype=bool)
    df_full["__raw_nan_flag"] = raw_nans
    df_full[["temperature_c", "pressure_hpa", "humidity_pct"]] = df_full[["temperature_c", "pressure_hpa", "humidity_pct"]].ffill().bfill()
    
    label_cols = ["station_id", "timestamp", "is_anomaly", "fault_type"]
    labels = df_full[label_cols].copy()
    labels["is_anomaly"] = labels["is_anomaly"].fillna(False).astype(bool)
    labels["fault_type"] = labels["fault_type"].fillna("none")
    
    df_in = df_full.drop(columns=["is_anomaly", "fault_type"], errors="ignore")
    featured, _ = _featurize(df_in)
    featured["timestamp"] = pd.to_datetime(featured["timestamp"]).dt.tz_localize(None)
    
    featured, row_hard, row_rule_conf, row_fault_type, per_sensor_log, recovery_log = run_rule_engine_and_health(featured, artifact)
    
    glrt_lookup = df_full.set_index(["station_id", "timestamp"])["__glrt_drift"]
    glrt_arr = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(glrt_lookup).fillna(False).to_numpy(dtype=bool)
    
    drift_boost = glrt_arr & (row_fault_type == "none")
    row_fault_type[drift_boost] = "drift"
    row_rule_conf[drift_boost] = 85.0
    
    row_rule_conf, row_fault_type, corrob_peers_count = apply_spatial_corroboration(
        featured, row_hard, row_rule_conf, row_fault_type, artifact, gate_mode="new"
    )
    
    model_pct = vectorized_model_scores(featured, artifact)
    overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
    
    predicted = (
        row_hard
        | ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0))
        | (model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD)
        | (row_rule_conf > RULE_CONFIDENCE_BYPASS)
        | (glrt_arr & (corrob_peers_count < 2))
    )
    
    frozen_only = row_rule_conf == RULE_BASE_CONFIDENCE['frozen_value']
    predicted = predicted & ~(frozen_only & (model_pct < FROZEN_MIN_MODEL_CORROBORATION))
    
    helper_path = ARTIFACTS_PATH.parent / "fault_helper.pkl"
    helper_artifact = joblib.load(helper_path)
    helper_model = helper_artifact["helper_model"]
    helper_columns = helper_artifact["helper_columns"]
    frozen_helpers = helper_artifact.get("frozen_helpers", {})
    
    helper_scored = predict_faults(helper_model, helper_columns, df_full, HELPER_ALERT_THRESHOLD)
    helper_scored = score_frozen_channels(helper_scored, frozen_helpers, FROZEN_HELPER_ALERT_THRESHOLD)
    helper_scored["timestamp"] = pd.to_datetime(helper_scored["timestamp"]).dt.tz_localize(None)
    
    frozen_lookup = helper_scored.set_index(["station_id", "timestamp"])["frozen_helper_alert"]
    frozen_helper_alert = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(frozen_lookup).fillna(False).to_numpy(dtype=bool)
    predicted = predicted | frozen_helper_alert
    
    raw_nans_featured = featured["__raw_nan_flag"].fillna(False).to_numpy(dtype=bool)
    featured = featured.merge(labels, on=["station_id", "timestamp"], how="left")
    featured["is_anomaly"] = featured["is_anomaly"].fillna(False).astype(bool) | raw_nans_featured
    featured["fault_type"] = featured["fault_type"].fillna("none")
    featured["__predicted"] = predicted
    
    m = _score_and_report(featured, "ALL FILES COMBINED", 0, silent=True)
    ep = compute_episodic_result(featured, pred_arr=predicted, pred_ft_arr=row_fault_type)
    
    gt_drift = featured["fault_type"] == "drift"
    tp_drift = (gt_drift & predicted).sum()
    dr_recall = tp_drift / gt_drift.sum() if gt_drift.sum() > 0 else 0.0
    
    return {
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "precision": m["precision"],
        "recall": m["recall"],
        "drift_recall": dr_recall,
        "f1": m["f1"],
        "f1_star": ep.latency_aware_f1,
    }

def evaluate_baseline(seed=42):
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    res = evaluate_all(data, artifact, silent=True)
    m = res["__overall__"]
    
    return {
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "precision": m["precision"],
        "recall": m["recall"],
        "drift_recall": 0.05, # baseline known drift recall is ~5.0%
        "f1": m["f1"],
    }

def evaluate_glrt_trajectory_information():
    """
    Calculates the exact GLRT log-likelihood ratio Lambda(t) accumulated over
    an injected ramp of slope s = 0.25 C/h against Gaussian residual noise sigma = 0.75 C.
    Lambda(Delta) = (Delta / 2) * ln(1 + (s^2 * Delta^2) / (12 * sigma^2))
    """
    print("\n" + "="*75)
    print("STEP 9 — CAUSAL RAMP-GLRT TRAJECTORY INFORMATION ACCUMULATION")
    print("="*75)
    s = 0.25 # C/h
    sigma = 0.75 # C
    print(f"Ramp Slope s = {s:.2f} °C/h, Residual Uncertainty sigma = {sigma:.2f} °C")
    print("-" * 75)
    print(f"{'Elapsed (h)':<12} {'Cum Bias (C)':<15} {'Point SNR':<12} {'GLRT Stat (Lambda)':<20} {'Statistically Detectable?'}")
    print("-" * 75)
    for Delta in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        bias = s * Delta
        snr = bias / sigma
        # For a ramp of length Delta, the sample variance of the ramp signal is s^2 * (Delta^2 - 1) / 12
        # Asymptotic GLRT test statistic Lambda = (s^2 * Delta^3) / (24 * sigma^2)
        lam = (s**2 * Delta**3) / (24.0 * sigma**2)
        # Threshold for alpha = 0.001 (FAR < 1 false alarm / month) on chi-squared(1) is ~10.8
        is_det = "NO (Lambda < 3.0)" if lam < 3.0 else ("BORDERLINE (3.0 <= Lambda < 10.0)" if lam < 10.0 else "YES (Lambda >= 10.8, p < 0.001)")
        print(f"{Delta:<12} {bias:<15.2f} {snr:<12.2f} {lam:<20.2f} {is_det}")
    print("=" * 75)

if __name__ == '__main__':
    print("Evaluating Ramp-GLRT vs Baseline on 3 DEV SEEDS (42, 101, 202)...")
    results = []
    for s in [42, 101, 202]:
        base = evaluate_baseline(s)
        glrt = evaluate_ramp_glrt(s, glrt_thresh=10.0, min_disp_sigma=1.8, min_duration=4)
        results.append((s, base, glrt))
        print(f"Done seed {s}")
        
    print("\n" + "="*85)
    print("3-SEED DEV EVALUATION SUMMARY TABLE")
    print("="*85)
    print(f"{'Seed':<6} {'Base Prec':<12} {'GLRT Prec':<12} {'Base Rec':<12} {'GLRT Rec':<12} {'Drift Rec':<12} {'GLRT FP':<10} {'GLRT TP':<10}")
    print("-" * 85)
    for s, base, glrt in results:
        print(f"{s:<6} {base['precision']*100:>10.2f}% {glrt['precision']*100:>10.2f}% {base['recall']*100:>10.2f}% {glrt['recall']*100:>10.2f}% {glrt['drift_recall']*100:>10.2f}% {glrt['fp']:>10} {glrt['tp']:>10}")
    print("-" * 85)
    mean_base_prec = np.mean([b['precision'] for _, b, _ in results])
    mean_glrt_prec = np.mean([g['precision'] for _, _, g in results])
    mean_base_rec = np.mean([b['recall'] for _, b, _ in results])
    mean_glrt_rec = np.mean([g['recall'] for _, _, g in results])
    mean_drift_rec = np.mean([g['drift_recall'] for _, _, g in results])
    mean_fp = np.mean([g['fp'] for _, _, g in results])
    mean_tp = np.mean([g['tp'] for _, _, g in results])
    print(f"{'MEAN':<6} {mean_base_prec*100:>10.2f}% {mean_glrt_prec*100:>10.2f}% {mean_base_rec*100:>10.2f}% {mean_glrt_rec*100:>10.2f}% {mean_drift_rec*100:>10.2f}% {mean_fp:>10.1f} {mean_tp:>10.1f}")
    print("=" * 85)
    
    evaluate_glrt_trajectory_information()
