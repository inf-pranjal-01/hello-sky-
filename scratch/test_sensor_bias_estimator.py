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

def precompute_clean_cluster_diurnal_offsets():
    """
    Precompute normal station-specific diurnal offsets relative to cluster median.
    B_normal_station(hour) = E[x_{s,p}(hour) - median_{j in Peers}(x_{j,p}(hour))]
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

DIURNAL_OFFSETS = precompute_clean_cluster_diurnal_offsets()

def run_causal_sensor_bias_estimator(df_all_dict, alpha=0.15, beta=0.05, change_thresh_sigma=2.5, min_streak=4):
    """
    Causal Sensor-Bias State Estimator.
    Estimates changing sensor bias F_sensor(t) via 2-state dynamic tracking:
      B_hat(t) = B_hat(t-1) + v_hat(t-1)
      v_hat(t) = v_hat(t-1) + beta * (r(t) - B_hat(t))
      B_hat(t) = B_hat(t) + alpha * (r(t) - B_hat(t))
    """
    station_drift_flags = {}
    
    for sid, df in df_all_dict.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        hours = pd.to_datetime(df['timestamp']).dt.hour
        n_rows = len(df)
        station_flag = np.zeros(n_rows, dtype=bool)
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            prefix = "temp" if param == "temperature_c" else ("pressure" if param == "pressure_hpa" else "humidity")
            m_off, s_off = DIURNAL_OFFSETS[(sid, prefix)]
            
            # 1. Causal Peer Consensus
            peer_vals = []
            for pid in peer_ids:
                if pid in df_all_dict:
                    peer_vals.append(df_all_dict[pid][param])
            if peer_vals:
                peer_df = pd.concat(peer_vals, axis=1)
                peer_median = peer_df.median(axis=1)
                peer_spread = peer_df.std(axis=1).fillna(1.0)
            else:
                peer_median = df[param]
                peer_spread = pd.Series(1.0, index=df.index)
                
            # 2. Expected target reading given peer consensus and clean diurnal offset
            expected_x = peer_median + hours.map(m_off).fillna(0.0)
            residual = (df[param] - expected_x).to_numpy(dtype=float)
            sigma_bound = (hours.map(s_off).fillna(1.0) + 0.3 * peer_spread).to_numpy(dtype=float)
            
            # 3. Dynamic Sequential Bias Tracking
            B_hat = np.zeros(n_rows, dtype=float)
            v_hat = np.zeros(n_rows, dtype=float)
            curr_B = 0.0
            curr_v = 0.0
            bias_streak = 0
            curr_sign = 0
            p_flags = np.zeros(n_rows, dtype=bool)
            
            for t in range(n_rows):
                r_t = residual[t]
                if np.isnan(r_t):
                    continue
                    
                # State prediction
                pred_B = curr_B + curr_v
                innov = r_t - pred_B
                
                # Innovation gating: robust against massive isolated spikes
                clipped_innov = np.clip(innov, -3.0 * sigma_bound[t], 3.0 * sigma_bound[t])
                
                # State update
                curr_B = pred_B + alpha * clipped_innov
                curr_v = curr_v + beta * clipped_innov
                
                B_hat[t] = curr_B
                v_hat[t] = curr_v
                
                # Check sign consistency over past window
                sign = 1 if curr_v > 0.02 else (-1 if curr_v < -0.02 else 0)
                if sign != 0 and sign == curr_sign:
                    bias_streak += 1
                elif sign != 0:
                    curr_sign = sign
                    bias_streak = 1
                else:
                    bias_streak = max(0, bias_streak - 1)
                    
                # 4. Bias magnitude check: has bias moved substantially beyond peer noise?
                bias_change_6h = abs(curr_B - B_hat[max(0, t - 6)])
                bias_rel_sigma = bias_change_6h / max(0.5, sigma_bound[t])
                
                # Confirmation condition: persistent velocity & significant cumulative bias change
                if bias_streak >= min_streak and bias_rel_sigma >= change_thresh_sigma and abs(curr_B) >= 1.5 * sigma_bound[t]:
                    p_flags[t] = True
                    
            station_flag |= p_flags
            
        station_drift_flags[sid] = station_flag
        
    return station_drift_flags

def evaluate_estimator(seed=42, alpha=0.15, beta=0.05, change_thresh_sigma=2.5, min_streak=4):
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    # Run Causal Bias Estimator
    drift_flags = run_causal_sensor_bias_estimator(
        data, alpha=alpha, beta=beta, change_thresh_sigma=change_thresh_sigma, min_streak=min_streak
    )
    
    frames = []
    for sid, df_raw in data.items():
        d = df_raw.copy()
        d["station_id"] = sid
        d["__bias_drift"] = drift_flags[sid]
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
    
    drift_lookup = df_full.set_index(["station_id", "timestamp"])["__bias_drift"]
    drift_arr = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(drift_lookup).fillna(False).to_numpy(dtype=bool)
    
    drift_boost = drift_arr & (row_fault_type == "none")
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
        | (drift_arr & (corrob_peers_count < 2))
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

def analyze_drift_snr():
    """
    Computes empirical Signal-to-Noise Ratio (SNR) for injected drift across the network.
    Signal: injected drift offset magnitude across time steps t in [1, 20] hours.
    Noise: inter-station residual uncertainty standard deviation sigma_peer.
    """
    print("\n" + "="*70)
    print("EMPIRICAL SIGNAL-TO-NOISE RATIO (SNR) OF EARLY SENSOR DRIFT")
    print("="*70)
    
    # Calculate empirical peer spread for temperature across all clusters
    temp_spreads = []
    for (sid, prefix), (m_off, s_off) in DIURNAL_OFFSETS.items():
        if prefix == "temp":
            temp_spreads.append(s_off.mean())
    mean_peer_sigma = np.mean(temp_spreads)
    
    # Typical drift slope in anomaly_injector is ~0.20 to 0.35 deg C / hour
    drift_slope = 0.25 # deg C / hour
    
    print(f"Network Inter-Station Residual Uncertainty (Noise Floor): sigma = {mean_peer_sigma:.2f} deg C")
    print(f"Drift Slope: {drift_slope:.2f} deg C / hour")
    print("-" * 70)
    print(f"{'Hour (t)':<10} {'Cumulative Bias (|B|)':<25} {'SNR (|B| / sigma)':<20} {'Point Identifiability'}")
    print("-" * 70)
    for t in [1, 2, 4, 6, 8, 12, 16, 20]:
        cum_bias = t * drift_slope
        snr = cum_bias / mean_peer_sigma
        identifiable = "UNIDENTIFIABLE (SNR < 1.0)" if snr < 1.0 else ("MARGINAL (1 <= SNR < 2.0)" if snr < 2.0 else "IDENTIFIABLE (SNR >= 2.0)")
        print(f"{t:<10} {cum_bias:<25.2f} {snr:<20.2f} {identifiable}")
    print("=" * 70)

if __name__ == '__main__':
    print("Running 3-Seed Dev Evaluation for Causal Sensor-Bias State Estimator...")
    for s in [42, 101, 202]:
        res = evaluate_estimator(seed=s)
        print(f"Seed {s:<5} | Prec: {res['precision']*100:.2f}%, Rec: {res['recall']*100:.2f}%, DriftRec: {res['drift_recall']*100:.2f}%, TP: {res['tp']}, FP: {res['fp']}, FN: {res['fn']}, F1*: {res['f1_star']:.4f}")
        
    analyze_drift_snr()
