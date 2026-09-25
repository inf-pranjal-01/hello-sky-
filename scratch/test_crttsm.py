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
    get_expected_roc, add_frozen_channel_labels_from_reference
)
from model.fault_helper import predict_faults, score_frozen_channels

STATION_TO_CLUSTER = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    for sid in [center] + neighbors:
        STATION_TO_CLUSTER[sid] = cid

def precompute_clean_roc_profiles():
    profiles = {}
    for sid in STATION_TO_CLUSTER:
        df_s = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        hours = df_s['timestamp'].dt.hour
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            prefix = "temp" if param == "temperature_c" else ("pressure" if param == "pressure_hpa" else "humidity")
            roc = df_s[param].diff()
            mean_roc = roc.groupby(hours).mean()
            std_roc = roc.groupby(hours).std().clip(lower=0.25)
            profiles[(sid, prefix)] = (mean_roc, std_roc)
    return profiles

ROC_PROFILES = precompute_clean_roc_profiles()

def run_causal_residual_trend_drift(df_all_dict):
    """
    Causal Residual-Trend Trajectory State Machine (CRTTSM).
    Tracks whether target residual trajectory persistently diverges from expected diurnal trajectory
    while cluster peers remain meteorologically coherent.
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
            mean_roc, std_roc = ROC_PROFILES[(sid, prefix)]
            
            # Target ROC residual
            target_roc = df[param].diff()
            exp_roc = hours.map(mean_roc)
            scale_roc = hours.map(std_roc).fillna(1.0).clip(lower=0.3)
            target_z = (target_roc - exp_roc) / scale_roc
            
            # Peer consensus ROC residual
            peer_z_list = []
            for pid in peer_ids:
                if pid in df_all_dict:
                    p_mean_roc, p_std_roc = ROC_PROFILES[(pid, prefix)]
                    p_roc = df_all_dict[pid][param].diff()
                    p_exp = hours.map(p_mean_roc)
                    p_scale = hours.map(p_std_roc).fillna(1.0).clip(lower=0.3)
                    peer_z_list.append((p_roc - p_exp) / p_scale)
            
            if peer_z_list:
                peer_z_df = pd.concat(peer_z_list, axis=1)
                peer_median_z = peer_z_df.median(axis=1)
                peer_spread = peer_z_df.std(axis=1).fillna(1.0)
            else:
                peer_median_z = pd.Series(0.0, index=df.index)
                peer_spread = pd.Series(1.0, index=df.index)
                
            # Trajectory differential: target residual minus peer environmental residual
            delta_z = target_z - peer_median_z
            
            splus, sminus = 0.0, 0.0
            pos_streak, neg_streak = 0, 0
            state = 0  # 0=NORMAL, 1=EMERGING, 2=CONFIRMED
            drift_dir = 0
            clean_streak = 0
            
            p_flags = np.zeros(n_rows, dtype=bool)
            
            for i in range(n_rows):
                dz = delta_z.iloc[i]
                p_med = peer_median_z.iloc[i]
                p_spr = peer_spread.iloc[i]
                
                if np.isnan(dz):
                    splus, sminus = 0.0, 0.0
                    pos_streak, neg_streak = 0, 0
                    state = 0
                    continue
                
                # Check peer meteorological coherence (peers not in chaotic front)
                peer_coherent = (abs(p_med) < 2.5) and (p_spr < 2.5)
                
                if peer_coherent:
                    if dz >= 0.6:
                        pos_streak += 1
                        neg_streak = 0
                        splus += (dz - 0.3)
                        sminus = max(0.0, sminus - 1.0)
                    elif dz <= -0.6:
                        neg_streak += 1
                        pos_streak = 0
                        sminus += (-dz - 0.3)
                        splus = max(0.0, splus - 1.0)
                    else:
                        pos_streak = max(0, pos_streak - 1)
                        neg_streak = max(0, neg_streak - 1)
                        splus = max(0.0, splus - 0.5)
                        sminus = max(0.0, sminus - 0.5)
                else:
                    # Weather front passing through entire cluster: drain trajectory
                    splus = max(0.0, splus - 1.5)
                    sminus = max(0.0, sminus - 1.5)
                    pos_streak = 0
                    neg_streak = 0
                
                # State Machine transitions
                if state == 0:  # NORMAL
                    if (pos_streak >= 3 and splus >= 3.0):
                        state = 2  # CONFIRMED
                        drift_dir = 1
                        clean_streak = 0
                    elif (neg_streak >= 3 and sminus >= 3.0):
                        state = 2  # CONFIRMED
                        drift_dir = -1
                        clean_streak = 0
                    elif (pos_streak >= 2 or neg_streak >= 2):
                        state = 1  # EMERGING
                elif state == 1:  # EMERGING
                    if (pos_streak >= 3 and splus >= 3.0):
                        state = 2  # CONFIRMED
                        drift_dir = 1
                        clean_streak = 0
                    elif (neg_streak >= 3 and sminus >= 3.0):
                        state = 2  # CONFIRMED
                        drift_dir = -1
                        clean_streak = 0
                    elif (pos_streak == 0 and neg_streak == 0 and splus < 1.0 and sminus < 1.0):
                        state = 0  # return to NORMAL
                elif state == 2:  # CONFIRMED
                    if (drift_dir == 1 and dz >= 0.2) or (drift_dir == -1 and dz <= -0.2):
                        clean_streak = 0
                    else:
                        clean_streak += 1
                        if clean_streak >= 2 or (drift_dir == 1 and dz < -0.5) or (drift_dir == -1 and dz > 0.5):
                            state = 0  # RECOVERY back to NORMAL
                            splus, sminus = 0.0, 0.0
                            
                if state == 2:
                    p_flags[i] = True
                    
            station_flag |= p_flags
            
        station_drift_flags[sid] = station_flag
        
    return station_drift_flags

def evaluate_with_crttsm(data_dict, artifact):
    """Integrates CRTTSM into full canonical evaluation pipeline."""
    crttsm_flags = run_causal_residual_trend_drift(data_dict)
    
    frames = []
    for sid, df_raw in data_dict.items():
        d = df_raw.copy()
        d["station_id"] = sid
        d["__crttsm_drift"] = crttsm_flags[sid]
        frames.append(d)
        
    df_full = pd.concat(frames, ignore_index=True)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"]).dt.tz_localize(None)
    df_full = add_frozen_channel_labels_from_reference(df_full)
    
    # Forward fill completeness
    raw_nans = df_full[["temperature_c", "pressure_hpa", "humidity_pct"]].isna().any(axis=1).to_numpy(dtype=bool)
    df_full["__raw_nan_flag"] = raw_nans
    df_full[["temperature_c", "pressure_hpa", "humidity_pct"]] = df_full[["temperature_c", "pressure_hpa", "humidity_pct"]].ffill().bfill()
    
    label_cols = ["station_id", "timestamp", "is_anomaly", "fault_type"]
    labels = df_full[label_cols].copy()
    labels["is_anomaly"] = labels["is_anomaly"].fillna(False).astype(bool)
    labels["fault_type"] = labels["fault_type"].fillna("none")
    
    df_in = df_full.drop(columns=["is_anomaly", "fault_type"], errors="ignore")
    
    # Featurize
    featured, _ = _featurize(df_in)
    featured["timestamp"] = pd.to_datetime(featured["timestamp"]).dt.tz_localize(None)
    
    # Rule engine
    featured, row_hard, row_rule_conf, row_fault_type, per_sensor_log, recovery_log = run_rule_engine_and_health(featured, artifact)
    
    # Inject CRTTSM drift evidence into rule confidence
    crttsm_lookup = df_full.set_index(["station_id", "timestamp"])["__crttsm_drift"]
    crttsm_arr = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(crttsm_lookup).fillna(False).to_numpy(dtype=bool)
    
    drift_boost = crttsm_arr & (row_fault_type == "none")
    row_fault_type[drift_boost] = "drift"
    row_rule_conf[drift_boost] = 85.0
    
    # Spatial corroboration
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
        | (crttsm_arr & (corrob_peers_count < 2))
    )
    
    # Frozen gate
    frozen_only = row_rule_conf == RULE_BASE_CONFIDENCE['frozen_value']
    predicted = predicted & ~(frozen_only & (model_pct < FROZEN_MIN_MODEL_CORROBORATION))
    
    # Helpers
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
        "ep_catch_rate": ep.episode_detection_rate,
        "ep_detected": ep.detected_episodes,
        "ep_total": ep.total_episodes,
    }

if __name__ == '__main__':
    artifact = joblib.load(ARTIFACTS_PATH)
    print("Evaluating 3 DEV SEEDS (42, 101, 202)...")
    for s in [42, 101, 202]:
        data = generate_network_benchmark(regime='benchmark_b', seed=s, save_to_disk=False)
        res = evaluate_with_crttsm(data, artifact)
        print(f"Seed {s:<5} | Prec: {res['precision']*100:.2f}%, Rec: {res['recall']*100:.2f}%, DriftRec: {res['drift_recall']*100:.2f}%, TP: {res['tp']}, FP: {res['fp']}, FN: {res['fn']}, F1*: {res['f1_star']:.4f}")
