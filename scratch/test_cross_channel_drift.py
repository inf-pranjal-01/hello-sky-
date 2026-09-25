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

def precompute_clean_diurnal_stats():
    stats = {}
    for sid in STATION_TO_CLUSTER:
        df_s = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        hours = df_s['timestamp'].dt.hour
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            prefix = "temp" if param == "temperature_c" else ("pressure" if param == "pressure_hpa" else "humidity")
            m = df_s[param].groupby(hours).mean()
            s = df_s[param].groupby(hours).std().clip(lower=0.3)
            stats[(sid, prefix)] = (m, s)
    return stats

DIURNAL_STATS = precompute_clean_diurnal_stats()

def run_cross_channel_consistency_drift(df_all_dict):
    """
    Causal Cross-Channel Physical Consistency Detector (CC-PCD).
    Detects when a single sensor channel persistently diverges while
    other local channels and peer stations remain meteorologically coherent.
    """
    station_drift_flags = {}
    
    for sid, df in df_all_dict.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        hours = pd.to_datetime(df['timestamp']).dt.hour
        n_rows = len(df)
        
        # Compute normalized diurnal residuals for all 3 channels
        z_dict = {}
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            prefix = "temp" if param == "temperature_c" else ("pressure" if param == "pressure_hpa" else "humidity")
            m, s = DIURNAL_STATS[(sid, prefix)]
            z_dict[prefix] = (df[param] - hours.map(m)) / hours.map(s).fillna(1.0).clip(lower=0.3)
            
        station_flag = np.zeros(n_rows, dtype=bool)
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            prefix = "temp" if param == "temperature_c" else ("pressure" if param == "pressure_hpa" else "humidity")
            other_prefixes = [p for p in ["temp", "pressure", "humidity"] if p != prefix]
            
            target_z = z_dict[prefix]
            other_z_mean = (z_dict[other_prefixes[0]].abs() + z_dict[other_prefixes[1]].abs()) / 2.0
            
            # Cross-channel isolation: target channel is abnormal while others are normal
            cross_channel_isolation = target_z.abs() / (other_z_mean + 0.5)
            
            # Peer consensus for this channel
            peer_z_list = []
            for pid in peer_ids:
                if pid in df_all_dict:
                    pm, ps = DIURNAL_STATS[(pid, prefix)]
                    pz = (df_all_dict[pid][param] - hours.map(pm)) / hours.map(ps).fillna(1.0).clip(lower=0.3)
                    peer_z_list.append(pz)
                    
            if peer_z_list:
                peer_z_df = pd.concat(peer_z_list, axis=1)
                peer_med = peer_z_df.median(axis=1)
                peer_spr = peer_z_df.std(axis=1).fillna(1.0)
            else:
                peer_med = pd.Series(0.0, index=df.index)
                peer_spr = pd.Series(1.0, index=df.index)
                
            # Peer divergence
            peer_div = (target_z - peer_med).abs()
            
            # Combined Physical Sensor-Fault Score
            # High when: (1) target channel deviates, (2) other local channels normal, (3) peers agree with each other
            p_flags = np.zeros(n_rows, dtype=bool)
            streak = 0
            
            for i in range(n_rows):
                tz = target_z.iloc[i]
                iso = cross_channel_isolation.iloc[i]
                pdiv = peer_div.iloc[i]
                pspr = peer_spr.iloc[i]
                
                if np.isnan(tz):
                    streak = 0
                    continue
                
                # Check invariant:
                # - Target channel is strongly deviating (|target_z| >= 2.0)
                # - Isolated to this sensor channel (iso >= 1.8)
                # - Isolated to this station (pdiv >= 1.8 and pspr <= 2.0)
                is_isolated_channel_fault = (abs(tz) >= 2.0) and (iso >= 1.8) and (pdiv >= 1.8) and (pspr <= 2.0)
                
                if is_isolated_channel_fault:
                    streak += 1
                else:
                    streak = max(0, streak - 1)
                    
                # Alarm confirmed after 3 consecutive hours of physical inconsistency
                if streak >= 3:
                    p_flags[i] = True
                    
            station_flag |= p_flags
            
        station_drift_flags[sid] = station_flag
        
    return station_drift_flags

def evaluate_cc_pcd(seed=42):
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    # Run CC-PCD
    cc_flags = run_cross_channel_consistency_drift(data)
    
    frames = []
    for sid, df_raw in data.items():
        d = df_raw.copy()
        d["station_id"] = sid
        d["__cc_drift"] = cc_flags[sid]
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
    
    cc_lookup = df_full.set_index(["station_id", "timestamp"])["__cc_drift"]
    cc_arr = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(cc_lookup).fillna(False).to_numpy(dtype=bool)
    
    drift_boost = cc_arr & (row_fault_type == "none")
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
        | (cc_arr & (corrob_peers_count < 2))
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

if __name__ == '__main__':
    print("Evaluating CC-PCD on 3 DEV SEEDS (42, 101, 202)...")
    for s in [42, 101, 202]:
        res = evaluate_cc_pcd(s)
        print(f"Seed {s:<5} | Prec: {res['precision']*100:.2f}%, Rec: {res['recall']*100:.2f}%, DriftRec: {res['drift_recall']*100:.2f}%, TP: {res['tp']}, FP: {res['fp']}, FN: {res['fn']}, F1*: {res['f1_star']:.4f}")
