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

def run_causal_slope_regime_drift(df_all_dict):
    """
    Causal Change-Point & Adaptive Slope-Regime Differential (CCASRD).
    Tests for sustained non-zero slope in the differential between target and peer consensus.
    """
    station_drift_flags = {}
    
    for sid, df in df_all_dict.items():
        cid = STATION_TO_CLUSTER.get(sid)
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        n_rows = len(df)
        station_flag = np.zeros(n_rows, dtype=bool)
        
        for param in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
            peer_vals = pd.DataFrame({p: df_all_dict[p][param] for p in peer_ids if p in df_all_dict})
            peer_med = peer_vals.median(axis=1)
            
            # Differential signal: target minus peer median
            diff = df[param] - peer_med
            # Hourly rate of change of the differential
            diff_roc = diff.diff()
            
            # Scale floor for differential ROC
            scale = 0.4 if param == "temperature_c" else (0.2 if param == "pressure_hpa" else 1.5)
            z_diff_roc = (diff_roc / scale).clip(-3.0, 3.0)
            
            # GLR / Page-Hinkley slope accumulator
            splus, sminus = 0.0, 0.0
            in_regime = False
            regime_dir = 0
            quiet_count = 0
            
            p_flags = np.zeros(n_rows, dtype=bool)
            allowance = 0.25
            h_threshold = 4.5
            
            for i in range(n_rows):
                v = z_diff_roc.iloc[i]
                if np.isnan(v):
                    splus, sminus = 0.0, 0.0
                    in_regime = False
                    continue
                
                # CUSUM on slope differential
                splus = max(0.0, splus + v - allowance)
                sminus = max(0.0, sminus - v - allowance)
                
                if not in_regime:
                    # Require strong slope accumulation AND cumulative deviation >= 2.0 scale
                    if splus >= h_threshold and abs(diff.iloc[i] - diff.iloc[max(0, i-6)]) >= (scale * 2.5):
                        in_regime = True
                        regime_dir = 1
                        quiet_count = 0
                    elif sminus >= h_threshold and abs(diff.iloc[i] - diff.iloc[max(0, i-6)]) >= (scale * 2.5):
                        in_regime = True
                        regime_dir = -1
                        quiet_count = 0
                else:
                    # Check if slope returned to flat or reversed
                    if (regime_dir == 1 and v < -0.5) or (regime_dir == -1 and v > 0.5):
                        quiet_count += 2
                    elif abs(v) < 0.2:
                        quiet_count += 1
                    else:
                        quiet_count = max(0, quiet_count - 1)
                        
                    if quiet_count >= 4:
                        in_regime = False
                        splus, sminus = 0.0, 0.0
                        
                if in_regime:
                    p_flags[i] = True
                    
            station_flag |= p_flags
            
        station_drift_flags[sid] = station_flag
    return station_drift_flags

def evaluate_ccasrd(seed=42):
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    # Run CCASRD
    ccasrd_flags = run_causal_slope_regime_drift(data)
    
    frames = []
    for sid, df_raw in data.items():
        d = df_raw.copy()
        d["station_id"] = sid
        d["__ccasrd_drift"] = ccasrd_flags[sid]
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
    
    ccasrd_lookup = df_full.set_index(["station_id", "timestamp"])["__ccasrd_drift"]
    ccasrd_arr = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(ccasrd_lookup).fillna(False).to_numpy(dtype=bool)
    
    drift_boost = ccasrd_arr & (row_fault_type == "none")
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
        | (ccasrd_arr & (corrob_peers_count < 2))
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
    print("Evaluating CCASRD on 3 DEV SEEDS (42, 101, 202)...")
    for s in [42, 101, 202]:
        res = evaluate_ccasrd(s)
        print(f"Seed {s:<5} | Prec: {res['precision']*100:.2f}%, Rec: {res['recall']*100:.2f}%, DriftRec: {res['drift_recall']*100:.2f}%, TP: {res['tp']}, FP: {res['fp']}, FN: {res['fn']}, F1*: {res['f1_star']:.4f}")
