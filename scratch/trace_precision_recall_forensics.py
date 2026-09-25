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

def trace_detailed_eval(seed=20260924):
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    # Run canonical offline evaluation pipeline
    res = evaluate_all(data, artifact, silent=True)
    m = res["__overall__"]
    ep = res["__episodic__"]
    
    # We also inspect per-row predictions
    frames = []
    for sid, df_raw in data.items():
        d = df_raw.copy()
        d["station_id"] = sid
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
    featured["is_anomaly_pred"] = predicted
    featured["fault_type_pred"] = row_fault_type
    featured["model_pct"] = model_pct
    featured["rule_conf"] = row_rule_conf
    featured["overall_conf"] = overall_confidence
    
    # Merge clean raw data for peer comparison
    clean_dfs = {}
    for sid in STATION_TO_CLUSTER:
        c_df = pd.read_csv(f"data/{sid}.csv", parse_dates=["timestamp"])
        c_df["timestamp"] = pd.to_datetime(c_df["timestamp"]).dt.tz_localize(None)
        clean_dfs[sid] = c_df
        
    return featured, data, clean_dfs

def analyze_breakdown(featured, data_dict, clean_dfs):
    print("="*80)
    print("1. MACRO PRECISION & RECALL BREAKDOWN")
    print("="*80)
    
    gt = featured["is_anomaly"].to_numpy()
    pred = featured["is_anomaly_pred"].to_numpy()
    
    tp = (gt & pred).sum()
    fp = (~gt & pred).sum()
    fn = (gt & ~pred).sum()
    tn = (~gt & ~pred).sum()
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Total Rows: {len(featured)}")
    print(f"True Positives  (TP): {tp} (Correct Alerts)")
    print(f"False Positives (FP): {fp} (False Alarms on Clean Data)")
    print(f"False Negatives (FN): {fn} (Missed Anomalies)")
    print(f"True Negatives  (TN): {tn} (Clean Timesteps Correctly Ignored)")
    print(f"Point Precision: {prec*100:.2f}%")
    print(f"Point Recall:    {rec*100:.2f}%\n")
    
    print("="*80)
    print("2. FALSE NEGATIVE (FN) ANALYSIS BY FAULT TYPE")
    print("="*80)
    for ft, group in featured[gt].groupby("fault_type"):
        g_tp = (group["is_anomaly_pred"]).sum()
        g_fn = (~group["is_anomaly_pred"]).sum()
        g_rec = g_tp / len(group) if len(group) > 0 else 0
        print(f"Fault: {ft:<28} | Total: {len(group):<5} | Caught (TP): {g_tp:<5} | Missed (FN): {g_fn:<5} | Recall: {g_rec*100:.1f}%")
        
    print("\n" + "="*80)
    print("3. EXACT ROW-LEVEL TRACE: WHY EARLY DRIFT IS MISSED (FALSE NEGATIVES)")
    print("="*80)
    drift_fn = featured[(featured["fault_type"] == "drift") & (~featured["is_anomaly_pred"])]
    # Pick a representative drift episode
    first_drift_station = drift_fn["station_id"].iloc[0]
    drift_ep = featured[(featured["station_id"] == first_drift_station) & (featured["fault_type"] == "drift")].sort_values("timestamp")
    
    cid = STATION_TO_CLUSTER[first_drift_station]
    peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != first_drift_station]
    
    print(f"Tracing Drift Episode on Station: {first_drift_station} (Cluster: {cid})")
    print(f"Peers in Cluster: {peer_ids}")
    print("-" * 105)
    print(f"{'Timestamp':<20} {'Hour':<5} {'Target T (C)':<14} {'Clean T (C)':<14} {'Injected d_T':<14} {'Peer Median T':<14} {'Peer Std':<10} {'Pred':<6} {'Reason'}")
    print("-" * 105)
    
    for idx, row in drift_ep.head(12).iterrows():
        ts = row["timestamp"]
        t_val = row["temperature_c"]
        # get clean T
        clean_row = clean_dfs[first_drift_station][clean_dfs[first_drift_station]["timestamp"] == ts]
        clean_t = clean_row["temperature_c"].values[0] if len(clean_row) > 0 else np.nan
        inj_delta = t_val - clean_t
        
        # Peer values at this timestamp
        peer_t_vals = []
        for pid in peer_ids:
            p_match = data_dict[pid][data_dict[pid]["timestamp"] == ts]
            if len(p_match) > 0:
                peer_t_vals.append(p_match["temperature_c"].values[0])
        p_med = np.median(peer_t_vals) if peer_t_vals else np.nan
        p_std = np.std(peer_t_vals) if peer_t_vals else np.nan
        
        pred_val = "ALERT" if row["is_anomaly_pred"] else "MISS"
        reason = f"d_T ({inj_delta:+.2f} C) < Peer Spread (+/-{p_std:.2f} C)" if abs(inj_delta) < 1.5 * p_std else "Large deviation caught"
        print(f"{str(ts):<20} {ts.hour:<5} {t_val:<14.2f} {clean_t:<14.2f} {inj_delta:<14.2f} {p_med:<14.2f} {p_std:<10.2f} {pred_val:<6} {reason}")

    print("\n" + "="*80)
    print("4. EXACT ROW-LEVEL TRACE: WHERE FALSE POSITIVES (FP) COME FROM")
    print("="*80)
    fps = featured[(~featured["is_anomaly"]) & (featured["is_anomaly_pred"])]
    print(f"Total FP Timesteps across clean data: {len(fps)}")
    print(f"FP Distribution by Station:")
    print(fps["station_id"].value_counts().head(8))
    print("\nTracing 5 Concrete False Alarm (FP) Timesteps on Clean Stations:")
    print("-" * 105)
    print(f"{'Station':<14} {'Timestamp':<20} {'T (°C)':<8} {'RH (%)':<8} {'P (hPa)':<10} {'Model%':<8} {'RuleConf':<10} {'Pred Fault':<15} {'Why Triggered'}")
    print("-" * 105)
    for idx, row in fps.head(6).iterrows():
        sid = row["station_id"]
        ts = row["timestamp"]
        t = row["temperature_c"]
        rh = row["humidity_pct"]
        p = row["pressure_hpa"]
        mp = row["model_pct"]
        rc = row["rule_conf"]
        pft = row["fault_type_pred"]
        
        # Why triggered
        if mp > MODEL_ALONE_OVERRIDE_THRESHOLD:
            why = f"Model override (Model Score {mp:.1f}% > 90%)"
        elif rc > 0:
            why = f"Rule trigger '{pft}' (Rule Conf {rc:.1f}%)"
        else:
            why = f"Helper alert or fusion confidence"
        print(f"{sid:<14} {str(ts):<20} {t:<8.1f} {rh:<8.1f} {p:<10.1f} {mp:<8.1f} {rc:<10.1f} {pft:<15} {why}")
    print("=" * 105)

if __name__ == "__main__":
    featured, data_dict, clean_dfs = trace_detailed_eval(seed=20260924)
    analyze_breakdown(featured, data_dict, clean_dfs)
