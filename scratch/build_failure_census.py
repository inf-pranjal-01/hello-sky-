import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    evaluate_all, ARTIFACTS_PATH,
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

def compute_census_for_seed(seed=20260924):
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    # Load clean data for true ground truth comparisons
    clean_dfs = {}
    for sid in STATION_TO_CLUSTER:
        c_df = pd.read_csv(f"data/{sid}.csv", parse_dates=["timestamp"])
        c_df["timestamp"] = pd.to_datetime(c_df["timestamp"]).dt.tz_localize(None)
        clean_dfs[sid] = c_df.set_index("timestamp")

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
    
    # Save census cases
    census_records = []
    cases_dir = Path("forensic_cases")
    cases_dir.mkdir(exist_ok=True)
    
    gt = featured["is_anomaly"].to_numpy()
    pred = featured["is_anomaly_pred"].to_numpy()
    fn_indices = np.where(gt & ~pred)[0]
    
    for idx in fn_indices:
        row = featured.iloc[idx]
        sid = row["station_id"]
        cid = STATION_TO_CLUSTER[sid]
        ts = row["timestamp"]
        ft = row["fault_type"]
        
        # Peer information
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_vals_t = []
        peer_vals_p = []
        peer_vals_rh = []
        for pid in peer_ids:
            if pid in clean_dfs and ts in clean_dfs[pid].index:
                peer_vals_t.append(clean_dfs[pid].loc[ts, "temperature_c"])
                peer_vals_p.append(clean_dfs[pid].loc[ts, "pressure_hpa"])
                peer_vals_rh.append(clean_dfs[pid].loc[ts, "humidity_pct"])
                
        # Target clean vs injected
        target_clean_t = clean_dfs[sid].loc[ts, "temperature_c"] if ts in clean_dfs[sid].index else np.nan
        target_clean_p = clean_dfs[sid].loc[ts, "pressure_hpa"] if ts in clean_dfs[sid].index else np.nan
        target_clean_rh = clean_dfs[sid].loc[ts, "humidity_pct"] if ts in clean_dfs[sid].index else np.nan
        
        inj_t = row["temperature_c"]
        inj_p = row["pressure_hpa"]
        inj_rh = row["humidity_pct"]
        
        delta_t = inj_t - target_clean_t
        delta_p = inj_p - target_clean_p
        delta_rh = inj_rh - target_clean_rh
        
        # Determine active channel
        if abs(delta_t) >= max(abs(delta_p), abs(delta_rh), 1e-3):
            ch = "temperature_c"
            delta = delta_t
            p_vals = peer_vals_t
            inj_val = inj_t
            clean_val = target_clean_t
        elif abs(delta_p) >= max(abs(delta_t), abs(delta_rh), 1e-3):
            ch = "pressure_hpa"
            delta = delta_p
            p_vals = peer_vals_p
            inj_val = inj_p
            clean_val = target_clean_p
        else:
            ch = "humidity_pct"
            delta = delta_rh
            p_vals = peer_vals_rh
            inj_val = inj_rh
            clean_val = target_clean_rh
            
        p_med = np.median(p_vals) if p_vals else np.nan
        p_mad = np.median(np.abs(np.array(p_vals) - p_med)) if p_vals else np.nan
        raw_res = inj_val - p_med
        
        # Category classification
        if abs(delta) < 1.5 * max(0.4, p_mad):
            category = "A. Fault signal genuinely below normal variability"
        elif ft == "frozen_value":
            category = "E. Frozen/noisy-frozen statistical signature exists"
        elif ft == "multivariate_inconsistency":
            category = "F. Cross-channel inconsistency exists"
        elif abs(raw_res) < 1.5 * p_mad:
            category = "B. Raw peer consensus hides fault"
        else:
            category = "C. Station-specific normal behavior would explain it"
            
        census_records.append({
            "seed": seed,
            "station": sid,
            "cluster": cid,
            "timestamp": str(ts),
            "fault_type": ft,
            "channel": ch,
            "clean_val": clean_val,
            "injected_val": inj_val,
            "fault_delta": delta,
            "peer_median": p_med,
            "peer_mad": p_mad,
            "raw_peer_residual": raw_res,
            "category": category,
            "model_pct": row["model_pct"],
            "rule_conf": row["rule_conf"]
        })
        
    df_census = pd.DataFrame(census_records)
    df_census.to_csv(cases_dir / f"failure_census_seed_{seed}.csv", index=False)
    
    print(f"=== FAILURE CENSUS SUMMARY FOR SEED {seed} ===")
    print(f"Total False Negatives (FN): {len(df_census)}")
    print("\nFN Breakdown by Root-Cause Category:")
    print(df_census["category"].value_counts())
    print("\nFN Breakdown by Fault Type:")
    print(df_census["fault_type"].value_counts())
    
    return df_census

if __name__ == "__main__":
    df_census = compute_census_for_seed(seed=20260924)
