import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    ARTIFACTS_PATH,
    vectorized_model_scores, run_rule_engine_and_health,
    apply_spatial_corroboration, _score_and_report, _featurize,
    MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD,
    MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS,
    add_frozen_channel_labels_from_reference
)
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel,
    ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel,
    ConditionalMultivariateDriftDetector,
    STATION_TO_CLUSTER
)

print("Fitting Causal Models on Clean 60% Calibration Slice...", flush=True)
normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
normal_model.fit()

uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
uncertainty_model.fit()

joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
joint_model.fit()

artifact = joblib.load(ARTIFACTS_PATH)
detector = ConditionalMultivariateDriftDetector(glrt_thresh=26.0, cusum_thresh=20.0, allowance=0.45, persist_req=2)

print("Generating benchmark for seed 42...", flush=True)
data = generate_network_benchmark(regime='operational_v1', seed=42, save_to_disk=False)

frames = []
for sid, df_raw in data.items():
    d = df_raw.copy()
    d['station_id'] = sid
    frames.append(d)
df_full = pd.concat(frames, ignore_index=True)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp']).dt.tz_localize(None)
df_full = add_frozen_channel_labels_from_reference(df_full)

raw_nans = df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']].isna().any(axis=1).to_numpy(dtype=bool)
df_full['__raw_nan_flag'] = raw_nans
df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']] = df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']].ffill().bfill()

label_cols = ['station_id', 'timestamp', 'is_anomaly', 'fault_type']
labels = df_full[label_cols].copy()
labels['is_anomaly'] = labels['is_anomaly'].fillna(False).astype(bool)
labels['fault_type'] = labels['fault_type'].fillna('none')

df_in = df_full.drop(columns=['is_anomaly', 'fault_type', 'injected_delta', 'injected_start', 'injected_end'], errors='ignore')

print("Featurizing...", flush=True)
featured, _ = _featurize(df_in)
featured['timestamp'] = pd.to_datetime(featured['timestamp']).dt.tz_localize(None)

print("Running rule engine...", flush=True)
featured_base, row_hard, row_rule_conf, row_fault_type, _, _ = run_rule_engine_and_health(featured.copy(), artifact)
row_rule_conf, row_fault_type, _ = apply_spatial_corroboration(
    featured_base, row_hard, row_rule_conf, row_fault_type, artifact, gate_mode='new'
)
model_pct = vectorized_model_scores(featured_base, artifact)
overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
is_regional = (row_fault_type == "REGIONAL_EVENT")

print("Streaming Drift GLRT...", flush=True)
det_flags_list = []
for sid in STATION_TO_CLUSTER:
    df = data[sid].sort_values('timestamp').reset_index(drop=True)
    df_clean = df.copy()
    df_clean[['temperature_c', 'pressure_hpa', 'humidity_pct']] = df_clean[['temperature_c', 'pressure_hpa', 'humidity_pct']].ffill().bfill()

    cid = STATION_TO_CLUSTER[sid]
    peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
    peer_dfs = {}
    for pid in peer_ids:
        p_df = data[pid].sort_values('timestamp').reset_index(drop=True)
        p_clean = p_df.copy()
        p_clean[['temperature_c', 'pressure_hpa', 'humidity_pct']] = p_clean[['temperature_c', 'pressure_hpa', 'humidity_pct']].ffill().bfill()
        peer_dfs[pid] = p_clean

    hr = uncertainty_model.get_hour_regime(pd.to_datetime(df_clean['timestamp']).dt.hour.values)

    Z_list = []
    for p in joint_model.channels:
        y_hat, _ = normal_model.predict_target_robust(sid, p, df_clean, peer_dfs)
        r = df_clean[p].values - y_hat
        sig = uncertainty_model.predict_sigma(sid, p, y_hat, df_clean['timestamp'].values)
        Z_list.append(np.nan_to_num(r / np.maximum(0.1, sig), nan=0.0))
    Z = np.column_stack(Z_list)
    n_rows = len(Z)

    u_T = np.zeros(n_rows)
    sig_cond_T = np.ones(n_rows)
    for i in range(n_rows):
        u, s = joint_model.get_conditional_innovation(sid, hr[i], Z[i], 0)
        u_T[i] = u
        sig_cond_T[i] = s

    u_T = np.nan_to_num(u_T, nan=0.0)
    sig_cond_T = np.nan_to_num(sig_cond_T, nan=1.0)
    res_det = detector.process_station_stream(u_T, sig_cond_T, df['timestamp'].values)
    df_stn = pd.DataFrame({
        'station_id': sid,
        'timestamp': pd.to_datetime(df['timestamp']).dt.tz_localize(None),
        'detector_online': res_det['online_predictions']
    })
    det_flags_list.append(df_stn)

all_det_flags = pd.concat(det_flags_list, ignore_index=True)
featured = featured.merge(all_det_flags, on=['station_id', 'timestamp'], how='left')
det_online_arr = featured['detector_online'].fillna(False).to_numpy(dtype=bool)
raw_nans = featured['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)

# Individual triggers
t_hard = np.asarray(row_hard, dtype=bool)
t_model_alone = (np.asarray(model_pct) > MODEL_ALONE_OVERRIDE_THRESHOLD) & (~np.asarray(is_regional, dtype=bool))
t_rule_bypass = (np.asarray(row_rule_conf) > RULE_CONFIDENCE_BYPASS) & (~np.asarray(is_regional, dtype=bool))
t_fusion = ((np.asarray(overall_confidence) > FUSION_ANOMALY_THRESHOLD) & (np.asarray(row_rule_conf) > 0)) & (~np.asarray(is_regional, dtype=bool))
t_glrt = det_online_arr & (~np.asarray(is_regional, dtype=bool))
t_nans = raw_nans

base_predicted = (
    row_hard
    | ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0) & (~is_regional))
    | ((model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD) & (~is_regional))
    | ((row_rule_conf > RULE_CONFIDENCE_BYPASS) & (~is_regional))
)
final_predicted = base_predicted | (det_online_arr & (~is_regional)) | raw_nans

feat_final = featured.copy()
feat_final = feat_final.merge(labels, on=['station_id', 'timestamp'], how='left')
feat_final['is_anomaly'] = feat_final['is_anomaly'].fillna(False).astype(bool) | raw_nans
feat_final['fault_type'] = feat_final['fault_type'].fillna('none')
feat_final['__predicted'] = final_predicted
feat_final['__predicted_fault_type'] = row_fault_type
feat_final.loc[det_online_arr & (feat_final['__predicted_fault_type'] == 'none'), '__predicted_fault_type'] = 'drift'
feat_final.loc[raw_nans, '__predicted_fault_type'] = 'dropout'

is_gt = np.asarray(feat_final['is_anomaly'], dtype=bool)
is_pred = np.asarray(final_predicted, dtype=bool)
gt_ft = np.asarray(feat_final['fault_type'])
pred_ft = np.asarray(feat_final['__predicted_fault_type'])

print("\n" + "=" * 100)
print("SEED 42 FALSE POSITIVE BREAKDOWN (4485 FPs)")
print("=" * 100)
fp_mask = (~is_gt) & is_pred
print(f"Total FP: {fp_mask.sum()}")
print(f"  FP from t_hard: {(fp_mask & t_hard).sum()}")
print(f"  FP from t_model_alone: {(fp_mask & t_model_alone).sum()}")
print(f"  FP from t_rule_bypass: {(fp_mask & t_rule_bypass).sum()}")
print(f"  FP from t_fusion: {(fp_mask & t_fusion).sum()}")
print(f"  FP from t_glrt: {(fp_mask & t_glrt).sum()}")
print(f"  FP from t_nans: {(fp_mask & t_nans).sum()}")

print("\nPredicted Fault Types for FP:")
print(pd.Series(pred_ft[fp_mask]).value_counts())

print("\n" + "=" * 100)
print("SEED 42 PER-FAULT RECALL & LOSS TRACE")
print("=" * 100)
for ft in ["drift", "spike", "frozen_value", "multivariate_inconsistency", "sensor_fail_low", "dropout", "unstructured_anomaly"]:
    mask = (gt_ft == ft)
    tot = mask.sum()
    tp = (mask & is_pred).sum()
    fn = (mask & ~is_pred).sum()
    rec = tp / tot if tot > 0 else 0.0
    print(f"\nFault: {ft} (Total GT: {tot}, TP: {tp}, FN: {fn}, Recall: {rec*100:.2f}%)")
    
    # What triggered on FN?
    fn_mask = mask & (~is_pred)
    if fn_mask.sum() > 0:
        # Check rule conf, model pct, overall conf, glrt
        rc_fn = np.asarray(row_rule_conf)[fn_mask]
        mp_fn = np.asarray(model_pct)[fn_mask]
        oc_fn = np.asarray(overall_confidence)[fn_mask]
        reg_fn = np.asarray(is_regional)[fn_mask]
        gl_fn = det_online_arr[fn_mask]
        
        print(f"  FN rule_conf mean: {rc_fn.mean():.2f}, max: {rc_fn.max():.2f}, >0: {(rc_fn > 0).sum()}/{len(rc_fn)}")
        print(f"  FN model_pct mean: {mp_fn.mean():.2f}, max: {mp_fn.max():.2f}, >50: {(mp_fn > 50).sum()}/{len(mp_fn)}")
        print(f"  FN overall_conf mean: {oc_fn.mean():.2f}, max: {oc_fn.max():.2f}, >50: {(oc_fn > 50).sum()}/{len(oc_fn)}")
        print(f"  FN is_regional true: {reg_fn.sum()}/{len(reg_fn)}")
        print(f"  FN det_online true: {gl_fn.sum()}/{len(gl_fn)}")

        # Print 3 representative FN examples with key features
        sample_indices = np.where(fn_mask)[0][:3]
        for s_idx in sample_indices:
            row = feat_final.iloc[s_idx]
            z_val = row.get('temp_z_roll24', 0.0)
            z_str = f"{z_val:.2f}" if isinstance(z_val, (int, float, np.floating)) else str(z_val)
            rc_val = float(np.asarray(row_rule_conf)[s_idx])
            mp_val = float(np.asarray(model_pct)[s_idx])
            oc_val = float(np.asarray(overall_confidence)[s_idx])
            print(f"    Example FN: Station={row['station_id']}, Time={row['timestamp']}, temp={row.get('temperature_c', 'N/A')}, press={row.get('pressure_hpa', 'N/A')}, hum={row.get('humidity_pct', 'N/A')}")
            print(f"      Features: temp_z_roll24={z_str}, rule_conf={rc_val:.1f}, model_pct={mp_val:.1f}, overall={oc_val:.1f}")
