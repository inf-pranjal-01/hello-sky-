import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

from config import CLUSTERS, RULE_BASE_CONFIDENCE
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    ARTIFACTS_PATH, PHYSICAL_BOUNDS,
    vectorized_model_scores, _featurize, _score_and_report,
    MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD,
    MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS,
    add_frozen_channel_labels_from_reference
)
from evaluation.episodic_eval import compute_episodic_result
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel, ConditionalMultivariateDriftDetector,
    STATION_TO_CLUSTER, run_calibrated_rules
)

def diagnose(seed=42):
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    detector_T = ConditionalMultivariateDriftDetector(glrt_thresh=32.0, cusum_thresh=26.0, allowance=0.45, persist_req=2)
    detector_RH = ConditionalMultivariateDriftDetector(glrt_thresh=42.0, cusum_thresh=32.0, allowance=0.55, persist_req=3)
    detector_P = ConditionalMultivariateDriftDetector(glrt_thresh=34.0, cusum_thresh=28.0, allowance=0.45, persist_req=2)
    artifact = joblib.load(ARTIFACTS_PATH)

    data = generate_network_benchmark(regime='observable_v2', seed=seed, save_to_disk=False)
    
    spatial_z_dict = defaultdict(dict)
    det_flags_list = []

    for sid in STATION_TO_CLUSTER:
        df = data[sid].sort_values('timestamp').reset_index(drop=True)
        df_clean = df.copy()
        df_clean[['temperature_c', 'pressure_hpa', 'humidity_pct']] = df_clean[['temperature_c', 'pressure_hpa', 'humidity_pct']].ffill().bfill()

        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: data[pid].sort_values('timestamp').reset_index(drop=True) for pid in peer_ids}

        hr = uncertainty_model.get_hour_regime(pd.to_datetime(df_clean['timestamp']).dt.hour.values)

        Z_list = []
        for p in joint_model.channels:
            y_hat, _ = normal_model.predict_target_robust(sid, p, df_clean, peer_dfs)
            r = df_clean[p].values - y_hat
            sig = uncertainty_model.predict_sigma(sid, p, y_hat, df_clean['timestamp'].values)
            z_p = np.nan_to_num(r / np.maximum(0.1, sig), nan=0.0)
            Z_list.append(z_p)
            spatial_z_dict[sid][p] = z_p
            
        Z = np.column_stack(Z_list)
        n_rows = len(Z)

        u_T, sig_T = np.zeros(n_rows), np.ones(n_rows)
        u_RH, sig_RH = np.zeros(n_rows), np.ones(n_rows)
        u_P, sig_P = np.zeros(n_rows), np.ones(n_rows)

        for i in range(n_rows):
            u_T[i], sig_T[i] = joint_model.get_conditional_innovation(sid, hr[i], Z[i], 0)
            u_RH[i], sig_RH[i] = joint_model.get_conditional_innovation(sid, hr[i], Z[i], 1)
            u_P[i], sig_P[i] = joint_model.get_conditional_innovation(sid, hr[i], Z[i], 2)

        res_T = detector_T.process_station_stream(np.nan_to_num(u_T, nan=0.0), np.nan_to_num(sig_T, nan=1.0), df['timestamp'].values)
        res_RH = detector_RH.process_station_stream(np.nan_to_num(u_RH, nan=0.0), np.nan_to_num(sig_RH, nan=1.0), df['timestamp'].values)
        res_P = detector_P.process_station_stream(np.nan_to_num(u_P, nan=0.0), np.nan_to_num(sig_P, nan=1.0), df['timestamp'].values)
        
        drift_combined = res_T['online_predictions'] | res_RH['online_predictions'] | res_P['online_predictions']

        df_stn = pd.DataFrame({
            'station_id': sid,
            'timestamp': pd.to_datetime(df['timestamp']).dt.tz_localize(None),
            'det_drift_T': res_T['online_predictions'],
            'det_drift_RH': res_RH['online_predictions'],
            'det_drift_P': res_P['online_predictions'],
            'detector_online': drift_combined
        })
        det_flags_list.append(df_stn)

    all_det_flags = pd.concat(det_flags_list, ignore_index=True)

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
    featured, _ = _featurize(df_in)
    featured['timestamp'] = pd.to_datetime(featured['timestamp']).dt.tz_localize(None)

    thresholds = artifact["rule_thresholds"]
    featured_base, row_hard, row_rule_conf, row_fault_type = run_calibrated_rules(featured.copy(), thresholds, spatial_z_dict)
    model_pct = vectorized_model_scores(featured_base, artifact)
    overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
    
    pred_hard = row_hard
    pred_fusion = (overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0)
    pred_model_alone = model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD
    pred_rule_bypass = row_rule_conf > RULE_CONFIDENCE_BYPASS

    featured = featured.merge(all_det_flags, on=['station_id', 'timestamp'], how='left')
    det_online_arr = featured['detector_online'].fillna(False).to_numpy(dtype=bool)
    raw_nans_arr = featured['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)

    base_predicted = pred_hard | pred_fusion | pred_model_alone | pred_rule_bypass
    final_predicted = base_predicted | det_online_arr | raw_nans_arr

    feat_final = featured.copy()
    feat_final = feat_final.merge(labels, on=['station_id', 'timestamp'], how='left')
    feat_final['is_anomaly'] = feat_final['is_anomaly'].fillna(False).astype(bool) | raw_nans_arr
    feat_final['fault_type'] = feat_final['fault_type'].fillna('none')
    feat_final['__predicted'] = final_predicted
    feat_final['__predicted_fault_type'] = row_fault_type
    feat_final.loc[det_online_arr & (feat_final['__predicted_fault_type'] == 'none'), '__predicted_fault_type'] = 'drift'
    feat_final.loc[raw_nans_arr, '__predicted_fault_type'] = 'dropout'

    is_gt = feat_final['is_anomaly'].to_numpy(dtype=bool)
    fault_types = feat_final['fault_type'].to_numpy()

    is_fp = final_predicted & (~is_gt)
    is_fn = (~final_predicted) & is_gt
    is_tp = final_predicted & is_gt

    print(f"Total rows: {len(feat_final)}")
    print(f"Total GT Anomaly: {is_gt.sum()}, Total GT Normal: {(~is_gt).sum()}")
    print(f"TP: {is_tp.sum()}, FP: {is_fp.sum()}, FN: {is_fn.sum()}")
    print(f"Precision: {is_tp.sum() / (is_tp.sum() + is_fp.sum()) * 100:.2f}%")
    print(f"Recall: {is_tp.sum() / (is_tp.sum() + is_fn.sum()) * 100:.2f}%")

    print("\n--- FP BREAKDOWN BY DETECTOR SOURCE ---")
    print(f"Total FPs: {is_fp.sum()}")
    print(f"  Drift Detector alone FPs: {(det_online_arr & ~is_gt & ~base_predicted).sum()}")
    print(f"  Drift Detector T (any FP): {(featured['det_drift_T'].fillna(False).to_numpy(dtype=bool) & ~is_gt).sum()}")
    print(f"  Drift Detector RH (any FP):{(featured['det_drift_RH'].fillna(False).to_numpy(dtype=bool) & ~is_gt).sum()}")
    print(f"  Drift Detector P (any FP): {(featured['det_drift_P'].fillna(False).to_numpy(dtype=bool) & ~is_gt).sum()}")
    print(f"  Model Alone FPs:          {(pred_model_alone & ~is_gt & ~pred_hard & ~pred_rule_bypass).sum()}")
    print(f"  Fusion FPs:               {(pred_fusion & ~is_gt).sum()}")
    print(f"  Rule Bypass FPs:          {(pred_rule_bypass & ~is_gt).sum()}")
    print(f"  Hard Rule FPs:            {(pred_hard & ~is_gt).sum()}")

    print("\n--- RULE FAULT TYPE ASSIGNMENT ON FPs ---")
    fp_df = pd.DataFrame({'rule_ft': row_fault_type[is_fp], 'rule_conf': row_rule_conf[is_fp]})
    print(fp_df['rule_ft'].value_counts())

    print("\n--- FN BREAKDOWN BY GROUND TRUTH FAULT TYPE ---")
    fn_df = pd.DataFrame({'fault_type': fault_types[is_fn]})
    print(fn_df['fault_type'].value_counts())

    print("\n--- RECALL BREAKDOWN BY GROUND TRUTH FAULT TYPE ---")
    for ft in np.unique(fault_types):
        if ft == 'none': continue
        mask_ft = (fault_types == ft)
        tps = (final_predicted & mask_ft).sum()
        total_ft = mask_ft.sum()
        print(f"  {ft:<28}: {tps}/{total_ft} ({tps/total_ft*100:.2f}%)")

if __name__ == '__main__':
    diagnose(42)
