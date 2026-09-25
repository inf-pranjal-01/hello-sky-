import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from config import CLUSTERS, MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD, MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    ARTIFACTS_PATH, vectorized_model_scores, run_rule_engine_and_health,
    apply_spatial_corroboration, _score_and_report, _featurize,
    add_frozen_channel_labels_from_reference
)
from evaluation.episodic_eval import compute_episodic_result
from scratch.run_benchmark_o_evaluation import (
    STATION_TO_CLUSTER, CausalNormalBehaviorModel,
    ConditionalResidualUncertaintyModel, ConditionalMultivariateJointModel,
    ConditionalMultivariateDriftDetector
)

def test_single_seed(seed=42, glrt_th=18.0, cusum_th=14.0, model_override_th=95.0):
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    detector = ConditionalMultivariateDriftDetector(glrt_thresh=glrt_th, cusum_thresh=cusum_th, allowance=0.35)
    artifact = joblib.load(ARTIFACTS_PATH)

    data = generate_network_benchmark(regime='operational_v1', seed=seed, save_to_disk=False)

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

    featured_base, row_hard, row_rule_conf, row_fault_type, _, _ = run_rule_engine_and_health(featured.copy(), artifact)
    row_rule_conf, row_fault_type, _ = apply_spatial_corroboration(
        featured_base, row_hard, row_rule_conf, row_fault_type, artifact, gate_mode='new'
    )
    model_pct = vectorized_model_scores(featured_base, artifact)
    overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
    is_regional = (row_fault_type == "REGIONAL_EVENT")

    base_predicted = (
        row_hard
        | ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0) & (~is_regional))
        | ((model_pct > model_override_th) & (~is_regional))
        | ((row_rule_conf > RULE_CONFIDENCE_BYPASS) & (~is_regional))
    )

    det_flags_list = []
    for sid in STATION_TO_CLUSTER:
        df = data[sid].sort_values('timestamp').reset_index(drop=True)
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: data[pid].sort_values('timestamp').reset_index(drop=True) for pid in peer_ids}
        hr = uncertainty_model.get_hour_regime(pd.to_datetime(df['timestamp']).dt.hour.values)

        Z_list = []
        for p in joint_model.channels:
            y_hat, _ = normal_model.predict_target_robust(sid, p, df, peer_dfs)
            r = df[p].values - y_hat
            sig = uncertainty_model.predict_sigma(sid, p, y_hat, df['timestamp'].values)
            Z_list.append(r / np.maximum(0.1, sig))
        Z = np.column_stack(Z_list)
        n_rows = len(Z)

        u_T = np.zeros(n_rows)
        sig_cond_T = np.ones(n_rows)
        for i in range(n_rows):
            u, s = joint_model.get_conditional_innovation(sid, hr[i], Z[i], 0)
            u_T[i] = u
            sig_cond_T[i] = s

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

    final_predicted = base_predicted | (det_online_arr & (~is_regional)) | raw_nans

    feat_final = featured.copy()
    feat_final = feat_final.merge(labels, on=['station_id', 'timestamp'], how='left')
    feat_final['is_anomaly'] = feat_final['is_anomaly'].fillna(False).astype(bool) | raw_nans
    feat_final['fault_type'] = feat_final['fault_type'].fillna('none')
    feat_final['__predicted'] = final_predicted
    feat_final['__predicted_fault_type'] = row_fault_type

    m_final = _score_and_report(feat_final, 'ALL FILES COMBINED', 0, silent=True)
    ep_final = compute_episodic_result(
        feat_final,
        pred_arr=feat_final['__predicted'].to_numpy(dtype=bool),
        pred_ft_arr=feat_final['__predicted_fault_type'].to_numpy()
    )

    print(f"Seed {seed} -> Precision: {m_final['precision']*100:.2f}% | Recall: {m_final['recall']*100:.2f}% | F1: {m_final['f1']:.4f} | TP: {m_final['tp']} | FP: {m_final['fp']} | FN: {m_final['fn']} | F1*: {ep_final.latency_aware_f1:.4f}")

    for ftype in ['drift', 'spike', 'frozen_value', 'multivariate_inconsistency', 'sensor_fail_low', 'dropout', 'unstructured_anomaly']:
        sub = feat_final[feat_final['fault_type'] == ftype]
        tot = len(sub)
        rec = (sub['__predicted'].to_numpy(dtype=bool)).sum() / tot if tot > 0 else 0.0
        print(f"  {ftype:<28}: {rec*100:.2f}% ({tot} rows)")

if __name__ == '__main__':
    test_single_seed()
