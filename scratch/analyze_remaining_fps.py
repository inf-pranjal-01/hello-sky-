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
from scratch.test_spike_physical_calibration import run_physical_rules
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel, ConditionalMultivariateDriftDetector,
    STATION_TO_CLUSTER
)

def analyze_remaining_fps(seed=42):
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    detector = ConditionalMultivariateDriftDetector(glrt_thresh=30.0, cusum_thresh=24.0, allowance=0.45, persist_req=2)
    artifact = joblib.load(ARTIFACTS_PATH)

    data = generate_network_benchmark(regime='observable_v1', seed=seed, save_to_disk=False)
    
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
            'detector_online': res_det['online_predictions'],
            'glrt_lambda': res_det['glrt_lambdas'],
            'u_T': u_T,
            'z_T': Z[:, 0],
            'z_RH': Z[:, 1],
            'z_P': Z[:, 2]
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
    featured_base, row_hard, row_rule_conf, row_fault_type = run_physical_rules(featured.copy(), thresholds)
    model_pct = vectorized_model_scores(featured_base, artifact)
    overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
    
    flag_hard = row_hard
    flag_fusion = ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0))
    flag_model_alone = (model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD)
    flag_rule_bypass = (row_rule_conf > RULE_CONFIDENCE_BYPASS)
    base_predicted = (flag_hard | flag_fusion | flag_model_alone | flag_rule_bypass)

    featured = featured.merge(all_det_flags, on=['station_id', 'timestamp'], how='left')
    det_online_arr = featured['detector_online'].fillna(False).to_numpy(dtype=bool)
    raw_nans_arr = featured['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)

    final_predicted = base_predicted | det_online_arr | raw_nans_arr

    df_eval = featured.copy()
    df_eval = df_eval.merge(labels, on=['station_id', 'timestamp'], how='left')
    df_eval['is_anomaly'] = df_eval['is_anomaly'].fillna(False).astype(bool) | raw_nans_arr
    df_eval['fault_type'] = df_eval['fault_type'].fillna('none')
    df_eval['__predicted'] = final_predicted
    df_eval['__predicted_fault_type'] = row_fault_type
    df_eval.loc[det_online_arr & (df_eval['__predicted_fault_type'] == 'none'), '__predicted_fault_type'] = 'drift'
    df_eval.loc[raw_nans_arr, '__predicted_fault_type'] = 'dropout'

    is_gt = df_eval['is_anomaly'].to_numpy(dtype=bool)
    is_pred = df_eval['__predicted'].to_numpy(dtype=bool)

    df_eval['is_tp'] = is_gt & is_pred
    df_eval['is_fp'] = (~is_gt) & is_pred
    df_eval['flag_hard'] = flag_hard
    df_eval['flag_fusion'] = flag_fusion
    df_eval['flag_model_alone'] = flag_model_alone
    df_eval['flag_rule_bypass'] = flag_rule_bypass
    df_eval['flag_drift_stream'] = det_online_arr

    fp_df = df_eval[df_eval['is_fp']].copy()
    print(f"Total FPs: {len(fp_df)}")
    
    print("\n--- FP BREAKDOWN BY ACTIVE DETECTOR ---")
    sources = {
        "Rule Bypass (>90)": fp_df['flag_rule_bypass'],
        "Fusion Rule+Model": fp_df['flag_fusion'],
        "Model Alone": fp_df['flag_model_alone'],
        "Drift Stream": fp_df['flag_drift_stream'],
    }
    for s_name, s_mask in sources.items():
        cnt = s_mask.sum()
        print(f"  {s_name:<25}: {cnt:>5} ({cnt/len(fp_df)*100:>5.1f}%)")

    print("\n--- FP BREAKDOWN BY PREDICTED FAULT TYPE ---")
    print(fp_df['__predicted_fault_type'].value_counts())

if __name__ == '__main__':
    analyze_remaining_fps(seed=42)
