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
    vectorized_model_scores, _featurize,
    MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD,
    MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS,
    add_frozen_channel_labels_from_reference
)
from model.spike_tracker import init_spike_state, step_spike_state
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel, ConditionalMultivariateDriftDetector,
    run_calibrated_rules, STATION_TO_CLUSTER
)

def run_fp_forensics(seed=42):
    print(f"=== DEEP FORENSIC FP TRACE ON SEED {seed} ===")
    
    # Fit causal models
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()
    
    theta_glrt = 30.0
    theta_cusum = 24.0
    detector = ConditionalMultivariateDriftDetector(glrt_thresh=theta_glrt, cusum_thresh=theta_cusum, allowance=0.45, persist_req=2)
    artifact = joblib.load(ARTIFACTS_PATH)

    data = generate_network_benchmark(regime='observable_v1', seed=seed, save_to_disk=False)
    
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

    # Labels
    label_cols = ['station_id', 'timestamp', 'is_anomaly', 'fault_type']
    labels = df_full[label_cols].copy()
    labels['is_anomaly'] = labels['is_anomaly'].fillna(False).astype(bool)
    labels['fault_type'] = labels['fault_type'].fillna('none')

    # Detector input
    df_in = df_full.drop(columns=['is_anomaly', 'fault_type', 'injected_delta', 'injected_start', 'injected_end'], errors='ignore')
    featured, _ = _featurize(df_in)
    featured['timestamp'] = pd.to_datetime(featured['timestamp']).dt.tz_localize(None)

    thresholds = artifact["rule_thresholds"]
    featured_base, row_hard, row_rule_conf, row_fault_type = run_calibrated_rules(featured.copy(), thresholds)
    model_pct = vectorized_model_scores(featured_base, artifact)
    overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
    
    flag_hard = row_hard
    flag_fusion = ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0))
    flag_model_alone = (model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD)
    flag_rule_bypass = (row_rule_conf > RULE_CONFIDENCE_BYPASS)
    base_predicted = (flag_hard | flag_fusion | flag_model_alone | flag_rule_bypass)

    # Stream Drift Detector
    det_flags_list = []
    drift_details_list = []
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
            'c_plus': res_det['c_plus'],
            'c_minus': res_det['c_minus'],
            'u_T': u_T,
            'z_T': Z[:, 0],
            'z_RH': Z[:, 1],
            'z_P': Z[:, 2],
        })
        det_flags_list.append(df_stn)

    all_det_flags = pd.concat(det_flags_list, ignore_index=True)
    featured = featured.merge(all_det_flags, on=['station_id', 'timestamp'], how='left')
    det_online_arr = featured['detector_online'].fillna(False).to_numpy(dtype=bool)
    raw_nans_arr = featured['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)

    final_predicted = base_predicted | det_online_arr | raw_nans_arr

    df_eval = featured.copy()
    df_eval = df_eval.merge(labels, on=['station_id', 'timestamp'], how='left')
    df_eval['is_anomaly'] = df_eval['is_anomaly'].fillna(False).astype(bool) | raw_nans_arr
    df_eval['fault_type'] = df_eval['fault_type'].fillna('none')
    df_eval['__predicted'] = final_predicted
    df_eval['flag_hard'] = flag_hard
    df_eval['flag_fusion'] = flag_fusion
    df_eval['flag_model_alone'] = flag_model_alone
    df_eval['flag_rule_bypass'] = flag_rule_bypass
    df_eval['flag_drift_stream'] = det_online_arr
    df_eval['flag_raw_nans'] = raw_nans_arr
    df_eval['row_rule_conf'] = row_rule_conf
    df_eval['row_fault_type'] = row_fault_type
    df_eval['model_pct'] = model_pct

    # Separate TP, FP, FN, TN
    is_gt = df_eval['is_anomaly'].to_numpy(dtype=bool)
    is_pred = df_eval['__predicted'].to_numpy(dtype=bool)

    df_eval['is_tp'] = is_gt & is_pred
    df_eval['is_fp'] = (~is_gt) & is_pred
    df_eval['is_fn'] = is_gt & (~is_pred)
    df_eval['is_tn'] = (~is_gt) & (~is_pred)

    fp_df = df_eval[df_eval['is_fp']].copy()
    tp_df = df_eval[df_eval['is_tp']].copy()

    total_fps = len(fp_df)
    total_tps = len(tp_df)
    total_fns = df_eval['is_fn'].sum()
    print(f"Total TPs: {total_tps}, Total FPs: {total_fps}, Total FNs: {total_fns}")
    print(f"Precision: {total_tps/(total_tps+total_fps)*100:.2f}%, Recall: {total_tps/(total_tps+total_fns)*100:.2f}%")

    # 1. DECOMPOSE FPS BY DETECTOR COMPONENT
    print("\n--- 1. FP BREAKDOWN BY DETECTOR SOURCE ---")
    sources = {
        "Hard Bounds / Raw Dropout": fp_df['flag_hard'] | fp_df['flag_raw_nans'],
        "Rule Confidence Bypass (>90)": fp_df['flag_rule_bypass'],
        "Fusion Rule+Model (>50)": fp_df['flag_fusion'],
        "Model Alone Override (>88)": fp_df['flag_model_alone'],
        "Drift Stream (GLRT/CUSUM)": fp_df['flag_drift_stream'],
    }
    for s_name, s_mask in sources.items():
        cnt = s_mask.sum()
        print(f"  {s_name:<30}: {cnt:>5} ({cnt/total_fps*100:>5.1f}%)")

    # 2. DECOMPOSE FPS BY RULE FAULT TYPE
    print("\n--- 2. FP BREAKDOWN BY ASSIGNED FAULT TYPE ---")
    print(fp_df['row_fault_type'].value_counts())

    # 3. DECOMPOSE FPS BY DRIFT STREAM VS RULE ENGINE OVERLAP
    print("\n--- 3. SOURCE OVERLAP IN FPS ---")
    drift_only = fp_df['flag_drift_stream'] & (~(fp_df['flag_rule_bypass'] | fp_df['flag_fusion'] | fp_df['flag_model_alone'] | fp_df['flag_hard']))
    rule_only = (~fp_df['flag_drift_stream']) & (fp_df['flag_rule_bypass'] | fp_df['flag_fusion'] | fp_df['flag_model_alone'] | fp_df['flag_hard'])
    both = fp_df['flag_drift_stream'] & (fp_df['flag_rule_bypass'] | fp_df['flag_fusion'] | fp_df['flag_model_alone'] | fp_df['flag_hard'])
    print(f"  Drift Stream Only : {drift_only.sum()} ({drift_only.sum()/total_fps*100:.1f}%)")
    print(f"  Rule Engine Only  : {rule_only.sum()} ({rule_only.sum()/total_fps*100:.1f}%)")
    print(f"  Both Active       : {both.sum()} ({both.sum()/total_fps*100:.1f}%)")

    # 4. DEEP DIVE: DRIFT STREAM FPS
    fp_drift = fp_df[fp_df['flag_drift_stream']].copy()
    print(f"\n--- 4. DRIFT STREAM FP ANALYSIS (N = {len(fp_drift)}) ---")
    print(f"  Mean GLRT Lambda : {fp_drift['glrt_lambda'].mean():.2f} (std: {fp_drift['glrt_lambda'].std():.2f}, min: {fp_drift['glrt_lambda'].min():.2f}, max: {fp_drift['glrt_lambda'].max():.2f})")
    print(f"  Mean |u_T|       : {fp_drift['u_T'].abs().mean():.2f}")
    print(f"  Mean |z_T|       : {fp_drift['z_T'].abs().mean():.2f}")
    print(f"  Mean |z_RH|      : {fp_drift['z_RH'].abs().mean():.2f}")
    print(f"  Mean |z_P|       : {fp_drift['z_P'].abs().mean():.2f}")

    # 5. DEEP DIVE: RULE ENGINE FPS
    fp_rule = fp_df[rule_only | both].copy()
    print(f"\n--- 5. RULE ENGINE FP ANALYSIS (N = {len(fp_rule)}) ---")
    for ft, g in fp_rule.groupby('row_fault_type'):
        print(f"  Fault Type: {ft:<26} Count: {len(g):>4} ({len(g)/len(fp_rule)*100:>5.1f}%)")

    # 6. TEMPORAL PERSISTENCE / CONTINUOUS EPISODE LENGTHS OF FPS
    print("\n--- 6. CONTINUOUS FP EPISODE LENGTH DISTRIBUTION ---")
    # Group by station and compute contiguous run lengths
    run_lengths = []
    for sid, g in df_eval.groupby('station_id'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        fp_mask = g['is_fp'].values
        cur_len = 0
        for val in fp_mask:
            if val:
                cur_len += 1
            else:
                if cur_len > 0:
                    run_lengths.append(cur_len)
                    cur_len = 0
        if cur_len > 0:
            run_lengths.append(cur_len)
    
    rl_series = pd.Series(run_lengths)
    print(f"  Total distinct FP episodes: {len(rl_series)}")
    print(f"  Isolated single-hour FPs (len=1): {(rl_series == 1).sum()} ({(rl_series == 1).sum()/len(rl_series)*100:.1f}%)")
    print(f"  Short FPs (len 2-4): {((rl_series >= 2) & (rl_series <= 4)).sum()} ({((rl_series >= 2) & (rl_series <= 4)).sum()/len(rl_series)*100:.1f}%)")
    print(f"  Medium FPs (len 5-12): {((rl_series >= 5) & (rl_series <= 12)).sum()} ({((rl_series >= 5) & (rl_series <= 12)).sum()/len(rl_series)*100:.1f}%)")
    print(f"  Long continuous FP episodes (len > 12): {(rl_series > 12).sum()} ({(rl_series > 12).sum()/len(rl_series)*100:.1f}%)")
    print(f"  Max continuous FP episode length: {rl_series.max()} hours")

    # 7. DIURNAL / HOUR-OF-DAY DISTRIBUTION OF FPS
    print("\n--- 7. HOUR-OF-DAY FP DISTRIBUTION ---")
    fp_df['hour'] = fp_df['timestamp'].dt.hour
    tp_df['hour'] = tp_df['timestamp'].dt.hour
    hour_df = pd.DataFrame({
        'FP_count': fp_df['hour'].value_counts().sort_index(),
        'TP_count': tp_df['hour'].value_counts().sort_index()
    })
    hour_df['FP_pct'] = hour_df['FP_count'] / total_fps * 100
    hour_df['TP_pct'] = hour_df['TP_count'] / total_tps * 100
    print(hour_df.to_string())

if __name__ == '__main__':
    run_fp_forensics(seed=42)
