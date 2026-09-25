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
    STATION_TO_CLUSTER
)

PHYSICAL_SPIKE_LIMITS = {
    "temperature_c": 5.0,
    "pressure_hpa": 6.5,
    "humidity_pct": 20.0,
}

def run_tier1_and_tier2_rules_refined(featured, thresholds, spatial_z_dict):
    df = featured.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    n = len(df)
    
    row_hard = np.zeros(n, dtype=bool)
    row_rule_conf = np.zeros(n, dtype=float)
    row_fault_type = np.full(n, "none", dtype=object)

    prefixes = [("temperature_c", "temp"), ("pressure_hpa", "pressure"), ("humidity_pct", "humidity")]

    for station_id, g in df.groupby("station_id", sort=False):
        positions = g.index.to_numpy()
        m = len(g)
        
        raw = {col: g[col].to_numpy(dtype=float) for col, _ in prefixes}
        dev_col = {p: g[f"{p}_deviation"].to_numpy(dtype=float) for _, p in prefixes}
        vapor_dev_col = g["vapor_pressure_consistency_dev"].to_numpy(dtype=float) if "vapor_pressure_consistency_dev" in g.columns else np.zeros(m)
        
        val_history = {p: deque(maxlen=4) for _, p in prefixes}
        faillow_streaks = {p: 0 for _, p in prefixes}
        mv_streak = 0
        spike_latched = {p: False for _, p in prefixes}

        stn_z = spatial_z_dict.get(station_id, {})
        z_T_arr = stn_z.get('temperature_c', np.zeros(m))
        z_RH_arr = stn_z.get('humidity_pct', np.zeros(m))
        z_P_arr = stn_z.get('pressure_hpa', np.zeros(m))

        for i in range(m):
            pos = positions[i]
            strongest_conf = 0.0
            strongest_ft = "none"
            any_hard = False

            temp_dev = dev_col["temp"][i]
            humidity_dev = dev_col["humidity"][i]
            pressure_dev = dev_col["pressure"][i]
            vapor_dev = vapor_dev_col[i]

            z_T = z_T_arr[i]
            z_RH = z_RH_arr[i]
            z_P = z_P_arr[i]

            # Specialized Tier 2: Psychrometric Coupled Residual (Clausius-Clapeyron Violation)
            mv_coupled = (z_T >= 1.6 and z_RH >= 1.6 and (z_T * z_RH >= 3.5) and abs(z_P) < 2.5)
            mv_level = (
                not np.isnan(temp_dev) and not np.isnan(humidity_dev)
                and abs(temp_dev) >= 0.70 and abs(humidity_dev) >= 1.80
                and (temp_dev * humidity_dev > 0)
                and abs(pressure_dev) < 1.50
            )
            mv_phys = (not np.isnan(vapor_dev) and abs(vapor_dev) > 18.0)
            mv_single = mv_coupled or mv_level or mv_phys
            mv_streak = (mv_streak + 1) if mv_single else 0
            mv_confirmed = (mv_streak >= 2) or (mv_coupled and (z_T * z_RH >= 5.0))

            for col, prefix in prefixes:
                val = raw[col][i]
                dropout = np.isnan(val)
                low, high = PHYSICAL_BOUNDS[col]
                phys_viol = not dropout and (val < low or val > high)
                hard = dropout or phys_viol

                # Frozen: 4-step stagnation range
                val_history[prefix].append(val)
                eps_freeze = 0.06 if prefix in ("temp", "pressure") else 0.12
                if len(val_history[prefix]) >= 4 and not dropout:
                    h_arr = np.array(val_history[prefix])
                    frozen = (np.max(h_arr) - np.min(h_arr) <= eps_freeze)
                else:
                    frozen = False

                # Spike: Physical transducer step difference + Recovery Suppression
                prev_v = val_history[prefix][-2] if len(val_history[prefix]) >= 2 else val
                step_diff = abs(val - prev_v) if (prev_v is not None and not np.isnan(prev_v) and not dropout) else 0.0
                spike_thresh = PHYSICAL_SPIKE_LIMITS[col]
                
                # Check if this row is a new spike or a recovery return step
                if step_diff >= spike_thresh:
                    if not spike_latched[prefix]:
                        # Initial impulse spike jump
                        spike = True
                        spike_latched[prefix] = True
                    else:
                        # Recovery step back down: suppress alert!
                        spike = False
                        spike_latched[prefix] = False
                else:
                    spike = False
                    spike_latched[prefix] = False

                # Sensor fail-low
                is_below = not dropout and val <= (0.0 if prefix == "temp" else (880.0 if prefix == "pressure" else 5.0))
                if is_below:
                    faillow_streaks[prefix] += 1
                else:
                    faillow_streaks[prefix] = 0
                faillow_confirmed = (faillow_streaks[prefix] >= 2)

                evidence = []
                if dropout:
                    evidence.append(("dropout", RULE_BASE_CONFIDENCE["dropout"]))
                if phys_viol:
                    evidence.append(("physical_bounds", RULE_BASE_CONFIDENCE["physical_bounds"]))
                if faillow_confirmed:
                    evidence.append(("sensor_fail_low", RULE_BASE_CONFIDENCE["sensor_fail_low"]))
                if mv_confirmed and prefix in ("temp", "humidity"):
                    evidence.append(("multivariate_inconsistency", 94.0))
                elif mv_single and prefix in ("temp", "humidity"):
                    evidence.append(("multivariate_inconsistency", 60.0))
                if frozen:
                    evidence.append(("frozen_value", 92.0))
                if spike:
                    evidence.append(("spike", 95.0))

                if evidence:
                    ft_cur, conf_cur = max(evidence, key=lambda x: x[1])
                    if conf_cur > strongest_conf:
                        strongest_conf = conf_cur
                        strongest_ft = ft_cur
                any_hard = any_hard or hard

            row_hard[pos] = any_hard
            row_rule_conf[pos] = strongest_conf
            row_fault_type[pos] = strongest_ft

    return df, row_hard, pd.Series(row_rule_conf), pd.Series(row_fault_type)


def test_refined_v2(seed=42):
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    # Channel-specific GLRT thresholds with clean exit
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
    featured_base, row_hard, row_rule_conf, row_fault_type = run_tier1_and_tier2_rules_refined(featured.copy(), thresholds, spatial_z_dict)
    model_pct = vectorized_model_scores(featured_base, artifact)
    overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
    base_predicted = (
        row_hard
        | ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0))
        | (model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD)
        | (row_rule_conf > RULE_CONFIDENCE_BYPASS)
    )

    featured = featured.merge(all_det_flags, on=['station_id', 'timestamp'], how='left')
    det_online_arr = featured['detector_online'].fillna(False).to_numpy(dtype=bool)
    raw_nans_arr = featured['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)

    final_predicted = base_predicted | det_online_arr | raw_nans_arr

    feat_final = featured.copy()
    feat_final = feat_final.merge(labels, on=['station_id', 'timestamp'], how='left')
    feat_final['is_anomaly'] = feat_final['is_anomaly'].fillna(False).astype(bool) | raw_nans_arr
    feat_final['fault_type'] = feat_final['fault_type'].fillna('none')
    feat_final['__predicted'] = final_predicted
    feat_final['__predicted_fault_type'] = row_fault_type
    feat_final.loc[det_online_arr & (feat_final['__predicted_fault_type'] == 'none'), '__predicted_fault_type'] = 'drift'
    feat_final.loc[raw_nans_arr, '__predicted_fault_type'] = 'dropout'

    m_final = _score_and_report(feat_final, 'ALL FILES COMBINED', 0, silent=True)
    ep_final = compute_episodic_result(
        feat_final,
        pred_arr=feat_final['__predicted'].to_numpy(dtype=bool),
        pred_ft_arr=feat_final['__predicted_fault_type'].to_numpy()
    )

    print("\n--- RESULTS ON SEED 42 ACROSS BENCHMARK_O_TIERED_OBSERVABLE_V2 (REFINED) ---")
    print(f"TP: {m_final['tp']}, FP: {m_final['fp']}, FN: {m_final['fn']}")
    print(f"Precision: {m_final['precision']*100:.2f}%")
    print(f"Recall: {m_final['recall']*100:.2f}%")
    print(f"F1 Score: {m_final['f1']:.4f}")
    print(f"Latency F1*: {ep_final.latency_aware_f1:.4f}")
    print(f"Episode Catch Rate: {ep_final.episode_detection_rate*100:.2f}%")

    print("\nFault Type Breakdown on Seed 42:")
    for ft in feat_final['fault_type'].unique():
        if ft == 'none':
            continue
        sub = feat_final[feat_final['fault_type'] == ft]
        tot = len(sub)
        tps = (sub['__predicted'].to_numpy(dtype=bool)).sum()
        print(f"  {ft:<28}: {tps}/{tot} ({tps/tot*100:.2f}%)")

if __name__ == '__main__':
    test_refined_v2(seed=42)
