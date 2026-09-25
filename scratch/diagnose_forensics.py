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

def run_diagnostics():
    print("Fitting Causal Models on Clean 60% Calibration Slice...")
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()

    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()

    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    artifact = joblib.load(ARTIFACTS_PATH)
    canonical_seeds = [42, 101, 202, 2024, 8888, 20260924, 45456231412727229999]

    detector = ConditionalMultivariateDriftDetector(glrt_thresh=26.0, cusum_thresh=20.0, allowance=0.45, persist_req=2)

    # Accumulators for detailed forensic tracing
    fp_sources = {
        "hard": 0,
        "model_alone": 0,
        "rule_bypass": 0,
        "fusion": 0,
        "glrt_det_online": 0,
        "raw_nans": 0,
        "combo": 0
    }
    
    fp_predicted_faults = {}
    
    # Fault type forensic tracking
    fault_forensics = {
        ft: {
            "total_gt": 0,
            "tp": 0,
            "fn": 0,
            "seen_by_hard": 0,
            "seen_by_model": 0,
            "seen_by_rule": 0,
            "seen_by_fusion": 0,
            "seen_by_glrt": 0,
            "seen_by_any_before_suppression": 0,
            "suppressed_by_regional": 0,
            "suppressed_by_rule_conf_zero": 0,
            "completely_unseen": 0,
        }
        for ft in ["drift", "spike", "frozen_value", "multivariate_inconsistency", "sensor_fail_low", "dropout", "unstructured_anomaly"]
    }

    print("Running 7-Seed Forensics...")
    for seed in canonical_seeds:
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

        # Stream Conditional Drift Detector
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

        # FP Analysis
        fp_mask = (~is_gt) & is_pred
        for i in np.where(fp_mask)[0]:
            h = t_hard[i]
            ma = t_model_alone[i]
            rb = t_rule_bypass[i]
            fu = t_fusion[i]
            gl = t_glrt[i]
            na = t_nans[i]

            sources = []
            if h: sources.append("hard")
            if ma: sources.append("model_alone")
            if rb: sources.append("rule_bypass")
            if fu: sources.append("fusion")
            if gl: sources.append("glrt_det_online")
            if na: sources.append("raw_nans")

            if len(sources) == 1:
                fp_sources[sources[0]] += 1
            else:
                fp_sources["combo"] += 1
                for s in sources:
                    fp_sources[s] += 1 # also track overlapping count

            pft = pred_ft[i]
            fp_predicted_faults[pft] = fp_predicted_faults.get(pft, 0) + 1

        # FN & Recall Analysis per Fault Type
        for ft in fault_forensics.keys():
            ft_mask = (gt_ft == ft)
            n_ft = ft_mask.sum()
            fault_forensics[ft]["total_gt"] += n_ft
            if n_ft == 0:
                continue

            sub_indices = np.where(ft_mask)[0]
            for i in sub_indices:
                pred = is_pred[i]
                if pred:
                    fault_forensics[ft]["tp"] += 1
                else:
                    fault_forensics[ft]["fn"] += 1

                # Check component activations before and after suppression
                h = t_hard[i]
                ma = (np.asarray(model_pct)[i] > MODEL_ALONE_OVERRIDE_THRESHOLD)
                rb = (np.asarray(row_rule_conf)[i] > RULE_CONFIDENCE_BYPASS)
                fu_pre = (np.asarray(overall_confidence)[i] > FUSION_ANOMALY_THRESHOLD)
                rc_nonzero = (np.asarray(row_rule_conf)[i] > 0)
                gl = det_online_arr[i]
                reg = np.asarray(is_regional)[i]

                if h: fault_forensics[ft]["seen_by_hard"] += 1
                if ma: fault_forensics[ft]["seen_by_model"] += 1
                if rb: fault_forensics[ft]["seen_by_rule"] += 1
                if fu_pre and rc_nonzero: fault_forensics[ft]["seen_by_fusion"] += 1
                if gl: fault_forensics[ft]["seen_by_glrt"] += 1

                any_pre = h or ma or rb or (fu_pre and rc_nonzero) or gl
                if any_pre:
                    fault_forensics[ft]["seen_by_any_before_suppression"] += 1
                    if reg and not h and not (gl and not reg):
                        fault_forensics[ft]["suppressed_by_regional"] += 1
                else:
                    if fu_pre and not rc_nonzero:
                        fault_forensics[ft]["suppressed_by_rule_conf_zero"] += 1
                    else:
                        fault_forensics[ft]["completely_unseen"] += 1

    print("\n" + "=" * 100)
    print("7-SEED FALSE POSITIVE BREAKDOWN (TOTAL & SOURCES)")
    print("=" * 100)
    for src, count in fp_sources.items():
        print(f"  FP Source '{src}': {count} (Mean per seed: {count/7:.1f})")

    print("\nPredicted Fault Types for False Positives:")
    for pft, count in sorted(fp_predicted_faults.items(), key=lambda x: x[1], reverse=True):
        print(f"  Predicted '{pft}': {count} (Mean per seed: {count/7:.1f})")

    print("\n" + "=" * 110)
    print("7-SEED PER-FAULT FORENSICS & ATTRIBUTION BREAKDOWN")
    print("=" * 110)
    print(f"{'Fault Type':<25} {'Total GT':<10} {'TP':<8} {'FN':<8} {'Recall':<8} {'Hard':<6} {'Model':<7} {'Rule':<6} {'GLRT':<6} {'RegSupp':<8} {'RC0Supp':<8} {'Unseen':<8}")
    print("-" * 110)
    for ft, s in fault_forensics.items():
        tot = s["total_gt"]
        rec = s["tp"] / tot if tot > 0 else 0.0
        print(f"{ft:<25} {tot:<10} {s['tp']:<8} {s['fn']:<8} {rec*100:>5.1f}%  {s['seen_by_hard']:<6} {s['seen_by_model']:<7} {s['seen_by_rule']:<6} {s['seen_by_glrt']:<6} {s['suppressed_by_regional']:<8} {s['suppressed_by_rule_conf_zero']:<8} {s['completely_unseen']:<8}")

if __name__ == '__main__':
    run_diagnostics()
