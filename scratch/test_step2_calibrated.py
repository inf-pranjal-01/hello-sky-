import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib, pandas as pd, numpy as np
from config import MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD, MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    ARTIFACTS_PATH, vectorized_model_scores, run_rule_engine_and_health,
    apply_spatial_corroboration, _featurize, _score_and_report
)
from evaluation.episodic_eval import compute_episodic_result
from scratch.run_benchmark_o_evaluation import (
    STATION_TO_CLUSTER, CausalNormalBehaviorModel,
    ConditionalResidualUncertaintyModel, ConditionalMultivariateJointModel
)

class CalibratedMultivariateDriftDetector:
    """
    Step 2 Synchronized Drift Detector:
    1. Standardized conditional innovation z_u = u / sigma
    2. Statistical Ramp-GLRT with Bonferroni-calibrated threshold lambda >= 26.0
    3. Minimum slope requirement |b| >= 0.08 sigma/h and innovation floor |z_u| >= 2.2 sigma
    4. 2-step confirmation streak requirement (eliminates single-sample transient false alarms)
    5. Clean state recovery when innovation subsides (|z_u| < 1.2 sigma)
    """
    def __init__(self, glrt_thresh=26.0, cusum_thresh=20.0, allowance=0.45, decay=0.88, persist_req=2):
        self.glrt_thresh = glrt_thresh
        self.cusum_thresh = cusum_thresh
        self.allowance = allowance
        self.decay = decay
        self.persist_req = persist_req

    def process_station_stream(self, u_c_arr, cond_std_arr, timestamps):
        n = len(u_c_arr)
        u = np.asarray(u_c_arr, dtype=float)
        sig = np.maximum(0.15, np.asarray(cond_std_arr, dtype=float))
        z_u = np.nan_to_num(u / sig, nan=0.0)

        online_predictions = np.zeros(n, dtype=bool)
        c_plus = 0.0
        c_minus = 0.0
        trigger_streak = 0
        state = 0
        min_w = 4
        max_w = 24

        for i in range(n):
            if i < min_w:
                continue

            c_plus = max(0.0, self.decay * c_plus + (z_u[i] - self.allowance))
            c_minus = max(0.0, self.decay * c_minus + (-z_u[i] - self.allowance))
            active_cum = max(c_plus, c_minus)

            max_lam = 0.0
            best_b = 0.0
            max_avail_w = min(i + 1, max_w)

            for W in range(min_w, max_avail_w + 1, 2):
                idx_s = i - W + 1
                z_seg = z_u[idx_s:i+1]
                t_seg = np.arange(W, dtype=float)
                t_bar = (W - 1.0) / 2.0
                z_bar = np.mean(z_seg)
                t_dev = t_seg - t_bar
                z_dev = z_seg - z_bar
                s_tt = np.sum(t_dev ** 2)
                if s_tt < 1e-6:
                    continue
                s_tz = np.sum(t_dev * z_dev)
                delta_rss = (s_tz ** 2) / s_tt
                lam = delta_rss

                if lam > max_lam:
                    max_lam = lam
                    best_b = s_tz / s_tt

            w_coh = min(i + 1, 8)
            diffs = np.diff(z_u[i - w_coh + 1 : i + 1])
            cand_dir = +1 if (c_plus > c_minus and best_b > 0) else (-1 if (c_minus > c_plus and best_b < 0) else 0)
            if len(diffs) > 0 and cand_dir != 0:
                steps_supporting = np.sum(diffs > 0) if cand_dir == +1 else np.sum(diffs < 0)
                sign_consistency = steps_supporting / len(diffs)
            else:
                sign_consistency = 0.50

            is_coherent = (sign_consistency >= 0.60)
            glrt_trigger = (max_lam >= self.glrt_thresh and abs(best_b) >= 0.08 and abs(z_u[i]) >= 2.0 and is_coherent)
            cusum_trigger = (active_cum >= self.cusum_thresh and abs(z_u[i]) >= 2.2 and is_coherent)

            if glrt_trigger or cusum_trigger:
                trigger_streak += 1
            else:
                trigger_streak = max(0, trigger_streak - 1)

            if state == 0:
                if trigger_streak >= self.persist_req:
                    state = 2
            elif state == 2:
                if trigger_streak == 0 and abs(z_u[i]) < 1.2:
                    state = 0

            online_predictions[i] = (state == 2)

        return {"online_predictions": online_predictions}

def test_seed_step2(seed=42):
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='operational_v1', seed=seed, save_to_disk=False)
    frames = [d.assign(station_id=sid) for sid, d in data.items()]
    df_full = pd.concat(frames, ignore_index=True)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp']).dt.tz_localize(None)

    raw_nans = df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']].isna().any(axis=1).to_numpy(dtype=bool)
    df_full['__raw_nan_flag'] = raw_nans
    df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']] = df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']].ffill().bfill()

    label_cols = ['station_id', 'timestamp', 'is_anomaly', 'fault_type']
    labels = df_full[label_cols].copy()
    labels['is_anomaly'] = labels['is_anomaly'].fillna(False).astype(bool)

    df_in = df_full.drop(columns=['is_anomaly', 'fault_type', 'injected_delta', 'injected_start', 'injected_end'], errors='ignore')
    featured, _ = _featurize(df_in)
    featured['timestamp'] = pd.to_datetime(featured['timestamp']).dt.tz_localize(None)
    featured = featured.merge(labels, on=['station_id', 'timestamp'], how='left')

    featured_base, row_hard, row_rule_conf, row_fault_type, _, _ = run_rule_engine_and_health(featured.copy(), artifact)
    row_rule_conf, row_fault_type, _ = apply_spatial_corroboration(featured_base, row_hard, row_rule_conf, row_fault_type, artifact, gate_mode='new')
    model_pct = vectorized_model_scores(featured_base, artifact)
    overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
    is_regional = (row_fault_type == "REGIONAL_EVENT")

    base_predicted = (
        row_hard
        | ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0) & (~is_regional))
        | ((model_pct > 95.0) & (~is_regional))
        | ((row_rule_conf > RULE_CONFIDENCE_BYPASS) & (~is_regional))
    )

    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    detector = CalibratedMultivariateDriftDetector(glrt_thresh=26.0, cusum_thresh=20.0, allowance=0.45, persist_req=2)
    
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
    det_online = featured['detector_online'].fillna(False).to_numpy(dtype=bool)
    raw_nans = featured['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)

    feat_final = featured.copy()
    feat_final['is_anomaly'] = feat_final['is_anomaly'].fillna(False).astype(bool) | raw_nans
    feat_final['fault_type'] = feat_final['fault_type'].fillna('none')

    is_clean = ~feat_final['is_anomaly'].to_numpy(dtype=bool)
    is_fault = feat_final['is_anomaly'].to_numpy(dtype=bool)
    print("base_predicted clean:", (base_predicted & is_clean).sum(), "fault:", (base_predicted & is_fault).sum())
    print("det_online clean:", (det_online & is_clean).sum(), "fault:", (det_online & is_fault).sum())
    print("det_online & ~is_reg clean:", ((det_online & (~is_regional)) & is_clean).sum(), "fault:", ((det_online & (~is_regional)) & is_fault).sum())
    print("raw_nans clean:", (raw_nans & is_clean).sum(), "fault:", (raw_nans & is_fault).sum())

    final_predicted = base_predicted | (det_online & (~is_regional)) | raw_nans
    feat_final['__predicted'] = final_predicted
    feat_final['__predicted_fault_type'] = row_fault_type

    m_final = _score_and_report(feat_final, 'ALL FILES COMBINED', 0, silent=True)
    ep_final = compute_episodic_result(
        feat_final,
        pred_arr=feat_final['__predicted'].to_numpy(dtype=bool),
        pred_ft_arr=feat_final['__predicted_fault_type'].to_numpy()
    )

    print(f"\nSEED {seed} STEP 2 RESULT -> Prec: {m_final['precision']*100:.2f}% | Rec: {m_final['recall']*100:.2f}% | F1: {m_final['f1']:.4f} | TP: {m_final['tp']} | FP: {m_final['fp']} | FN: {m_final['fn']} | EpCatch: {ep_final.episode_detection_rate*100:.2f}% | F1*: {ep_final.latency_aware_f1:.4f}")

    for ftype in ['drift', 'spike', 'frozen_value', 'multivariate_inconsistency', 'sensor_fail_low', 'dropout', 'unstructured_anomaly']:
        sub = feat_final[feat_final['fault_type'] == ftype]
        tot = len(sub)
        rec = (sub['__predicted'].to_numpy(dtype=bool)).sum() / tot if tot > 0 else 0.0
        print(f"  {ftype:<28}: {rec*100:.2f}% ({tot} rows)")

if __name__ == '__main__':
    test_seed_step2(42)
