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
from evaluation.episodic_eval import compute_episodic_result
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel,
    ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel,
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

# ==============================================================================
# CALIBRATED DRIFT DETECTOR WITH CLEAN EXIT
# ==============================================================================
class CalibratedMultivariateDriftDetector:
    def __init__(self, glrt_thresh=28.0, cusum_thresh=22.0, min_coherence=0.60, decay=0.88, allowance=0.45, persist_req=2):
        self.glrt_thresh = glrt_thresh
        self.cusum_thresh = cusum_thresh
        self.min_coherence = min_coherence
        self.decay = decay
        self.allowance = allowance
        self.persist_req = persist_req

    def process_station_stream(self, u_c_arr, cond_std_arr, timestamps):
        n = len(u_c_arr)
        u = np.asarray(u_c_arr, dtype=float)
        sig = np.maximum(0.15, np.asarray(cond_std_arr, dtype=float))
        z_u = np.nan_to_num(u / sig, nan=0.0)

        online_predictions = np.zeros(n, dtype=bool)
        episode_states = np.zeros(n, dtype=int)
        forensic_onsets = np.full(n, -1, dtype=int)
        glrt_lambdas = np.zeros(n, dtype=float)
        glrt_slopes = np.zeros(n, dtype=float)

        state = 0  # 0: NORMAL, 1: SUSPECT, 2: CONFIRMED
        c_plus = 0.0
        c_minus = 0.0
        trigger_streak = 0
        untriggered_streak = 0

        min_w = 4
        max_w = 24

        for i in range(n):
            if i < min_w:
                episode_states[i] = 0
                online_predictions[i] = False
                continue

            c_plus = max(0.0, self.decay * c_plus + (z_u[i] - self.allowance))
            c_minus = max(0.0, self.decay * c_minus + (-z_u[i] - self.allowance))

            max_lam = 0.0
            best_b = 0.0
            best_onset = i
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
                    best_onset = idx_s

            glrt_lambdas[i] = max_lam
            glrt_slopes[i] = best_b

            w_coh = min(i + 1, 8)
            diffs = np.diff(z_u[i - w_coh + 1 : i + 1])
            cand_dir = +1 if (c_plus > c_minus and best_b > 0) else (-1 if (c_minus > c_plus and best_b < 0) else 0)
            if len(diffs) > 0 and cand_dir != 0:
                steps_supporting = np.sum(diffs > 0) if cand_dir == +1 else np.sum(diffs < 0)
                sign_consistency = steps_supporting / len(diffs)
            else:
                sign_consistency = 0.50

            active_cum = max(c_plus, c_minus)
            is_coherent = (sign_consistency >= self.min_coherence)

            glrt_trigger = (max_lam >= self.glrt_thresh and abs(best_b) >= 0.08 and abs(z_u[i]) >= 2.0 and is_coherent)
            cusum_trigger = (active_cum >= self.cusum_thresh and abs(z_u[i]) >= 2.2 and is_coherent)

            if glrt_trigger or cusum_trigger:
                trigger_streak += 1
                untriggered_streak = 0
            else:
                trigger_streak = max(0, trigger_streak - 1)
                untriggered_streak += 1

            if state == 0:
                if trigger_streak >= self.persist_req:
                    state = 2
                elif max_lam >= 14.0 or active_cum >= 10.0:
                    state = 1
            elif state == 1:
                if trigger_streak >= self.persist_req:
                    state = 2
                elif max_lam < 8.0 and active_cum < 5.0:
                    state = 0
            elif state == 2:
                # Clean exit when trigger conditions cease
                if untriggered_streak >= 2 and (max_lam < 14.0 or abs(z_u[i]) < 1.8):
                    state = 0

            episode_states[i] = state
            online_predictions[i] = (state == 2)
            forensic_onsets[i] = best_onset if state == 2 else -1

        return {
            "online_predictions": online_predictions,
            "episode_states": episode_states,
            "forensic_onsets": forensic_onsets,
            "glrt_lambdas": glrt_lambdas,
            "glrt_slopes": glrt_slopes
        }

detector = CalibratedMultivariateDriftDetector(glrt_thresh=28.0, cusum_thresh=22.0, allowance=0.45, persist_req=2)

print("Testing Seed 42 with calibrated detector & rules...", flush=True)
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

featured, _ = _featurize(df_in)
featured['timestamp'] = pd.to_datetime(featured['timestamp']).dt.tz_localize(None)

featured_base, row_hard, row_rule_conf, row_fault_type, _, _ = run_rule_engine_and_health(featured.copy(), artifact)
row_rule_conf, row_fault_type, _ = apply_spatial_corroboration(
    featured_base, row_hard, row_rule_conf, row_fault_type, artifact, gate_mode='new'
)
model_pct = vectorized_model_scores(featured_base, artifact)
overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
is_regional = (row_fault_type == "REGIONAL_EVENT")

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
        'detector_online': res_det['online_predictions']
    })
    det_flags_list.append(df_stn)

all_det_flags = pd.concat(det_flags_list, ignore_index=True)
featured = featured.merge(all_det_flags, on=['station_id', 'timestamp'], how='left')
det_online_arr = featured['detector_online'].fillna(False).to_numpy(dtype=bool)
raw_nans = featured['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)

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

m = _score_and_report(feat_final, 'SEED 42 TEST', 0, silent=False)
ep = compute_episodic_result(feat_final, pred_arr=feat_final['__predicted'].to_numpy(dtype=bool), pred_ft_arr=feat_final['__predicted_fault_type'].to_numpy())
print(f"Results: Prec: {m['precision']*100:.2f}%, Rec: {m['recall']*100:.2f}%, F1: {m['f1']:.4f}, FP: {m['fp']}, EpCatch: {ep.episode_detection_rate*100:.2f}%")
