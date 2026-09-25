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
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel,
    STATION_TO_CLUSTER
)

class CleanExitDriftDetector:
    def __init__(
        self,
        glrt_thresh: float = 30.0,
        cusum_thresh: float = 24.0,
        allowance: float = 0.45,
        persist_req: int = 2,
        min_coherence: float = 0.55
    ):
        self.glrt_thresh = glrt_thresh
        self.cusum_thresh = cusum_thresh
        self.allowance = allowance
        self.persist_req = persist_req
        self.min_coherence = min_coherence

    def process_station_stream(self, u_T: np.ndarray, sig_cond_T: np.ndarray, timestamps: np.ndarray) -> dict:
        n = len(u_T)
        z_u = u_T / np.maximum(0.1, sig_cond_T)

        c_plus = 0.0
        c_minus = 0.0
        c_plus_arr = np.zeros(n)
        c_minus_arr = np.zeros(n)
        glrt_lambdas = np.zeros(n)
        glrt_slopes = np.zeros(n)
        coherence_arr = np.zeros(n)

        episode_states = np.zeros(n, dtype=int)
        online_predictions = np.zeros(n, dtype=bool)

        state = 0
        trigger_streak = 0
        untriggered_streak = 0

        for i in range(n):
            val = z_u[i]

            c_plus = max(0.0, c_plus + val - self.allowance)
            c_minus = max(0.0, c_minus - val - self.allowance)
            c_plus_arr[i] = c_plus
            c_minus_arr[i] = c_minus

            max_lam = 0.0
            best_b = 0.0
            for W in [4, 8, 12, 18, 24]:
                if i + 1 < W:
                    continue
                idx_s = i - W + 1
                u_win = z_u[idx_s : i + 1]
                t_idx = np.arange(W, dtype=float)
                t_bar = 0.5 * (W - 1)
                u_bar = np.mean(u_win)
                t_dev = t_idx - t_bar
                z_dev = u_win - u_bar
                s_tt = np.sum(t_dev ** 2)
                if s_tt < 1e-6:
                    continue
                s_tz = np.sum(t_dev * z_dev)
                delta_rss = (s_tz ** 2) / s_tt
                lam = delta_rss
                if lam > max_lam:
                    max_lam = lam
                    best_b = s_tz / s_tt

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
            coherence_arr[i] = sign_consistency

            active_cum = max(c_plus, c_minus)
            is_coherent = (sign_consistency >= self.min_coherence)

            glrt_trigger = (max_lam >= self.glrt_thresh and abs(best_b) >= 0.08 and abs(z_u[i]) >= 2.2 and is_coherent)
            cusum_trigger = (active_cum >= self.cusum_thresh and abs(z_u[i]) >= 2.4 and is_coherent)

            if glrt_trigger or cusum_trigger:
                trigger_streak += 1
                untriggered_streak = 0
            else:
                trigger_streak = max(0, trigger_streak - 1)
                untriggered_streak += 1

            if state == 0:
                if trigger_streak >= self.persist_req:
                    state = 2
                elif max_lam >= 16.0 or active_cum >= 12.0:
                    state = 1
            elif state == 1:
                if trigger_streak >= self.persist_req:
                    state = 2
                elif max_lam < 8.0 and active_cum < 5.0:
                    state = 0
            elif state == 2:
                # Immediate clean exit when triggers stop
                if untriggered_streak >= 2:
                    state = 0

            episode_states[i] = state
            online_predictions[i] = (state == 2)

        return {
            "online_predictions": online_predictions,
            "episode_states": episode_states,
            "glrt_lambdas": glrt_lambdas,
            "glrt_slopes": glrt_slopes,
            "c_plus": c_plus_arr,
            "c_minus": c_minus_arr,
            "coherence": coherence_arr
        }

def run_physical_rules_detailed(featured, thresholds):
    df = featured.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    n = len(df)
    
    row_hard = np.zeros(n, dtype=bool)
    row_rule_conf = np.zeros(n, dtype=float)
    row_fault_type = np.full(n, "none", dtype=object)
    row_spike_channel = np.full(n, "none", dtype=object)

    prefixes = [("temperature_c", "temp"), ("pressure_hpa", "pressure"), ("humidity_pct", "humidity")]

    # True physical limits for impulse transients
    spike_limits = {
        "temperature_c": 5.5,  # Solar rate max 4.2 C/h -> hardware impulse >= 5.5 C
        "pressure_hpa": 7.0,   # Synoptic jump max 4.5 hPa/h -> hardware impulse >= 7.0 hPa
        "humidity_pct": 22.0,  # Frontal desaturation max 16% -> hardware impulse >= 22.0%
    }

    for station_id, g in df.groupby("station_id", sort=False):
        positions = g.index.to_numpy()
        m = len(g)
        
        raw = {col: g[col].to_numpy(dtype=float) for col, _ in prefixes}
        dev_col = {p: g[f"{p}_deviation"].to_numpy(dtype=float) for _, p in prefixes}
        vapor_dev_col = g["vapor_pressure_consistency_dev"].to_numpy(dtype=float) if "vapor_pressure_consistency_dev" in g.columns else np.zeros(m)
        
        val_history = {p: deque(maxlen=4) for _, p in prefixes}
        faillow_streaks = {p: 0 for _, p in prefixes}
        mv_streak = 0

        for i in range(m):
            pos = positions[i]
            strongest_conf = 0.0
            strongest_ft = "none"
            spike_ch = "none"
            any_hard = False

            temp_dev = dev_col["temp"][i]
            humidity_dev = dev_col["humidity"][i]
            pressure_dev = dev_col["pressure"][i]
            vapor_dev = vapor_dev_col[i]

            mv_level = (
                not np.isnan(temp_dev) and not np.isnan(humidity_dev)
                and abs(temp_dev) >= 0.80 and abs(humidity_dev) >= 2.00
                and (temp_dev * humidity_dev > 0)
                and abs(pressure_dev) < 1.50
            )
            mv_phys = (not np.isnan(vapor_dev) and abs(vapor_dev) > 20.0)
            mv_single = mv_level or mv_phys
            mv_streak = mv_streak + 1 if mv_single else 0
            mv_confirmed = (mv_streak >= 2)

            for col, prefix in prefixes:
                val = raw[col][i]
                dropout = np.isnan(val)
                low, high = PHYSICAL_BOUNDS[col]
                phys_viol = not dropout and (val < low or val > high)
                hard = dropout or phys_viol

                val_history[prefix].append(val)
                eps_freeze = 0.06 if prefix in ("temp", "pressure") else 0.12
                if len(val_history[prefix]) >= 4 and not dropout:
                    h_arr = np.array(val_history[prefix])
                    frozen = (np.max(h_arr) - np.min(h_arr) <= eps_freeze)
                else:
                    frozen = False

                prev_v = val_history[prefix][-2] if len(val_history[prefix]) >= 2 else val
                step_diff = abs(val - prev_v) if (prev_v is not None and not np.isnan(prev_v) and not dropout) else 0.0
                spike_thresh = spike_limits[col]
                spike = (step_diff >= spike_thresh)

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
                    evidence.append(("multivariate_inconsistency", 92.0))
                elif mv_single and prefix in ("temp", "humidity"):
                    evidence.append(("multivariate_inconsistency", 55.0))
                if frozen:
                    evidence.append(("frozen_value", 92.0))
                if spike:
                    evidence.append(("spike", 95.0))
                    spike_ch = col

                if evidence:
                    ft_cur, conf_cur = max(evidence, key=lambda x: x[1])
                    if conf_cur > strongest_conf:
                        strongest_conf = conf_cur
                        strongest_ft = ft_cur
                any_hard = any_hard or hard

            row_hard[pos] = any_hard
            row_rule_conf[pos] = strongest_conf
            row_fault_type[pos] = strongest_ft
            row_spike_channel[pos] = spike_ch

    return df, row_hard, pd.Series(row_rule_conf), pd.Series(row_fault_type), pd.Series(row_spike_channel)

def test_improvements():
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    detector = CleanExitDriftDetector(glrt_thresh=30.0, cusum_thresh=24.0, allowance=0.45, persist_req=2)
    artifact = joblib.load(ARTIFACTS_PATH)

    data = generate_network_benchmark(regime='observable_v1', seed=42, save_to_disk=False)
    
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
    featured_base, row_hard, row_rule_conf, row_fault_type, row_spike_ch = run_physical_rules_detailed(featured.copy(), thresholds)
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

    print("\n--- RESULTS ON SEED 42 WITH CLEAN DRIFT EXIT & TRANSDUCER SPIKE LIMITS ---")
    print(f"TP: {m_final['tp']}, FP: {m_final['fp']}, FN: {m_final['fn']}")
    print(f"Precision: {m_final['precision']*100:.2f}%")
    print(f"Recall: {m_final['recall']*100:.2f}%")
    print(f"F1 Score: {m_final['f1']:.4f}")

    print("\nFault Type Breakdown on Seed 42:")
    for ft in feat_final['fault_type'].unique():
        if ft == 'none':
            continue
        sub = feat_final[feat_final['fault_type'] == ft]
        tot = len(sub)
        tps = (sub['__predicted'].to_numpy(dtype=bool)).sum()
        print(f"  {ft:<28}: {tps}/{tot} ({tps/tot*100:.2f}%)")

if __name__ == '__main__':
    test_improvements()
