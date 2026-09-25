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
    ConditionalMultivariateJointModel, STATION_TO_CLUSTER
)

# ==============================================================================
# TIER 1: HIGH-CONFIDENCE DIRECT PHYSICAL RULES
# ==============================================================================
PHYSICAL_SPIKE_LIMITS = {
    "temperature_c": 4.8,  # Minimum hardware jump exceeding diurnal solar warming
    "pressure_hpa": 6.0,   # Minimum hardware jump exceeding synoptic changes
    "humidity_pct": 18.0,  # Minimum hardware jump exceeding rapid desaturation
}

def evaluate_tier1_physical(val, prev_val, col, val_history_prefix, faillow_streak):
    """
    Tier 1: Evaluates direct physical failure signatures:
    - Sensor Fail-Low (ground short / hardware rail)
    - Missing / Null Telemetry (Dropout)
    - Physical Bounds Breach
    - High-SNR Transducer Step Spike
    - Transducer Stagnation (Frozen Value)
    """
    prefix = "temp" if col == "temperature_c" else ("pressure" if col == "pressure_hpa" else "humidity")
    dropout = np.isnan(val)
    low, high = PHYSICAL_BOUNDS[col]
    phys_viol = not dropout and (val < low or val > high)
    
    # Fail-low
    is_below = not dropout and val <= (0.0 if prefix == "temp" else (880.0 if prefix == "pressure" else 5.0))
    faillow_confirmed = (faillow_streak >= 2)

    # Frozen: 4-step stagnation
    eps_freeze = 0.06 if prefix in ("temp", "pressure") else 0.12
    if len(val_history_prefix) >= 4 and not dropout:
        h_arr = np.array(val_history_prefix)
        frozen = (np.max(h_arr) - np.min(h_arr) <= eps_freeze)
    else:
        frozen = False

    # Spike: Physical transducer step difference
    step_diff = abs(val - prev_val) if (prev_val is not None and not np.isnan(prev_val) and not dropout) else 0.0
    spike_thresh = PHYSICAL_SPIKE_LIMITS[col]
    spike = (step_diff >= spike_thresh)

    evidence = []
    if dropout:
        evidence.append(("dropout", 100.0, True))
    if phys_viol:
        evidence.append(("physical_bounds", 100.0, True))
    if faillow_confirmed:
        evidence.append(("sensor_fail_low", 99.0, True))
    if frozen:
        evidence.append(("frozen_value", 92.0, False))
    if spike:
        evidence.append(("spike", 95.0, False))

    if evidence:
        ft, conf, is_hard = max(evidence, key=lambda x: x[1])
        return ft, conf, is_hard
    return "none", 0.0, False

# ==============================================================================
# TIER 2: CONTEXTUAL / MULTIVARIATE DETECTION LAYER
# ==============================================================================
def evaluate_tier2_multivariate(temp_dev, humidity_dev, pressure_dev, vapor_dev, mv_streak, model_score):
    """
    Tier 2: Evaluates psychrometric / thermodynamic dislocation
    grounded in Clausius-Clapeyron relation with ML model corroboration.
    """
    mv_level = (
        not np.isnan(temp_dev) and not np.isnan(humidity_dev)
        and abs(temp_dev) >= 0.65 and abs(humidity_dev) >= 1.60
        and (temp_dev * humidity_dev > 0)
        and abs(pressure_dev) < 1.50
    )
    mv_phys = (not np.isnan(vapor_dev) and abs(vapor_dev) > 18.0)
    mv_single = mv_level or mv_phys

    if mv_streak >= 2:
        return "multivariate_inconsistency", 92.0
    elif mv_single:
        # Confidence-aware fusion with ML model evidence
        w_phys = 40.0
        w_model = 0.50 * model_score
        score_mv = w_phys + w_model
        return "multivariate_inconsistency", score_mv
    return "none", 0.0

# ==============================================================================
# TIER 3: CAUSAL TEMPORAL DRIFT DETECTOR
# ==============================================================================
class Tier3DriftDetector:
    def __init__(
        self,
        glrt_thresh: float = 28.0,
        cusum_thresh: float = 22.0,
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

            # 1. CUSUM
            c_plus = max(0.0, c_plus + val - self.allowance)
            c_minus = max(0.0, c_minus - val - self.allowance)
            c_plus_arr[i] = c_plus
            c_minus_arr[i] = c_minus

            # 2. Multi-window Linear Ramp GLRT
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

            # 3. Trajectory Sign Coherence
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
                # Causal Exit: Return to baseline innovation
                if untriggered_streak >= 2 and (abs(z_u[i]) < 1.6 or max_lam < 14.0):
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

def run_tiered_detection(seed=42):
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    detector = Tier3DriftDetector(glrt_thresh=28.0, cusum_thresh=22.0, allowance=0.45, persist_req=2)
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
            'tier3_drift_online': res_det['online_predictions']
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

    # ML Model scores
    model_pct = vectorized_model_scores(featured, artifact)
    featured['model_score'] = model_pct

    # Run Tier 1 and Tier 2 over stations
    df = featured.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    n = len(df)
    
    tier1_hard = np.zeros(n, dtype=bool)
    tier1_conf = np.zeros(n, dtype=float)
    tier1_ft = np.full(n, "none", dtype=object)

    tier2_conf = np.zeros(n, dtype=float)
    tier2_ft = np.full(n, "none", dtype=object)

    prefixes = [("temperature_c", "temp"), ("pressure_hpa", "pressure"), ("humidity_pct", "humidity")]

    for station_id, g in df.groupby("station_id", sort=False):
        positions = g.index.to_numpy()
        m = len(g)
        
        raw = {col: g[col].to_numpy(dtype=float) for col, _ in prefixes}
        dev_col = {p: g[f"{p}_deviation"].to_numpy(dtype=float) for _, p in prefixes}
        vapor_dev_col = g["vapor_pressure_consistency_dev"].to_numpy(dtype=float) if "vapor_pressure_consistency_dev" in g.columns else np.zeros(m)
        model_scores = g["model_score"].to_numpy(dtype=float)

        val_history = {p: deque(maxlen=4) for _, p in prefixes}
        faillow_streaks = {p: 0 for _, p in prefixes}
        mv_streak = 0

        for i in range(m):
            pos = positions[i]
            temp_dev = dev_col["temp"][i]
            humidity_dev = dev_col["humidity"][i]
            pressure_dev = dev_col["pressure"][i]
            vapor_dev = vapor_dev_col[i]
            m_score = model_scores[i]

            # 1. Update Tier 1 state & evaluate
            t1_best_conf = 0.0
            t1_best_ft = "none"
            t1_is_hard = False

            for col, prefix in prefixes:
                val = raw[col][i]
                dropout = np.isnan(val)
                val_history[prefix].append(val)
                prev_v = val_history[prefix][-2] if len(val_history[prefix]) >= 2 else val

                is_below = not dropout and val <= (0.0 if prefix == "temp" else (880.0 if prefix == "pressure" else 5.0))
                faillow_streaks[prefix] = (faillow_streaks[prefix] + 1) if is_below else 0

                ft_p, conf_p, hard_p = evaluate_tier1_physical(val, prev_v, col, val_history[prefix], faillow_streaks[prefix])
                if conf_p > t1_best_conf:
                    t1_best_conf = conf_p
                    t1_best_ft = ft_p
                t1_is_hard = t1_is_hard or hard_p

            tier1_hard[pos] = t1_is_hard
            tier1_conf[pos] = t1_best_conf
            tier1_ft[pos] = t1_best_ft

            # 2. Update Tier 2 Multivariate state & evaluate
            mv_level = (
                not np.isnan(temp_dev) and not np.isnan(humidity_dev)
                and abs(temp_dev) >= 0.65 and abs(humidity_dev) >= 1.60
                and (temp_dev * humidity_dev > 0)
                and abs(pressure_dev) < 1.50
            )
            mv_phys = (not np.isnan(vapor_dev) and abs(vapor_dev) > 18.0)
            mv_single = mv_level or mv_phys
            mv_streak = (mv_streak + 1) if mv_single else 0

            ft_mv, conf_mv = evaluate_tier2_multivariate(temp_dev, humidity_dev, pressure_dev, vapor_dev, mv_streak, m_score)
            tier2_conf[pos] = conf_mv
            tier2_ft[pos] = ft_mv

    featured['tier1_hard'] = tier1_hard
    featured['tier1_conf'] = tier1_conf
    featured['tier1_ft'] = tier1_ft
    featured['tier2_conf'] = tier2_conf
    featured['tier2_ft'] = tier2_ft

    # Fusion across Tiers
    # Tier 1 Bypass
    t1_pred = (tier1_hard | (tier1_conf > 90.0))

    # Tier 2 Fusion (MV confidence > 50 or Model Alone > 88)
    t2_pred = ((tier2_conf > 50.0) | (featured['model_score'] > 88.0))

    featured = featured.merge(all_det_flags, on=['station_id', 'timestamp'], how='left')
    t3_drift_pred = featured['tier3_drift_online'].fillna(False).to_numpy(dtype=bool)
    raw_nans_arr = featured['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)

    # Combined Tiered Decision
    final_predicted = t1_pred | t2_pred | t3_drift_pred | raw_nans_arr

    # Predicted Fault Type Attribution
    final_ft = np.full(n, "none", dtype=object)
    for i in range(n):
        if tier1_hard[i] or tier1_conf[i] > 90.0:
            final_ft[i] = tier1_ft[i]
        elif t3_drift_pred[i]:
            final_ft[i] = "drift"
        elif tier2_conf[i] > 50.0:
            final_ft[i] = "multivariate_inconsistency"
        elif featured['model_score'].iloc[i] > 88.0:
            final_ft[i] = "unstructured_anomaly"
        elif raw_nans_arr[i]:
            final_ft[i] = "dropout"

    feat_final = featured.copy()
    feat_final = feat_final.merge(labels, on=['station_id', 'timestamp'], how='left')
    feat_final['is_anomaly'] = feat_final['is_anomaly'].fillna(False).astype(bool) | raw_nans_arr
    feat_final['fault_type'] = feat_final['fault_type'].fillna('none')
    feat_final['__predicted'] = final_predicted
    feat_final['__predicted_fault_type'] = final_ft

    m_final = _score_and_report(feat_final, 'ALL FILES COMBINED', 0, silent=True)
    ep_final = compute_episodic_result(
        feat_final,
        pred_arr=feat_final['__predicted'].to_numpy(dtype=bool),
        pred_ft_arr=feat_final['__predicted_fault_type'].to_numpy()
    )

    print("\n--- TIERED ARCHITECTURE RESULTS ON SEED 42 ---")
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
    run_tiered_detection(seed=42)
