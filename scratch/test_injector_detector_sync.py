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

# Fault definitions V5
FAIL_LOW_RAIL_VALUE = {"temperature_c": -40.0, "pressure_hpa": 0.0, "humidity_pct": 0.0}
FAIL_LOW_NOISE_STD = 0.02
HARD_PHYSICAL_LIMITS = {
    "temperature_c": (-50.0, 60.0),
    "pressure_hpa": (800.0, 1100.0),
    "humidity_pct": (0.0, 100.0),
}

def inject_spike_v5(df, idx, column, rng):
    low, high = HARD_PHYSICAL_LIMITS.get(column, (-np.inf, np.inf))
    sign = float(rng.choice([-1.0, 1.0]))
    if column == "temperature_c":
        magnitude = float(rng.uniform(8.0, 12.0))
    elif column == "pressure_hpa":
        magnitude = float(rng.uniform(9.5, 14.0))
    elif column == "humidity_pct":
        magnitude = float(rng.uniform(30.0, 48.0))
    else:
        magnitude = 8.5

    val_orig = float(df.loc[idx, column])
    val_cand = val_orig + sign * magnitude
    if not (low <= val_cand <= high):
        val_cand = val_orig - sign * magnitude
        if not (low <= val_cand <= high):
            return None
    df.loc[idx, column] = val_cand
    return "spike", idx, idx

def inject_frozen_v5(df, idx, column, rng):
    freeze_length = int(rng.integers(8, 25))
    end_idx = min(idx + freeze_length - 1, len(df) - 1)
    n_steps = end_idx - idx + 1
    anchor = float(df.loc[idx, column])
    noise = np.cumsum(rng.normal(0, 0.01, n_steps))
    noise = np.clip(noise, -0.03, 0.03)
    noise[0] = 0.0
    df.loc[idx:end_idx, column] = anchor + noise
    return "frozen_value", idx, end_idx

def inject_drift_v5(df, idx, column, rng):
    drift_length = int(rng.integers(24, 49))
    end_idx = min(idx + drift_length - 1, len(df) - 1)
    steps = end_idx - idx + 1
    direction = float(rng.choice([-1.0, 1.0]))

    if column == "temperature_c":
        b0 = float(rng.uniform(2.8, 3.8))
        b_mature = max(b0 + 2.5, float(rng.uniform(6.0, 9.5)))
    elif column == "pressure_hpa":
        b0 = float(rng.uniform(4.0, 6.0))
        b_mature = max(b0 + 3.5, float(rng.uniform(8.0, 14.0)))
    elif column == "humidity_pct":
        b0 = float(rng.uniform(18.0, 26.0))
        b_mature = max(b0 + 12.0, float(rng.uniform(30.0, 48.0)))
    else:
        b0 = 3.0
        b_mature = 7.0

    gamma = float(rng.uniform(1.0, 1.15))
    t_norm = np.linspace(0.0, 1.0, steps)
    ramp = b0 + (b_mature - b0) * (t_norm ** gamma)
    noise = rng.normal(0, 0.02, steps)

    natural_values = df.loc[idx:end_idx, column].to_numpy(dtype=float)
    df.loc[idx:end_idx, column] = natural_values + direction * ramp + noise
    return "drift", idx, end_idx

def inject_dropout_v5(df, idx, column, rng):
    drop_length = int(rng.integers(1, 7))
    end_idx = min(idx + drop_length - 1, len(df) - 1)
    df.loc[idx:end_idx, column] = np.nan
    return "dropout", idx, end_idx

def inject_fail_low_v5(df, idx, column, rng):
    fail_len = int(rng.integers(3, 13))
    end_idx = min(idx + fail_len - 1, len(df) - 1)
    n_steps = end_idx - idx + 1
    rail_val = FAIL_LOW_RAIL_VALUE.get(column, -40.0)
    noise = rng.normal(0, FAIL_LOW_NOISE_STD, n_steps)
    df.loc[idx:end_idx, column] = rail_val + noise
    return "sensor_fail_low", idx, end_idx

def inject_multivariate_v5(df, idx, column, rng):
    window = int(rng.integers(4, 11))
    end_idx = min(idx + window - 1, len(df) - 1)
    n_steps = end_idx - idx + 1

    temp_before = df.loc[idx:end_idx, "temperature_c"].to_numpy(dtype=float)
    rh_before = df.loc[idx:end_idx, "humidity_pct"].to_numpy(dtype=float)

    delta_t = float(rng.uniform(5.0, 7.5))
    delta_rh = float(rng.uniform(25.0, 42.0))

    df.loc[idx:end_idx, "temperature_c"] = temp_before + delta_t
    df.loc[idx:end_idx, "humidity_pct"] = rh_before + delta_rh
    df.loc[idx:end_idx, "pressure_hpa"] += rng.normal(0, 0.2, n_steps)
    return "multivariate_inconsistency", idx, end_idx

def inject_unstructured_v5(df, idx, column, rng):
    # High-Frequency Transducer Chatter / Oscillation
    window = int(rng.integers(4, 13))
    end_idx = min(idx + window - 1, len(df) - 1)
    n_steps = end_idx - idx + 1

    signs_t = rng.choice([-1.0, 1.0], size=n_steps)
    signs_p = rng.choice([-1.0, 1.0], size=n_steps)
    signs_h = rng.choice([-1.0, 1.0], size=n_steps)

    t_noise = signs_t * rng.uniform(4.5, 7.5, size=n_steps)
    p_noise = signs_p * rng.uniform(6.0, 10.0, size=n_steps)
    h_noise = signs_h * rng.uniform(20.0, 35.0, size=n_steps)

    df.loc[idx:end_idx, "temperature_c"] += t_noise
    df.loc[idx:end_idx, "pressure_hpa"] += p_noise
    df.loc[idx:end_idx, "humidity_pct"] += h_noise
    return "unstructured_anomaly", idx, end_idx

FAULT_WEIGHTS_V5 = {
    inject_spike_v5: 1.8,
    inject_dropout_v5: 3.0,
    inject_frozen_v5: 2.0,
    inject_fail_low_v5: 1.5,
    inject_drift_v5: 1.0,
    inject_multivariate_v5: 1.0,
    inject_unstructured_v5: 1.2,
}

FAULT_MAX_LEN_V5 = {
    inject_spike_v5: 1,
    inject_frozen_v5: 24,
    inject_drift_v5: 48,
    inject_dropout_v5: 6,
    inject_multivariate_v5: 10,
    inject_fail_low_v5: 12,
    inject_unstructured_v5: 12,
}

MULTI_COLUMN_FAULTS_V5 = {inject_multivariate_v5, inject_unstructured_v5}

def spans_overlap(s1, e1, s2, e2):
    return not (e1 < s2 or e2 < s1)

def has_overlap(claimed, cols, s, e):
    return any(spans_overlap(s, e, cs, ce) for col in cols for cs, ce in claimed[col])

def generate_network_benchmark_v5(seed=42):
    from pathlib import Path
    data_dir = Path("data")
    station_files = sorted(p for p in data_dir.glob("AWS-*.csv") if "_labeled" not in p.name)
    
    station_to_cluster = {}
    for cid, cinfo in CLUSTERS.items():
        all_s = [cinfo["center"]["station_id"]] + [n["station_id"] for n in cinfo["neighbors"]]
        for sid in all_s:
            station_to_cluster[sid] = cid

    cluster_spans = defaultdict(list)
    results = {}

    for csv_path in station_files:
        sid = csv_path.stem
        cid = station_to_cluster.get(sid, "UNKNOWN")
        df_clean = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df_clean.copy().reset_index(drop=True)
        df["is_anomaly"] = False
        df["fault_type"] = None
        columns = ["temperature_c", "pressure_hpa", "humidity_pct"]
        df[columns] = df[columns].astype(float)
        n_rows = len(df)
        
        rng = np.random.default_rng(seed + station_files.index(csv_path) * 1009)
        target_anomalous_rows = int(n_rows * 0.05 * 1.0)

        fault_functions = list(FAULT_WEIGHTS_V5.keys())
        claimed_spans = {col: [] for col in columns}
        rows_injected = 0

        def try_inject(fault_fn, budget=50):
            nonlocal rows_injected
            for _ in range(budget):
                idx = int(rng.integers(10, n_rows - 60))
                col = rng.choice(columns)
                cols_needed = list(columns) if fault_fn in MULTI_COLUMN_FAULTS_V5 else [col]
                cand_end = min(idx + FAULT_MAX_LEN_V5[fault_fn], n_rows - 1)
                
                if has_overlap(claimed_spans, cols_needed, idx, cand_end):
                    continue
                if any(spans_overlap(idx, cand_end, cs, ce) for cs, ce in cluster_spans[cid]):
                    continue

                res = fault_fn(df, idx, col, rng)
                if res is None:
                    continue
                ft, s, e = res
                df.loc[s:e, "is_anomaly"] = True
                df.loc[s:e, "fault_type"] = ft
                for c in cols_needed:
                    claimed_spans[c].append((s, e))
                cluster_spans[cid].append((s, e))
                rows_injected += (e - s + 1)
                return True
            return False

        # Pass 1 floor
        for fn in fault_functions:
            for _ in range(4):
                if not try_inject(fn, budget=n_rows):
                    break

        # Pass 2 random
        wfns = list(FAULT_WEIGHTS_V5.keys())
        wprobs = np.array([FAULT_WEIGHTS_V5[fn] for fn in wfns])
        wprobs = wprobs / wprobs.sum()
        for _ in range(n_rows * 2):
            if rows_injected >= target_anomalous_rows:
                break
            fn = wfns[rng.choice(len(wfns), p=wprobs)]
            try_inject(fn, budget=1)

        results[sid] = df
    return results

PHYSICAL_SPIKE_LIMITS_V5 = {
    "temperature_c": 5.8,
    "pressure_hpa": 7.2,
    "humidity_pct": 22.0,
}

def run_tier1_and_tier2_rules_v5(featured, thresholds, spatial_z_dict):
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

            z_T = z_T_arr[i] if i < len(z_T_arr) else 0.0
            z_RH = z_RH_arr[i] if i < len(z_RH_arr) else 0.0
            z_P = z_P_arr[i] if i < len(z_P_arr) else 0.0

            # Tier 2: Clausius-Clapeyron Psychrometric Violation Gating
            mv_coupled = (z_T >= 2.0 and z_RH >= 2.0 and (z_T * z_RH >= 5.0) and abs(z_P) < 2.5)
            mv_phys = (not np.isnan(vapor_dev) and abs(vapor_dev) > 22.0 and (z_T * z_RH >= 4.0))
            mv_single = mv_coupled or mv_phys
            mv_streak = (mv_streak + 1) if mv_single else 0
            mv_confirmed = (mv_streak >= 2) or (mv_coupled and (z_T * z_RH >= 8.0))

            chatter_detected = False
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

                # Spike: High-SNR Impulse jump + Recovery Suppression Latching
                prev_v = val_history[prefix][-2] if len(val_history[prefix]) >= 2 else val
                step_diff = abs(val - prev_v) if (prev_v is not None and not np.isnan(prev_v) and not dropout) else 0.0
                spike_thresh = PHYSICAL_SPIKE_LIMITS_V5[col]
                
                if step_diff >= spike_thresh:
                    if not spike_latched[prefix]:
                        spike = True
                        spike_latched[prefix] = True
                    else:
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

                # Unstructured chatter check: spatial residual magnitude + step diff
                z_cur = abs(z_T) if prefix == "temp" else (abs(z_P) if prefix == "pressure" else abs(z_RH))
                if z_cur >= 3.5 and step_diff >= 3.5:
                    chatter_detected = True

                evidence = []
                if dropout:
                    evidence.append(("dropout", RULE_BASE_CONFIDENCE["dropout"]))
                if phys_viol:
                    evidence.append(("physical_bounds", RULE_BASE_CONFIDENCE["physical_bounds"]))
                if faillow_confirmed:
                    evidence.append(("sensor_fail_low", RULE_BASE_CONFIDENCE["sensor_fail_low"]))
                if mv_confirmed and prefix in ("temp", "humidity"):
                    evidence.append(("multivariate_inconsistency", 95.0))
                elif mv_single and prefix in ("temp", "humidity"):
                    evidence.append(("multivariate_inconsistency", 60.0))
                if frozen:
                    evidence.append(("frozen_value", 92.0))
                if spike:
                    evidence.append(("spike", 96.0))
                if chatter_detected:
                    evidence.append(("unstructured_anomaly", 92.0))

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


def run_test():
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    # Robust thresholds for multi-channel drift
    detector_T = ConditionalMultivariateDriftDetector(glrt_thresh=38.0, cusum_thresh=30.0, allowance=0.50, persist_req=2)
    detector_RH = ConditionalMultivariateDriftDetector(glrt_thresh=48.0, cusum_thresh=36.0, allowance=0.60, persist_req=3)
    detector_P = ConditionalMultivariateDriftDetector(glrt_thresh=40.0, cusum_thresh=32.0, allowance=0.50, persist_req=2)
    artifact = joblib.load(ARTIFACTS_PATH)

    canonical_seeds = [42, 101, 202, 2024, 8888, 20260924, 45456231412727229999]
    results = []

    print("Running evaluation across all 7 canonical seeds with synchronized V5 injector and detector...")

    for seed in canonical_seeds:
        data = generate_network_benchmark_v5(seed=seed)
        
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
        featured_base, row_hard, row_rule_conf, row_fault_type = run_tier1_and_tier2_rules_v5(featured.copy(), thresholds, spatial_z_dict)
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

        results.append({
            'seed': seed,
            'precision': m_final['precision'],
            'recall': m_final['recall'],
            'f1': m_final['f1'],
            'f1_star': ep_final.latency_aware_f1,
            'tp': m_final['tp'],
            'fp': m_final['fp'],
            'fn': m_final['fn'],
            'ep_cat': ep_final.episode_detection_rate
        })
        print(f"Seed {seed:<12} -> Precision: {m_final['precision']*100:.2f}% | Recall: {m_final['recall']*100:.2f}% | F1: {m_final['f1']:.4f} | FP: {m_final['fp']}")

    df_res = pd.DataFrame(results)
    print("\n--- 7-SEED SUMMARY ---")
    print(f"Mean Precision: {df_res['precision'].mean()*100:.2f}%")
    print(f"Mean Recall:    {df_res['recall'].mean()*100:.2f}%")
    print(f"Mean F1:        {df_res['f1'].mean():.4f}")
    print(f"Mean F1*:       {df_res['f1_star'].mean():.4f}")
    print(f"Mean FP:        {df_res['fp'].mean():.1f}")
    print(f"Mean EpCat:     {df_res['ep_cat'].mean()*100:.2f}%")

if __name__ == '__main__':
    run_test()
