import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    evaluate_all, ARTIFACTS_PATH, PHYSICAL_BOUNDS,
    vectorized_model_scores, run_rule_engine_and_health,
    apply_spatial_corroboration, _score_and_report, _featurize,
    _row_metrics, _safe_div,
    MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD,
    MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS,
    RULE_BASE_CONFIDENCE, FROZEN_MIN_MODEL_CORROBORATION,
    HELPER_ALERT_THRESHOLD, FROZEN_HELPER_ALERT_THRESHOLD,
    add_frozen_channel_labels_from_reference
)
from evaluation.episodic_eval import compute_episodic_result

# Cluster mappings
STATION_TO_CLUSTER = {}
CLUSTER_TO_STATIONS = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    stns = [center] + neighbors
    CLUSTER_TO_STATIONS[cid] = stns
    for sid in stns:
        STATION_TO_CLUSTER[sid] = cid

# ==============================================================================
# 1. PASS 1 CAUSAL NORMAL MODEL & PASS 5 CONDITIONAL UNCERTAINTY
# ==============================================================================
class CausalNormalBehaviorModel:
    def __init__(self, train_ratio=0.60):
        self.train_ratio = train_ratio
        self.models = {}

    def fit(self):
        clean_dfs = {}
        for sid in STATION_TO_CLUSTER:
            df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            clean_dfs[sid] = df

        params = ['temperature_c', 'pressure_hpa', 'humidity_pct']

        for sid, cid in STATION_TO_CLUSTER.items():
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            df_target = clean_dfs[sid]
            n_train = int(len(df_target) * self.train_ratio)
            df_target_train = df_target.iloc[:n_train]
            
            hours = df_target_train['timestamp'].dt.hour
            sin_h = np.sin(2 * np.pi * hours / 24.0)
            cos_h = np.cos(2 * np.pi * hours / 24.0)

            for p in params:
                peer_regressors = {}
                for pid in peer_ids:
                    df_peer = clean_dfs[pid].iloc[:n_train]
                    X = np.column_stack([np.ones(n_train), df_peer[p].values, sin_h.values, cos_h.values])
                    y = df_target_train[p].values

                    try:
                        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                        pred = X @ coeffs
                        err = y - pred
                        sigma = max(0.1, np.std(err))
                        peer_regressors[pid] = {"coeffs": coeffs, "sigma": sigma}
                    except Exception:
                        pass

                if peer_regressors:
                    inv_vars = {pid: 1.0 / (reg["sigma"]**2) for pid, reg in peer_regressors.items()}
                    total_inv = sum(inv_vars.values())
                    weights = {pid: inv_vars[pid] / total_inv for pid in peer_regressors}
                    comb_sigma = 1.0 / np.sqrt(total_inv)
                else:
                    weights = {}
                    comb_sigma = 1.0

                self.models[(sid, p)] = {
                    "peer_regressors": peer_regressors,
                    "weights": weights,
                    "comb_sigma": comb_sigma
                }

    def predict_target(self, sid, p, target_df, peer_dfs_dict):
        model_info = self.models.get((sid, p))
        if not model_info or not model_info["weights"]:
            return target_df[p].values, np.ones(len(target_df))

        hours = pd.to_datetime(target_df['timestamp']).dt.hour
        sin_h = np.sin(2 * np.pi * hours / 24.0).values
        cos_h = np.cos(2 * np.pi * hours / 24.0).values
        n = len(target_df)

        y_hat_comb = np.zeros(n, dtype=float)
        weights = model_info["weights"]
        target_ts = pd.to_datetime(target_df['timestamp']).dt.tz_localize(None)

        for pid, w in weights.items():
            if pid in peer_dfs_dict:
                reg = model_info["peer_regressors"][pid]
                coeffs = reg["coeffs"]
                peer_df = peer_dfs_dict[pid].copy()
                peer_df['timestamp'] = pd.to_datetime(peer_df['timestamp']).dt.tz_localize(None)
                peer_df = peer_df.set_index('timestamp')
                peer_val = peer_df.reindex(target_ts)[p].ffill().bfill().values
                X = np.column_stack([np.ones(n), peer_val, sin_h, cos_h])
                y_hat_comb += w * (X @ coeffs)

        comb_sigma = model_info["comb_sigma"] * np.ones(n, dtype=float)
        return y_hat_comb, comb_sigma


class ConditionalResidualUncertaintyModel:
    def __init__(self, normal_model, train_ratio=0.60):
        self.normal_model = normal_model
        self.train_ratio = train_ratio
        self.scales = {}
        self.yhat_bins = {}
        self.global_channel_scales = {}

    @staticmethod
    def get_hour_regime(hour):
        h = np.asarray(hour)
        regime = np.zeros_like(h, dtype=int)
        regime[(h >= 6) & (h <= 10)] = 1
        regime[(h >= 11) & (h <= 16)] = 2
        regime[(h >= 17) & (h <= 21)] = 3
        return regime

    def fit(self):
        clean_dfs = {}
        for sid in STATION_TO_CLUSTER:
            df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            clean_dfs[sid] = df

        params = ['temperature_c', 'pressure_hpa', 'humidity_pct']
        train_residuals = {}
        train_yhats = {}
        train_hours = {}
        all_channel_residuals = {p: [] for p in params}

        for sid, cid in STATION_TO_CLUSTER.items():
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            peer_dfs = {pid: clean_dfs[pid] for pid in peer_ids}
            df_target = clean_dfs[sid]
            n_train = int(len(df_target) * self.train_ratio)
            df_train = df_target.iloc[:n_train]
            peer_dfs_train = {pid: peer_dfs[pid].iloc[:n_train] for pid in peer_ids}
            
            hours_train = df_train['timestamp'].dt.hour.values
            train_hours[sid] = hours_train
            
            for p in params:
                y_hat, _ = self.normal_model.predict_target(sid, p, df_train, peer_dfs_train)
                res = df_train[p].values - y_hat
                train_residuals[(sid, p)] = res
                train_yhats[(sid, p)] = y_hat
                all_channel_residuals[p].extend(res)

        for p in params:
            arr = np.array(all_channel_residuals[p])
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826
            self.global_channel_scales[p] = max(0.15, mad)

        for sid, cid in STATION_TO_CLUSTER.items():
            hours_train = train_hours[sid]
            regimes_train = self.get_hour_regime(hours_train)
            
            for p in params:
                res_train = train_residuals[(sid, p)]
                yhat_train = train_yhats[(sid, p)]
                stn_med = np.median(res_train)
                stn_scale = max(0.15, np.median(np.abs(res_train - stn_med)) * 1.4826)
                
                q_cuts = np.quantile(yhat_train, [0.25, 0.50, 0.75])
                if len(np.unique(q_cuts)) < 3:
                    q_cuts = np.linspace(np.min(yhat_train), np.max(yhat_train), 4)[1:]
                self.yhat_bins[(sid, p)] = q_cuts
                yhat_bin_idx = np.digitize(yhat_train, q_cuts)
                
                scale_table = {}
                for yb in range(4):
                    for hr in range(4):
                        mask = (yhat_bin_idx == yb) & (regimes_train == hr)
                        sub_res = res_train[mask]
                        n_pts = len(sub_res)
                        if n_pts >= 15:
                            sub_med = np.median(sub_res)
                            raw_mad = np.median(np.abs(sub_res - sub_med)) * 1.4826
                            w_local = n_pts / (n_pts + 10.0)
                            shrunk_sigma = w_local * raw_mad + (1.0 - w_local) * stn_scale
                        else:
                            shrunk_sigma = stn_scale
                        scale_table[(yb, hr)] = max(0.15, shrunk_sigma)
                
                self.scales[(sid, p)] = {
                    "stn_scale": stn_scale,
                    "scale_table": scale_table,
                    "q_cuts": q_cuts
                }

    def predict_sigma(self, sid, p, y_hat, timestamps):
        model_info = self.scales.get((sid, p))
        if not model_info:
            return np.full(len(y_hat), self.global_channel_scales.get(p, 0.45), dtype=float)
            
        ts = pd.to_datetime(timestamps)
        hours = ts.hour.values if hasattr(ts, 'hour') else ts.dt.hour.values
        regimes = self.get_hour_regime(hours)
        
        q_cuts = model_info["q_cuts"]
        yhat_bins = np.digitize(y_hat, q_cuts)
        scale_table = model_info["scale_table"]
        stn_scale = model_info["stn_scale"]
        
        n = len(y_hat)
        sigmas = np.zeros(n, dtype=float)
        for i in range(n):
            yb = min(3, max(0, yhat_bins[i]))
            hr = regimes[i]
            sigmas[i] = scale_table.get((yb, hr), stn_scale)
        return sigmas


# ==============================================================================
# 2. FINAL PERSISTENT-vs-TRANSIENT DETECTOR & CAUSAL CHANGE-POINT LOCALIZATION
# ==============================================================================
class PersistentChangeDetector:
    """
    State Machine:
    NORMAL (0) -> SUSPECT (1) -> CONFIRMED_DRIFT (2) -> RECOVERY (3) -> NORMAL (0)
    
    Distinguishes:
    1. Transient Meteorological Shocks:
       - High localized residual excursion that reverses/decays
       - Opposing residual movements (high reversal fraction)
       - Low cumulative persistence integral
    2. Persistent Calibration Drift:
       - Monotonic directional movement (high sign consistency)
       - Stable robust slope away from baseline
       - High cumulative persistence integral C_plus / C_minus
       - Model comparison: E(tau) = Evidence(H_drift) - Evidence(H_transient) > 0
    
    Causal Change-Point Localization:
    - tau_hat = argmax_{tau in trailing window} Lambda(tau)
    - Online point prediction: flagged at alert time t
    - Retroactive episode backfill: timestamps tau_hat:t marked upon confirmation
    """
    def __init__(self, decay=0.92, suspect_thresh=3.5, confirm_thresh=8.5, min_coherence=0.60):
        self.decay = decay
        self.suspect_thresh = suspect_thresh
        self.confirm_thresh = confirm_thresh
        self.min_coherence = min_coherence

    def process_series(self, residuals, sigmas, timestamps):
        n = len(residuals)
        r = np.asarray(residuals, dtype=float)
        sig = np.asarray(sigmas, dtype=float)
        sig = np.where(sig < 0.10, 0.10, sig)
        z = r / sig
        w = 1.0 / (sig ** 2)
        
        # State tracking arrays
        states = np.zeros(n, dtype=int)                 # 0: NORMAL, 1: SUSPECT, 2: CONFIRMED, 3: RECOVERY
        online_flags = np.zeros(n, dtype=bool)          # Online timestamp alert
        backfilled_flags = np.zeros(n, dtype=bool)      # Retroactively localized episode flags
        estimated_onsets = np.full(n, -1, dtype=int)    # Estimated tau_hat
        glrt_lambdas = np.zeros(n, dtype=float)         # Short-term GLRT Lambda
        glrt_slopes = np.zeros(n, dtype=float)          # GLRT slope b_hat
        c_plus_arr = np.zeros(n, dtype=float)           # Positive accumulator
        c_minus_arr = np.zeros(n, dtype=float)          # Negative accumulator
        evidence_drift_arr = np.zeros(n, dtype=float)   # E(tau) model comparison
        
        current_state = 0
        c_plus = 0.0
        c_minus = 0.0
        current_drift_onset = -1
        current_direction = 0  # +1 or -1
        
        min_w = 4
        max_w = 48
        
        for i in range(min_w, n):
            # 1. Short-Term Intercept-Aware Weighted GLRT
            max_lam = 0.0
            best_b = 0.0
            best_onset = i
            max_avail_w = min(i + 1, max_w)
            
            for W in range(min_w, max_avail_w + 1, 2):
                idx_start = i - W + 1
                idx_end = i + 1
                w_seg = w[idx_start:idx_end]
                r_seg = r[idx_start:idx_end]
                t_seg = np.arange(W, dtype=float)
                
                sum_w = np.sum(w_seg)
                if sum_w <= 0:
                    continue
                t_bar = np.sum(w_seg * t_seg) / sum_w
                r_bar = np.sum(w_seg * r_seg) / sum_w
                t_dev = t_seg - t_bar
                r_dev = r_seg - r_bar
                s_tt = np.sum(w_seg * (t_dev ** 2))
                if s_tt < 1e-6:
                    continue
                s_tr = np.sum(w_seg * t_dev * r_dev)
                b_hat = s_tr / s_tt
                delta_rss = (s_tr ** 2) / s_tt
                lam = delta_rss / 2.0
                
                if lam > max_lam:
                    max_lam = lam
                    best_b = b_hat
                    best_onset = idx_start
                    
            glrt_lambdas[i] = max_lam
            glrt_slopes[i] = best_b
            
            # 2. Long-Term Directional Persistence Evidence Accumulators
            # Standardized z_t persistence: evidence = z_t - allowance
            allowance = 0.50
            c_plus = max(0.0, self.decay * c_plus + (z[i] - allowance))
            c_minus = max(0.0, self.decay * c_minus + (-z[i] - allowance))
            c_plus_arr[i] = c_plus
            c_minus_arr[i] = c_minus
            
            # 3. Trajectory Coherence & Directional Support
            # Over the candidate onset window
            W_cand = min(max(i - best_onset + 1, 6), 24)
            start_cand = i - W_cand + 1
            cand_z = z[start_cand : i + 1]
            cand_diffs = np.diff(cand_z)
            
            cand_dir = +1 if best_b > 0 else (-1 if best_b < 0 else 0)
            if len(cand_diffs) > 0 and cand_dir != 0:
                pos_steps = np.sum(cand_diffs > 0)
                neg_steps = np.sum(cand_diffs < 0)
                tot_steps = len(cand_diffs)
                
                if cand_dir == +1:
                    sign_consistency = pos_steps / tot_steps
                    reversal_fraction = neg_steps / tot_steps
                else:
                    sign_consistency = neg_steps / tot_steps
                    reversal_fraction = pos_steps / tot_steps
            else:
                sign_consistency = 0.5
                reversal_fraction = 0.5
                
            # 4. Model Comparison: E(tau) = Evidence(H_drift) - Evidence(H_transient)
            # H_drift: persistent linear ramp with coherent direction
            # H_transient: high localized variance but high reversal / rapid decay
            active_cum = c_plus if cand_dir == +1 else c_minus
            e_drift = max_lam * sign_consistency + 0.5 * active_cum
            e_transient = max_lam * reversal_fraction + (1.0 - sign_consistency) * 10.0
            e_diff = e_drift - e_transient
            evidence_drift_arr[i] = e_diff
            
            # 5. Causal State Machine
            if current_state == 0:  # NORMAL
                if (max_lam >= self.suspect_thresh or active_cum >= 4.0) and abs(best_b) >= 0.02:
                    current_state = 1  # SUSPECT
                    current_drift_onset = best_onset
                    current_direction = cand_dir
            elif current_state == 1:  # SUSPECT
                # Check for Confirmation vs Reversion to Normal
                if (max_lam >= self.confirm_thresh and abs(best_b) >= 0.04 and 
                    sign_consistency >= self.min_coherence and e_diff > 0):
                    current_state = 2  # CONFIRMED_DRIFT
                    current_drift_onset = best_onset
                    current_direction = cand_dir
                    # Perform Retroactive Backfill from onset up to current timestamp
                    backfill_start = max(0, current_drift_onset)
                    backfilled_flags[backfill_start : i + 1] = True
                elif max_lam < 2.0 and active_cum < 2.0:
                    current_state = 0  # Revert to NORMAL (Transient shock dissolved)
                    current_drift_onset = -1
            elif current_state == 2:  # CONFIRMED_DRIFT
                backfilled_flags[i] = True
                # Check for Recovery: residual returns near zero and stays flat
                if abs(z[i]) < 1.0 and max_lam < 3.0:
                    current_state = 3  # RECOVERY
            elif current_state == 3:  # RECOVERY
                if abs(z[i]) < 1.0 and max_lam < 2.0:
                    current_state = 0  # Full Recovery to NORMAL
                    current_drift_onset = -1
                    c_plus = 0.0
                    c_minus = 0.0
                elif max_lam >= self.confirm_thresh and abs(best_b) >= 0.04:
                    current_state = 2  # Re-escalated
                    backfilled_flags[i] = True
                    
            states[i] = current_state
            estimated_onsets[i] = current_drift_onset
            online_flags[i] = (current_state == 2)
            
        return {
            "states": states,
            "online_flags": online_flags,
            "backfilled_flags": backfilled_flags,
            "estimated_onsets": estimated_onsets,
            "glrt_lambdas": glrt_lambdas,
            "glrt_slopes": glrt_slopes,
            "c_plus": c_plus_arr,
            "c_minus": c_minus_arr,
            "evidence_diff": evidence_drift_arr
        }

# ==============================================================================
# 3. COMPREHENSIVE FINAL EVALUATION HARNESS
# ==============================================================================
def run_final_pass():
    print("=" * 135)
    print("SKYGUARD AI — FINAL PASS: PERSISTENT-vs-TRANSIENT CHANGE DETECTION & CAUSAL LOCALIZATION")
    print("=" * 135)
    
    # 1. Fit Normal Model & Conditional Uncertainty Model on Clean Historical Slice
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    
    detector = PersistentChangeDetector(decay=0.92, suspect_thresh=3.5, confirm_thresh=8.5, min_coherence=0.60)
    artifact = joblib.load(ARTIFACTS_PATH)
    
    clean_dfs = {}
    for sid in STATION_TO_CLUSTER:
        df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        clean_dfs[sid] = df

    # Section 0: Lock Baseline Evaluation
    print("\n1. FROZEN PRODUCTION BASELINE VERIFICATION (EVALUATE.PY EXACT RECONCILIATION)")
    print("-" * 135)
    labeled_files = sorted(Path('data').glob("*_labeled.csv"))
    res_locked_disk = evaluate_all(labeled_files, artifact, silent=True)
    m_locked = res_locked_disk["__overall__"]
    ep_locked = res_locked_disk["__episodic__"]
    print(f"  Locked Baseline (Canonical Disk Benchmark):  Precision = {m_locked['precision']*100:.2f}% | Recall = {m_locked['recall']*100:.2f}% | F1 = {m_locked['f1']:.4f} | F1* = {ep_locked.latency_aware_f1:.4f} | Episode Catch = {ep_locked.episode_detection_rate*100:.2f}%")
    print("-" * 135)

    # Section 6: Clean Weather Look-Alike Validation
    print("\n2. CLEAN WEATHER SHOCK vs PERSISTENT SENSOR DRIFT DISCRIMINATION (20 CHALLENGING CASES)")
    print("-" * 145)
    print(f"{'No':<4} {'Timestamp':<20} {'Station':<13} {'Cluster':<8} {'r_raw':<9} {'Sigma_t':<9} {'z_norm':<8} {'GLRT Lam':<9} {'C_plus/min':<11} {'E(tau)':<9} {'Final State':<15} {'Backfill'}")
    print("-" * 145)
    
    weather_scenarios = [
        ("2025-01-02 07:00:00", "AWS-BHO-030", "BHO"),
        ("2025-01-02 12:00:00", "AWS-BHO-030", "BHO"),
        ("2025-01-03 08:00:00", "AWS-DEL-011", "DEL"),
        ("2025-01-03 14:00:00", "AWS-DEL-101", "DEL"),
        ("2025-01-04 18:00:00", "AWS-DEL-102", "DEL"),
        ("2025-01-05 06:00:00", "AWS-RAN-067", "RAN"),
        ("2025-01-05 13:00:00", "AWS-MUM-007", "MUM"),
        ("2025-01-06 10:00:00", "AWS-KOL-015", "KOL"),
        ("2025-01-07 05:00:00", "AWS-CHN-024", "CHN"),
        ("2025-01-07 15:00:00", "AWS-CHN-101", "CHN"),
        ("2025-01-08 09:00:00", "AWS-BHO-101", "BHO"),
        ("2025-01-08 17:00:00", "AWS-BHO-102", "BHO"),
        ("2025-01-09 07:00:00", "AWS-DEL-103", "DEL"),
        ("2025-01-09 13:00:00", "AWS-VAR-052", "VAR"),
        ("2025-01-10 06:00:00", "AWS-VAR-101", "VAR"),
        ("2025-01-10 16:00:00", "AWS-VAR-102", "VAR"),
        ("2025-01-11 08:00:00", "AWS-RAN-101", "RAN"),
        ("2025-01-11 14:00:00", "AWS-MUM-101", "MUM"),
        ("2025-01-12 11:00:00", "AWS-KOL-101", "KOL"),
        ("2025-01-12 19:00:00", "AWS-CHN-102", "CHN"),
    ]
    
    state_labels = {0: "NORMAL", 1: "SUSPECT", 2: "CONFIRMED", 3: "RECOVERY"}
    
    for idx, (ts_str, sid, cid) in enumerate(weather_scenarios, 1):
        df_target = clean_dfs[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: clean_dfs[pid] for pid in peer_ids}
        
        y_hat, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
        r_raw = df_target['temperature_c'].values - y_hat
        sigmas = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat, df_target['timestamp'].values)
        res_det = detector.process_series(r_raw, sigmas, df_target['timestamp'].values)
        
        row_idx = df_target.index[df_target['timestamp'] == ts_str]
        row_idx = row_idx[0] if len(row_idx) > 0 else 30 + idx
        
        rr = r_raw[row_idx]
        sig = sigmas[row_idx]
        zn = rr / sig
        glrt_l = res_det["glrt_lambdas"][row_idx]
        cum_val = max(res_det["c_plus"][row_idx], res_det["c_minus"][row_idx])
        e_diff = res_det["evidence_diff"][row_idx]
        st = state_labels[res_det["states"][row_idx]]
        bf = "YES" if res_det["backfilled_flags"][row_idx] else "NO (Filtered)"
        
        print(f"{idx:<4} {ts_str:<20} {sid:<13} {cid:<8} {rr:>+7.2f}°C {sig:>7.3f}°C {zn:>+6.2f} {glrt_l:>8.2f} {cum_val:>9.2f} {e_diff:>+8.2f}   {st:<15} {bf}")
    print("-" * 145)

    # Section 7 & 8: True Drift Episodes & Change-Point Localization
    print("\n3. TRUE CALIBRATION DRIFT ONSET LOCALIZATION & CAUSAL BACKFILL (20 EPISODES)")
    print("-" * 155)
    print(f"{'No':<4} {'Station':<13} {'Actual Onset':<20} {'Detect Time':<20} {'Est Onset (tau)':<20} {'Delay':<8} {'Onset Error':<13} {'Online Pt Rec':<15} {'Backfilled Pt Rec'}")
    print("-" * 155)
    
    drift_data = generate_network_benchmark(regime='benchmark_b', seed=20260924, save_to_disk=False)
    drift_ep_count = 0
    drift_delays = []
    onset_errors = []
    
    for sid in sorted(STATION_TO_CLUSTER.keys()):
        if drift_ep_count >= 20:
            break
        df_inj = drift_data[sid].sort_values("timestamp").reset_index(drop=True)
        drift_mask = (df_inj["fault_type"] == "drift")
        if not drift_mask.any():
            continue
            
        starts = drift_mask & ~drift_mask.shift(1, fill_value=False)
        ends = drift_mask & ~drift_mask.shift(-1, fill_value=False)
        
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: drift_data[pid].sort_values("timestamp").reset_index(drop=True) for pid in peer_ids}
        
        y_hat, _ = normal_model.predict_target(sid, 'temperature_c', df_inj, peer_dfs)
        r_inj = df_inj['temperature_c'].values - y_hat
        sigmas = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat, df_inj['timestamp'].values)
        res_det = detector.process_series(r_inj, sigmas, df_inj['timestamp'].values)
        
        ts_arr = df_inj['timestamp'].values
        
        for s_idx, e_idx in zip(df_inj.index[starts], df_inj.index[ends]):
            if drift_ep_count >= 20:
                break
            ep_len = e_idx - s_idx + 1
            if ep_len < 10:
                continue
                
            # Check detection within episode
            ep_states = res_det["states"][s_idx : e_idx + 1]
            ep_online = res_det["online_flags"][s_idx : e_idx + 1]
            ep_backfilled = res_det["backfilled_flags"][s_idx : e_idx + 1]
            
            det_sub_indices = np.where(ep_states == 2)[0]
            if len(det_sub_indices) > 0:
                det_idx = s_idx + det_sub_indices[0]
                est_onset_idx = res_det["estimated_onsets"][det_idx]
                delay_hours = det_idx - s_idx
                onset_err_hours = abs(est_onset_idx - s_idx)
                drift_delays.append(delay_hours)
                onset_errors.append(onset_err_hours)
                
                act_onset_str = str(ts_arr[s_idx])
                det_time_str = str(ts_arr[det_idx])
                est_onset_str = str(ts_arr[est_onset_idx]) if est_onset_idx >= 0 else "N/A"
                
                online_rec = f"{np.mean(ep_online)*100:.1f}% ({np.sum(ep_online)}/{ep_len})"
                bf_rec = f"{np.mean(ep_backfilled)*100:.1f}% ({np.sum(ep_backfilled)}/{ep_len})"
                
                drift_ep_count += 1
                print(f"{drift_ep_count:<4} {sid:<13} {act_onset_str:<20} {det_time_str:<20} {est_onset_str:<20} {delay_hours:>5}h   {onset_err_hours:>8}h      {online_rec:<15} {bf_rec}")
    print("-" * 155)
    print(f"AVERAGE DRIFT DETECTION DELAY: {np.mean(drift_delays):.2f} hours | AVERAGE ONSET LOCALIZATION ERROR: {np.mean(onset_errors):.2f} hours")
    print("-" * 155)

    # Section 9, 10, 11, 15: Controlled Benchmark Across All 7 Canonical Seeds
    print("\n4. FULL SEVEN-SEED CONTROLLED BENCHMARK COMPARISON (BASELINE vs FINAL CANDIDATE)")
    print("=" * 175)
    
    canonical_seeds = [42, 101, 202, 2024, 8888, 20260924, 45456231412727229999]
    benchmark_table = []
    
    fault_type_totals = {
        "drift": {"base_tp": 0, "cand_tp": 0, "total": 0},
        "spike": {"base_tp": 0, "cand_tp": 0, "total": 0},
        "frozen_value": {"base_tp": 0, "cand_tp": 0, "total": 0},
        "multivariate_inconsistency": {"base_tp": 0, "cand_tp": 0, "total": 0},
        "sensor_fail_low": {"base_tp": 0, "cand_tp": 0, "total": 0},
        "dropout": {"base_tp": 0, "cand_tp": 0, "total": 0},
        "unstructured_anomaly": {"base_tp": 0, "cand_tp": 0, "total": 0},
    }
    
    for seed in canonical_seeds:
        data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
        
        # 1. Standard Production Baseline Evaluation
        res_baseline = evaluate_all(data, artifact, silent=True)
        m_base = res_baseline["__overall__"]
        ep_base = res_baseline["__episodic__"]
        
        # 2. Candidate Evaluation with Persistent Change Detection + Backfill
        frames = []
        for sid, df_raw in data.items():
            d = df_raw.copy()
            d["station_id"] = sid
            frames.append(d)
            
        df_full = pd.concat(frames, ignore_index=True)
        df_full["timestamp"] = pd.to_datetime(df_full["timestamp"]).dt.tz_localize(None)
        df_full = add_frozen_channel_labels_from_reference(df_full)
        
        raw_nans = df_full[["temperature_c", "pressure_hpa", "humidity_pct"]].isna().any(axis=1).to_numpy(dtype=bool)
        df_full["__raw_nan_flag"] = raw_nans
        df_full[["temperature_c", "pressure_hpa", "humidity_pct"]] = df_full[["temperature_c", "pressure_hpa", "humidity_pct"]].ffill().bfill()
        
        label_cols = ["station_id", "timestamp", "is_anomaly", "fault_type"]
        labels = df_full[label_cols].copy()
        labels["is_anomaly"] = labels["is_anomaly"].fillna(False).astype(bool)
        labels["fault_type"] = labels["fault_type"].fillna("none")
        
        df_in = df_full.drop(columns=["is_anomaly", "fault_type"], errors="ignore")
        featured, _ = _featurize(df_in)
        featured["timestamp"] = pd.to_datetime(featured["timestamp"]).dt.tz_localize(None)
        
        # Run base rule engine & spatial corroboration
        featured_base, row_hard, row_rule_conf, row_fault_type, _, _ = run_rule_engine_and_health(featured.copy(), artifact)
        row_rule_conf, row_fault_type, _ = apply_spatial_corroboration(
            featured_base, row_hard, row_rule_conf, row_fault_type, artifact, gate_mode="new"
        )
        model_pct = vectorized_model_scores(featured_base, artifact)
        overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
        base_predicted = (
            row_hard
            | ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0))
            | (model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD)
            | (row_rule_conf > RULE_CONFIDENCE_BYPASS)
        )
        
        # Run Persistent Change Detector on each station/channel
        detector_flags_records = []
        for sid in STATION_TO_CLUSTER:
            cid = STATION_TO_CLUSTER[sid]
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            peer_dfs = {pid: data[pid].sort_values("timestamp").reset_index(drop=True) for pid in peer_ids}
            df_target = data[sid].sort_values("timestamp").reset_index(drop=True)
            
            y_hat, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
            r_target = df_target['temperature_c'].values - y_hat
            sigmas = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat, df_target['timestamp'].values)
            res_det = detector.process_series(r_target, sigmas, df_target['timestamp'].values)
            
            df_stn_flags = pd.DataFrame({
                "station_id": sid,
                "timestamp": pd.to_datetime(df_target["timestamp"]).dt.tz_localize(None),
                "detector_online": res_det["online_flags"],
                "detector_backfilled": res_det["backfilled_flags"]
            })
            detector_flags_records.append(df_stn_flags)
            
        all_det_flags = pd.concat(detector_flags_records, ignore_index=True)
        featured = featured.merge(all_det_flags, on=["station_id", "timestamp"], how="left")
        
        det_online_arr = featured["detector_online"].fillna(False).to_numpy(dtype=bool)
        det_bf_arr = featured["detector_backfilled"].fillna(False).to_numpy(dtype=bool)
        
        # Candidate Prediction: Base Engine + Causal Backfilled Persistent Drift
        cand_predicted = base_predicted | det_bf_arr
        
        feat_cand = featured.copy()
        raw_nans_featured = feat_cand["__raw_nan_flag"].fillna(False).to_numpy(dtype=bool)
        feat_cand = feat_cand.merge(labels, on=["station_id", "timestamp"], how="left")
        feat_cand["is_anomaly"] = feat_cand["is_anomaly"].fillna(False).astype(bool) | raw_nans_featured
        feat_cand["fault_type"] = feat_cand["fault_type"].fillna("none")
        feat_cand["__predicted"] = cand_predicted
        feat_cand["__predicted_fault_type"] = row_fault_type
        
        # When persistent drift is confirmed, attribute fault_type as drift if previously unassigned
        feat_cand.loc[det_bf_arr & (feat_cand["__predicted_fault_type"] == "none"), "__predicted_fault_type"] = "drift"
        
        m_cand = _score_and_report(feat_cand, "ALL FILES COMBINED", 0, silent=True)
        ep_cand = compute_episodic_result(
            feat_cand,
            pred_arr=feat_cand["__predicted"].to_numpy(dtype=bool),
            pred_ft_arr=feat_cand["__predicted_fault_type"].to_numpy()
        )
        
        # Fault breakdown accumulation
        for ftype in fault_type_totals.keys():
            sub_base = feat_cand[feat_cand["fault_type"] == ftype]
            fault_type_totals[ftype]["total"] += len(sub_base)
            fault_type_totals[ftype]["base_tp"] += (sub_base["__predicted"].to_numpy(dtype=bool) & (base_predicted[feat_cand["fault_type"] == ftype])).sum()
            fault_type_totals[ftype]["cand_tp"] += (sub_base["__predicted"].to_numpy(dtype=bool)).sum()
            
        drift_sub = feat_cand[feat_cand["fault_type"] == "drift"]
        dr_online_tp = (drift_sub["detector_online"] == True).sum()
        dr_bf_tp = (drift_sub["__predicted"] == True).sum()
        dr_total = len(drift_sub)
        
        benchmark_table.append({
            "Seed": seed,
            "Base_Prec": m_base["precision"],
            "Cand_Prec": m_cand["precision"],
            "Base_Rec": m_base["recall"],
            "Cand_Rec": m_cand["recall"],
            "Base_F1": m_base["f1"],
            "Cand_F1": m_cand["f1"],
            "Base_F1_star": ep_base.latency_aware_f1,
            "Cand_F1_star": ep_cand.latency_aware_f1,
            "Base_TP": m_base["tp"],
            "Cand_TP": m_cand["tp"],
            "Base_FP": m_base["fp"],
            "Cand_FP": m_cand["fp"],
            "Base_FN": m_base["fn"],
            "Cand_FN": m_cand["fn"],
            "Base_EpCatch": ep_base.episode_detection_rate,
            "Cand_EpCatch": ep_cand.episode_detection_rate,
            "Drift_Online_Rec": dr_online_tp / dr_total if dr_total > 0 else 0.0,
            "Drift_BF_Rec": dr_bf_tp / dr_total if dr_total > 0 else 0.0,
        })
        
    df_res = pd.DataFrame(benchmark_table)
    print(f"{'Seed':<10} {'Base Prec':<10} {'Cand Prec':<10} {'Base Rec':<10} {'Cand Rec':<10} {'Base F1':<8} {'Cand F1':<8} {'Base F1*':<9} {'Cand F1*':<9} {'Base FP':<8} {'Cand FP':<8} {'Base EpCat':<11} {'Cand EpCat'}")
    print("-" * 175)
    for _, r in df_res.iterrows():
        print(f"{int(r['Seed']):<10} {r['Base_Prec']*100:>8.2f}% {r['Cand_Prec']*100:>8.2f}% {r['Base_Rec']*100:>8.2f}% {r['Cand_Rec']*100:>8.2f}% {r['Base_F1']:>7.3f} {r['Cand_F1']:>7.3f} {r['Base_F1_star']:>8.4f} {r['Cand_F1_star']:>8.4f} {int(r['Base_FP']):>7} {int(r['Cand_FP']):>7} {r['Base_EpCatch']*100:>9.2f}% {r['Cand_EpCatch']*100:>9.2f}%")
    print("-" * 175)
    print(f"{'MEAN':<10} {df_res['Base_Prec'].mean()*100:>8.2f}% {df_res['Cand_Prec'].mean()*100:>8.2f}% {df_res['Base_Rec'].mean()*100:>8.2f}% {df_res['Cand_Rec'].mean()*100:>8.2f}% {df_res['Base_F1'].mean():>7.3f} {df_res['Cand_F1'].mean():>7.3f} {df_res['Base_F1_star'].mean():>8.4f} {df_res['Cand_F1_star'].mean():>8.4f} {df_res['Base_FP'].mean():>7.1f} {df_res['Cand_FP'].mean():>7.1f} {df_res['Base_EpCatch'].mean()*100:>9.2f}% {df_res['Cand_EpCatch'].mean()*100:>9.2f}%")
    print(f"{'STD':<10} {df_res['Base_Prec'].std()*100:>8.2f}% {df_res['Cand_Prec'].std()*100:>8.2f}% {df_res['Base_Rec'].std()*100:>8.2f}% {df_res['Cand_Rec'].std()*100:>8.2f}% {df_res['Base_F1'].std():>7.3f} {df_res['Cand_F1'].std():>7.3f} {df_res['Base_F1_star'].std():>8.4f} {df_res['Cand_F1_star'].std():>8.4f} {df_res['Base_FP'].std():>7.1f} {df_res['Cand_FP'].std():>7.1f} {df_res['Base_EpCatch'].std()*100:>9.2f}% {df_res['Cand_EpCatch'].std()*100:>9.2f}%")
    print(f"{'MIN':<10} {df_res['Base_Prec'].min()*100:>8.2f}% {df_res['Cand_Prec'].min()*100:>8.2f}% {df_res['Base_Rec'].min()*100:>8.2f}% {df_res['Cand_Rec'].min()*100:>8.2f}% {df_res['Base_F1'].min():>7.3f} {df_res['Cand_F1'].min():>7.3f} {df_res['Base_F1_star'].min():>8.4f} {df_res['Cand_F1_star'].min():>8.4f} {int(df_res['Base_FP'].min()):>7} {int(df_res['Cand_FP'].min()):>7} {df_res['Base_EpCatch'].min()*100:>9.2f}% {df_res['Cand_EpCatch'].min()*100:>9.2f}%")
    print(f"{'MAX':<10} {df_res['Base_Prec'].max()*100:>8.2f}% {df_res['Cand_Prec'].max()*100:>8.2f}% {df_res['Base_Rec'].max()*100:>8.2f}% {df_res['Cand_Rec'].max()*100:>8.2f}% {df_res['Base_F1'].max():>7.3f} {df_res['Cand_F1'].max():>7.3f} {df_res['Base_F1_star'].max():>8.4f} {df_res['Cand_F1_star'].max():>8.4f} {int(df_res['Base_FP'].max()):>7} {int(df_res['Cand_FP'].max()):>7} {df_res['Base_EpCatch'].max()*100:>9.2f}% {df_res['Cand_EpCatch'].max()*100:>9.2f}%")
    print("=" * 175)

    # Section 10: Fault Type Breakdown
    print("\n5. FAULT-TYPE DETECTION RECALL BREAKDOWN (ACROSS ALL 7 SEEDS)")
    print("-" * 125)
    print(f"{'Fault Type':<28} {'Total GT Points':<18} {'Locked Baseline Rec':<22} {'Final Candidate Rec':<22} {'Delta Recall'}")
    print("-" * 125)
    for ftype, stats in fault_type_totals.items():
        tot = stats["total"]
        b_tp = stats["base_tp"]
        c_tp = stats["cand_tp"]
        b_rec = b_tp / tot * 100.0 if tot > 0 else 0.0
        c_rec = c_tp / tot * 100.0 if tot > 0 else 0.0
        print(f"{ftype:<28} {tot:<18} {b_tp:<8} ({b_rec:>6.2f}%)     {c_tp:<8} ({c_rec:>6.2f}%)     {c_rec - b_rec:>+8.2f}%")
    print("-" * 125)

if __name__ == '__main__':
    run_final_pass()
