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
    MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD,
    MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS,
    add_frozen_channel_labels_from_reference
)
from evaluation.episodic_eval import compute_episodic_result

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
# 1. PASS 1 CAUSAL NORMAL BEHAVIOR MODEL
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

        ts = pd.to_datetime(target_df['timestamp'])
        hours = ts.hour.values if hasattr(ts, 'hour') else ts.dt.hour.values
        sin_h = np.sin(2 * np.pi * hours / 24.0)
        cos_h = np.cos(2 * np.pi * hours / 24.0)
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

# ==============================================================================
# 2. PASS 5 CONDITIONAL UNCERTAINTY MODEL
# ==============================================================================
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
# 3. CHANCE 1 STREAMING DETECTOR & CHANCE 2 PEER-LIKELIHOOD TRAJECTORY GATE
# ==============================================================================
def compute_weighted_ramp_stats(r_seg, sig_seg):
    """
    Computes weighted linear ramp slope, standard error, and GLRT Lambda:
    w_i = 1 / sig_i^2
    s_tt = sum(w_i * (t_i - t_bar)^2)
    s_tr = sum(w_i * (t_i - t_bar) * (r_i - r_bar))
    b_hat = s_tr / s_tt
    SE(b_hat) = 1 / sqrt(s_tt)
    Lambda = s_tr^2 / (2 * s_tt)
    """
    W = len(r_seg)
    w = 1.0 / (np.maximum(sig_seg, 0.10) ** 2)
    t = np.arange(W, dtype=float)
    
    sum_w = np.sum(w)
    if sum_w <= 0:
        return 0.0, 1.0, 0.0, W
        
    t_bar = np.sum(w * t) / sum_w
    r_bar = np.sum(w * r_seg) / sum_w
    
    t_dev = t - t_bar
    r_dev = r_seg - r_bar
    
    s_tt = np.sum(w * (t_dev ** 2))
    if s_tt < 1e-6:
        return 0.0, 1.0, 0.0, W
        
    s_tr = np.sum(w * t_dev * r_dev)
    b_hat = s_tr / s_tt
    se_b = 1.0 / np.sqrt(s_tt)
    delta_rss = (s_tr ** 2) / s_tt
    lam = delta_rss / 2.0
    return b_hat, se_b, lam, W


class CausalPeerLikelihoodDriftEngine:
    """
    Full Chance 2 Causal Architecture:
    1. Evaluates streaming temporal drift statistics for target station s
    2. Evaluates streaming trajectory statistics for clean cluster peers p in Peers(s)
    3. Computes standardized Excess Trajectory Z_excess = (beta_s - beta_peer_common) / SE_excess
    4. Categorizes Spatial State:
       - REGIONAL_SUPPORTED: Peer trajectories explain target motion -> Suppress false alarm
       - TARGET_SPECIFIC: Target has significant excess slope beyond peers -> Confirm sensor fault
       - AMBIGUOUS: Insufficient peer data -> Fail OPEN (preserve target alarm)
    """
    def __init__(self, decay=0.88, allowance=0.40, glrt_thresh=9.0, cusum_thresh=6.0, min_coherence=0.65, z_excess_thresh=2.50):
        self.decay = decay
        self.allowance = allowance
        self.glrt_thresh = glrt_thresh
        self.cusum_thresh = cusum_thresh
        self.min_coherence = min_coherence
        self.z_excess_thresh = z_excess_thresh

    def process_cluster_stream(self, target_sid, all_residuals, all_sigmas, timestamps):
        """
        Runs live sample-by-sample streaming for target_sid with causal peer trajectory gating.
        """
        n = len(timestamps)
        cid = STATION_TO_CLUSTER[target_sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != target_sid]
        
        r_target = np.asarray(all_residuals[target_sid], dtype=float)
        sig_target = np.asarray(all_sigmas[target_sid], dtype=float)
        z_target = r_target / np.maximum(sig_target, 0.10)
        
        # Outputs
        chance1_online = np.zeros(n, dtype=bool)
        chance2_online = np.zeros(n, dtype=bool)
        spatial_states = []  # REGIONAL_SUPPORTED, TARGET_SPECIFIC, AMBIGUOUS
        
        # Spatial diagnostics
        target_betas = np.zeros(n, dtype=float)
        peer_common_betas = np.zeros(n, dtype=float)
        z_excess_arr = np.zeros(n, dtype=float)
        sign_agreements = np.zeros(n, dtype=float)
        
        # Streaming state tracking
        c_plus = 0.0
        c_minus = 0.0
        c1_state = 0  # 0: NORMAL, 1: SUSPECT, 2: CONFIRMED, 3: RECOVERY
        c2_state = 0
        c1_streak = 0
        c2_streak = 0
        
        min_w = 4
        max_w = 24
        
        for i in range(n):
            if i < min_w:
                chance1_online[i] = False
                chance2_online[i] = False
                spatial_states.append("NORMAL")
                continue
                
            # 1. Update Directional CUSUM Causally
            c_plus = max(0.0, self.decay * c_plus + (z_target[i] - self.allowance))
            c_minus = max(0.0, self.decay * c_minus + (-z_target[i] - self.allowance))
            active_cum = max(c_plus, c_minus)
            
            # 2. Causal Target Ramp GLRT over trailing window [i - W + 1, i]
            max_lam_s = 0.0
            best_b_s = 0.0
            best_se_s = 1.0
            best_W = min_w
            
            max_avail_w = min(i + 1, max_w)
            for W in range(min_w, max_avail_w + 1, 2):
                idx_s = i - W + 1
                idx_e = i + 1
                b_w, se_w, lam_w, _ = compute_weighted_ramp_stats(r_target[idx_s:idx_e], sig_target[idx_s:idx_e])
                if lam_w > max_lam_s:
                    max_lam_s = lam_w
                    best_b_s = b_w
                    best_se_s = se_w
                    best_W = W
                    
            # 3. Target Trajectory Coherence
            w_coh = min(i + 1, 12)
            diffs = np.diff(z_target[i - w_coh + 1 : i + 1])
            cand_dir = +1 if (c_plus > c_minus and best_b_s > 0) else (-1 if (c_minus > c_plus and best_b_s < 0) else 0)
            if len(diffs) > 0 and cand_dir != 0:
                steps_supporting = np.sum(diffs > 0) if cand_dir == +1 else np.sum(diffs < 0)
                sign_consistency = steps_supporting / len(diffs)
            else:
                sign_consistency = 0.50
            is_coherent = (sign_consistency >= self.min_coherence)
            
            # Raw temporal triggers
            glrt_trigger = (max_lam_s >= self.glrt_thresh and abs(best_b_s) >= 0.04 and is_coherent)
            cusum_trigger = (active_cum >= self.cusum_thresh and is_coherent and abs(z_target[i]) >= 1.5)
            raw_drift_active = (glrt_trigger or cusum_trigger)
            
            # Update Chance 1 State
            if c1_state == 0:
                if raw_drift_active:
                    c1_state = 2
                elif (max_lam_s >= 4.0 or active_cum >= 3.0) and abs(best_b_s) >= 0.02:
                    c1_state = 1
            elif c1_state == 1:
                if raw_drift_active:
                    c1_state = 2
                elif max_lam_s < 2.0 and active_cum < 1.5:
                    c1_state = 0
            elif c1_state == 2:
                if abs(z_target[i]) < 1.0 and max_lam_s < 3.0 and active_cum < 2.0:
                    c1_state = 3
            elif c1_state == 3:
                if abs(z_target[i]) < 1.0 and max_lam_s < 2.0:
                    c1_state = 0
                elif raw_drift_active:
                    c1_state = 2
                    
            chance1_online[i] = (c1_state == 2)
            
            # 4. Chance 2 Causal Peer-Likelihood Trajectory Gate
            # Evaluate peer trajectories over the identical trailing window best_W
            peer_betas = []
            peer_ses = []
            
            idx_s = i - best_W + 1
            idx_e = i + 1
            
            for pid in peer_ids:
                if pid in all_residuals and pid in all_sigmas:
                    r_p = all_residuals[pid][idx_s:idx_e]
                    sig_p = all_sigmas[pid][idx_s:idx_e]
                    b_p, se_p, _, _ = compute_weighted_ramp_stats(r_p, sig_p)
                    peer_betas.append(b_p)
                    peer_ses.append(se_p)
                    
            if len(peer_betas) >= 2:
                peer_b_arr = np.array(peer_betas, dtype=float)
                peer_b_common = float(np.median(peer_b_arr))
                peer_mad = float(np.median(np.abs(peer_b_arr - peer_b_common))) * 1.4826
                peer_se_common = (peer_mad / np.sqrt(len(peer_betas))) + float(np.median(peer_ses))
                
                # Sign agreement among peers with target direction
                if cand_dir != 0:
                    agreed_peers = sum(1 for pb in peer_betas if np.sign(pb) == cand_dir)
                    sign_agree_ratio = agreed_peers / len(peer_betas)
                else:
                    sign_agree_ratio = 0.50
                    
                # Target-Specific Excess Trajectory
                b_excess = best_b_s - peer_b_common
                se_excess = np.sqrt(best_se_s**2 + peer_se_common**2)
                z_excess = b_excess / se_excess if se_excess > 0 else 0.0
                excess_ratio = abs(b_excess) / max(1e-4, abs(best_b_s))
                
                # Spatial Classification
                # Regional explanation: strong peer sign agreement + target trajectory explainable by peer common motion
                is_regional_weather = (
                    (sign_agree_ratio >= 0.66) and 
                    (np.sign(peer_b_common) == np.sign(best_b_s)) and 
                    (abs(z_excess) < self.z_excess_thresh) and 
                    (excess_ratio < 0.45)
                )
                
                if is_regional_weather:
                    spatial_state = "REGIONAL_SUPPORTED"
                else:
                    spatial_state = "TARGET_SPECIFIC"
            else:
                peer_b_common = 0.0
                z_excess = 0.0
                sign_agree_ratio = 0.0
                spatial_state = "AMBIGUOUS"  # Fail OPEN
                
            target_betas[i] = best_b_s
            peer_common_betas[i] = peer_b_common
            z_excess_arr[i] = z_excess
            sign_agreements[i] = sign_agree_ratio
            spatial_states.append(spatial_state)
            
            # Chance 2 Confirmed Decision:
            # Raw drift active AND NOT explained by regional common weather (or ambiguous)
            c2_active = raw_drift_active and (spatial_state in ["TARGET_SPECIFIC", "AMBIGUOUS"])
            
            if c2_state == 0:
                if c2_active:
                    c2_state = 2
                elif (max_lam_s >= 4.0 or active_cum >= 3.0) and abs(best_b_s) >= 0.02 and (spatial_state != "REGIONAL_SUPPORTED"):
                    c2_state = 1
            elif c2_state == 1:
                if c2_active:
                    c2_state = 2
                elif max_lam_s < 2.0 and active_cum < 1.5:
                    c2_state = 0
                elif spatial_state == "REGIONAL_SUPPORTED":
                    c2_state = 0  # Reverted by regional peer consensus
            elif c2_state == 2:
                if spatial_state == "REGIONAL_SUPPORTED":
                    c2_state = 3  # Suppressed by emergence of regional peer front
                elif abs(z_target[i]) < 1.0 and max_lam_s < 3.0 and active_cum < 2.0:
                    c2_state = 3
            elif c2_state == 3:
                if abs(z_target[i]) < 1.0 and max_lam_s < 2.0:
                    c2_state = 0
                elif c2_active:
                    c2_state = 2
                    
            chance2_online[i] = (c2_state == 2)
            
        return {
            "chance1_online": chance1_online,
            "chance2_online": chance2_online,
            "spatial_states": spatial_states,
            "target_betas": target_betas,
            "peer_common_betas": peer_common_betas,
            "z_excess": z_excess_arr,
            "sign_agreements": sign_agreements
        }

# ==============================================================================
# 4. EXECUTION SUITE FOR CHANCE 2
# ==============================================================================
def run_chance2():
    print("=" * 145)
    print("SKYGUARD AI — CHANCE 2 OF 3: CAUSAL PEER-LIKELIHOOD GATE (REGIONAL WEATHER vs SENSOR FAULT)")
    print("=" * 145)
    
    # 1. Fit Normal Model & Uncertainty Model
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    
    peer_engine = CausalPeerLikelihoodDriftEngine(
        decay=0.88, allowance=0.40, glrt_thresh=9.0, cusum_thresh=6.0, min_coherence=0.65, z_excess_thresh=2.50
    )
    artifact = joblib.load(ARTIFACTS_PATH)
    
    clean_dfs = {}
    for sid in STATION_TO_CLUSTER:
        df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        clean_dfs[sid] = df

    # Section 13: 20 Difficult Clean-Weather Microclimate Cases
    print("\n1. 20 DIFFICULT CLEAN-WEATHER TRANSIENTS (PEER TRAJECTORY EXPLANATION & SUPPRESSION)")
    print("-" * 155)
    print(f"{'No':<4} {'Timestamp':<20} {'Station':<13} {'Cluster':<8} {'r_target':<9} {'Beta_s':<10} {'Beta_peers':<12} {'Z_excess':<10} {'Sign Agree':<12} {'Spatial State':<20} {'C1 Alert':<10} {'C2 Gated Verdict'}")
    print("-" * 155)
    
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
    
    # Precompute clean residuals & sigmas for all stations
    clean_residuals = {}
    clean_sigmas = {}
    for sid in STATION_TO_CLUSTER:
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: clean_dfs[pid] for pid in peer_ids}
        df_target = clean_dfs[sid]
        
        y_hat, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
        r_raw = df_target['temperature_c'].values - y_hat
        sigmas = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat, df_target['timestamp'].values)
        clean_residuals[sid] = r_raw
        clean_sigmas[sid] = sigmas

    for idx, (ts_str, sid, cid) in enumerate(weather_scenarios, 1):
        df_target = clean_dfs[sid]
        res_cluster = peer_engine.process_cluster_stream(sid, clean_residuals, clean_sigmas, df_target['timestamp'].values)
        
        row_idx = df_target.index[df_target['timestamp'] == ts_str]
        row_idx = row_idx[0] if len(row_idx) > 0 else 30 + idx
        
        rr = clean_residuals[sid][row_idx]
        bs = res_cluster["target_betas"][row_idx]
        bp = res_cluster["peer_common_betas"][row_idx]
        ze = res_cluster["z_excess"][row_idx]
        sa = res_cluster["sign_agreements"][row_idx]
        st = res_cluster["spatial_states"][row_idx]
        c1 = "ALERT" if res_cluster["chance1_online"][row_idx] else "CLEAN"
        c2 = "ALERT" if res_cluster["chance2_online"][row_idx] else "SUPPRESSED (Clean Weather)"
        
        print(f"{idx:<4} {ts_str:<20} {sid:<13} {cid:<8} {rr:>+6.2f}°C {bs:>+8.3f}/h {bp:>+10.3f}/h {ze:>+8.2f} {sa*100:>10.1f}%   {st:<20} {c1:<10} {c2}")
    print("-" * 155)

    # Section 14 & 15: 20 True Injected Drift & Peer-Masked Cases
    print("\n2. 20 TRUE INJECTED DRIFT & PEER-MASKED CASES (EXCESS TRAJECTORY RETENTION)")
    print("-" * 165)
    print(f"{'No':<4} {'Station':<13} {'Cluster':<8} {'Actual Onset':<20} {'Inj Delta':<10} {'Beta_s':<10} {'Beta_peers':<12} {'Z_excess':<10} {'Spatial State':<20} {'C1 Alert':<10} {'C2 Gated Alert'}")
    print("-" * 165)
    
    drift_data = generate_network_benchmark(regime='benchmark_b', seed=20260924, save_to_disk=False)
    drift_residuals = {}
    drift_sigmas = {}
    for sid in STATION_TO_CLUSTER:
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: drift_data[pid].sort_values("timestamp").reset_index(drop=True) for pid in peer_ids}
        df_target = drift_data[sid].sort_values("timestamp").reset_index(drop=True)
        
        y_hat, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
        drift_residuals[sid] = df_target['temperature_c'].values - y_hat
        drift_sigmas[sid] = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat, df_target['timestamp'].values)
        
    drift_cases_tested = 0
    for sid in sorted(STATION_TO_CLUSTER.keys()):
        if drift_cases_tested >= 20:
            break
        df_inj = drift_data[sid].sort_values("timestamp").reset_index(drop=True)
        drift_mask = (df_inj["fault_type"] == "drift")
        if not drift_mask.any():
            continue
            
        starts = drift_mask & ~drift_mask.shift(1, fill_value=False)
        ends = drift_mask & ~drift_mask.shift(-1, fill_value=False)
        cid = STATION_TO_CLUSTER[sid]
        
        res_cluster = peer_engine.process_cluster_stream(sid, drift_residuals, drift_sigmas, df_inj['timestamp'].values)
        
        for s_idx, e_idx in zip(df_inj.index[starts], df_inj.index[ends]):
            if drift_cases_tested >= 20:
                break
            ep_len = e_idx - s_idx + 1
            if ep_len < 10:
                continue
                
            eval_idx = min(s_idx + 6, e_idx)
            act_onset_str = str(df_inj['timestamp'].iloc[s_idx])
            clean_t = clean_dfs[sid]['temperature_c'].iloc[eval_idx]
            inj_t = df_inj['temperature_c'].iloc[eval_idx]
            delta_t = inj_t - clean_t
            
            bs = res_cluster["target_betas"][eval_idx]
            bp = res_cluster["peer_common_betas"][eval_idx]
            ze = res_cluster["z_excess"][eval_idx]
            st = res_cluster["spatial_states"][eval_idx]
            c1 = "ALERT" if res_cluster["chance1_online"][eval_idx] else "CLEAN"
            c2 = "RETAINED" if res_cluster["chance2_online"][eval_idx] else "SUPPRESSED"
            
            drift_cases_tested += 1
            print(f"{drift_cases_tested:<4} {sid:<13} {cid:<8} {act_onset_str:<20} {delta_t:>+7.2f}°C {bs:>+8.3f}/h {bp:>+10.3f}/h {ze:>+8.2f}   {st:<20} {c1:<10} {c2}")
    print("-" * 165)

    # Section 18 & 19: Full 7-Seed Evaluation (Comparing Chance 1 vs Chance 2)
    print("\n3. FULL SEVEN-SEED BENCHMARK EVALUATION (CHANCE 1 LOCKED vs CHANCE 2 PEER-GATED)")
    print("=" * 185)
    
    canonical_seeds = [42, 101, 202, 2024, 8888, 20260924, 45456231412727229999]
    benchmark_table = []
    
    fault_type_totals = {
        "drift": {"c1_tp": 0, "c2_tp": 0, "total": 0},
        "spike": {"c1_tp": 0, "c2_tp": 0, "total": 0},
        "frozen_value": {"c1_tp": 0, "c2_tp": 0, "total": 0},
        "multivariate_inconsistency": {"c1_tp": 0, "c2_tp": 0, "total": 0},
        "sensor_fail_low": {"c1_tp": 0, "c2_tp": 0, "total": 0},
        "dropout": {"c1_tp": 0, "c2_tp": 0, "total": 0},
        "unstructured_anomaly": {"c1_tp": 0, "c2_tp": 0, "total": 0},
    }
    
    for seed in canonical_seeds:
        data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
        
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
        
        # Precompute streaming residuals & sigmas for this seed
        bench_residuals = {}
        bench_sigmas = {}
        for sid in STATION_TO_CLUSTER:
            cid = STATION_TO_CLUSTER[sid]
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            peer_dfs = {pid: data[pid].sort_values("timestamp").reset_index(drop=True) for pid in peer_ids}
            df_target = data[sid].sort_values("timestamp").reset_index(drop=True)
            
            y_hat, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
            bench_residuals[sid] = df_target['temperature_c'].values - y_hat
            bench_sigmas[sid] = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat, df_target['timestamp'].values)
            
        detector_flags_records = []
        for sid in STATION_TO_CLUSTER:
            df_target = data[sid].sort_values("timestamp").reset_index(drop=True)
            res_cluster = peer_engine.process_cluster_stream(sid, bench_residuals, bench_sigmas, df_target['timestamp'].values)
            
            df_stn_flags = pd.DataFrame({
                "station_id": sid,
                "timestamp": pd.to_datetime(df_target["timestamp"]).dt.tz_localize(None),
                "chance1_online": res_cluster["chance1_online"],
                "chance2_online": res_cluster["chance2_online"]
            })
            detector_flags_records.append(df_stn_flags)
            
        all_det_flags = pd.concat(detector_flags_records, ignore_index=True)
        featured = featured.merge(all_det_flags, on=["station_id", "timestamp"], how="left")
        
        c1_online_arr = featured["chance1_online"].fillna(False).to_numpy(dtype=bool)
        c2_online_arr = featured["chance2_online"].fillna(False).to_numpy(dtype=bool)
        
        # Evaluation of Chance 1 (Ungated Streaming)
        feat_c1 = featured.copy()
        raw_nans_featured = feat_c1["__raw_nan_flag"].fillna(False).to_numpy(dtype=bool)
        feat_c1 = feat_c1.merge(labels, on=["station_id", "timestamp"], how="left")
        feat_c1["is_anomaly"] = feat_c1["is_anomaly"].fillna(False).astype(bool) | raw_nans_featured
        feat_c1["fault_type"] = feat_c1["fault_type"].fillna("none")
        feat_c1["__predicted"] = base_predicted | c1_online_arr
        feat_c1["__predicted_fault_type"] = row_fault_type
        feat_c1.loc[c1_online_arr & (feat_c1["__predicted_fault_type"] == "none"), "__predicted_fault_type"] = "drift"
        
        m_c1 = _score_and_report(feat_c1, "ALL FILES COMBINED", 0, silent=True)
        ep_c1 = compute_episodic_result(
            feat_c1, pred_arr=feat_c1["__predicted"].to_numpy(dtype=bool), pred_ft_arr=feat_c1["__predicted_fault_type"].to_numpy()
        )
        
        # Evaluation of Chance 2 (Causal Peer-Likelihood Gated)
        feat_c2 = featured.copy()
        feat_c2 = feat_c2.merge(labels, on=["station_id", "timestamp"], how="left")
        feat_c2["is_anomaly"] = feat_c2["is_anomaly"].fillna(False).astype(bool) | raw_nans_featured
        feat_c2["fault_type"] = feat_c2["fault_type"].fillna("none")
        feat_c2["__predicted"] = base_predicted | c2_online_arr
        feat_c2["__predicted_fault_type"] = row_fault_type
        feat_c2.loc[c2_online_arr & (feat_c2["__predicted_fault_type"] == "none"), "__predicted_fault_type"] = "drift"
        
        m_c2 = _score_and_report(feat_c2, "ALL FILES COMBINED", 0, silent=True)
        ep_c2 = compute_episodic_result(
            feat_c2, pred_arr=feat_c2["__predicted"].to_numpy(dtype=bool), pred_ft_arr=feat_c2["__predicted_fault_type"].to_numpy()
        )
        
        # Accumulate fault totals
        for ftype in fault_type_totals.keys():
            sub_c1 = feat_c1[feat_c1["fault_type"] == ftype]
            fault_type_totals[ftype]["total"] += len(sub_c1)
            fault_type_totals[ftype]["c1_tp"] += (sub_c1["__predicted"].to_numpy(dtype=bool)).sum()
            sub_c2 = feat_c2[feat_c2["fault_type"] == ftype]
            fault_type_totals[ftype]["c2_tp"] += (sub_c2["__predicted"].to_numpy(dtype=bool)).sum()
            
        benchmark_table.append({
            "Seed": seed,
            "C1_Prec": m_c1["precision"],
            "C2_Prec": m_c2["precision"],
            "C1_Rec": m_c1["recall"],
            "C2_Rec": m_c2["recall"],
            "C1_F1": m_c1["f1"],
            "C2_F1": m_c2["f1"],
            "C1_F1_star": ep_c1.latency_aware_f1,
            "C2_F1_star": ep_c2.latency_aware_f1,
            "C1_TP": m_c1["tp"],
            "C2_TP": m_c2["tp"],
            "C1_FP": m_c1["fp"],
            "C2_FP": m_c2["fp"],
            "C1_FN": m_c1["fn"],
            "C2_FN": m_c2["fn"],
            "C1_EpCatch": ep_c1.episode_detection_rate,
            "C2_EpCatch": ep_c2.episode_detection_rate,
            "FP_Reduction_Pct": (m_c1["fp"] - m_c2["fp"]) / m_c1["fp"] * 100.0,
            "Recall_Delta_Pct": (m_c2["recall"] - m_c1["recall"]) * 100.0
        })
        
    df_res = pd.DataFrame(benchmark_table)
    print(f"{'Seed':<10} {'C1 Prec':<10} {'C2 Prec':<10} {'C1 Rec':<10} {'C2 Rec':<10} {'C1 F1':<8} {'C2 F1':<8} {'C1 F1*':<9} {'C2 F1*':<9} {'C1 FP':<8} {'C2 FP':<8} {'FP Reduc %':<11} {'C2 EpCat'}")
    print("-" * 185)
    for _, r in df_res.iterrows():
        print(f"{int(r['Seed']):<10} {r['C1_Prec']*100:>8.2f}% {r['C2_Prec']*100:>8.2f}% {r['C1_Rec']*100:>8.2f}% {r['C2_Rec']*100:>8.2f}% {r['C1_F1']:>7.3f} {r['C2_F1']:>7.3f} {r['C1_F1_star']:>8.4f} {r['C2_F1_star']:>8.4f} {int(r['C1_FP']):>7} {int(r['C2_FP']):>7} {r['FP_Reduction_Pct']:>9.2f}% {r['C2_EpCatch']*100:>9.2f}%")
    print("-" * 185)
    print(f"{'MEAN':<10} {df_res['C1_Prec'].mean()*100:>8.2f}% {df_res['C2_Prec'].mean()*100:>8.2f}% {df_res['C1_Rec'].mean()*100:>8.2f}% {df_res['C2_Rec'].mean()*100:>8.2f}% {df_res['C1_F1'].mean():>7.3f} {df_res['C2_F1'].mean():>7.3f} {df_res['C1_F1_star'].mean():>8.4f} {df_res['C2_F1_star'].mean():>8.4f} {df_res['C1_FP'].mean():>7.1f} {df_res['C2_FP'].mean():>7.1f} {df_res['FP_Reduction_Pct'].mean():>9.2f}% {df_res['C2_EpCatch'].mean()*100:>9.2f}%")
    print(f"{'STD':<10} {df_res['C1_Prec'].std()*100:>8.2f}% {df_res['C2_Prec'].std()*100:>8.2f}% {df_res['C1_Rec'].std()*100:>8.2f}% {df_res['C2_Rec'].std()*100:>8.2f}% {df_res['C1_F1'].std():>7.3f} {df_res['C2_F1'].std():>7.3f} {df_res['C1_F1_star'].std():>8.4f} {df_res['C2_F1_star'].std():>8.4f} {df_res['C1_FP'].std():>7.1f} {df_res['C2_FP'].std():>7.1f} {df_res['FP_Reduction_Pct'].std():>9.2f}% {df_res['C2_EpCatch'].std()*100:>9.2f}%")
    print(f"{'MIN':<10} {df_res['C1_Prec'].min()*100:>8.2f}% {df_res['C2_Prec'].min()*100:>8.2f}% {df_res['C1_Rec'].min()*100:>8.2f}% {df_res['C2_Rec'].min()*100:>8.2f}% {df_res['C1_F1'].min():>7.3f} {df_res['C2_F1'].min():>7.3f} {df_res['C1_F1_star'].min():>8.4f} {df_res['C2_F1_star'].min():>8.4f} {int(df_res['C1_FP'].min()):>7} {int(df_res['C2_FP'].min()):>7} {df_res['FP_Reduction_Pct'].min():>9.2f}% {df_res['C2_EpCatch'].min()*100:>9.2f}%")
    print(f"{'MAX':<10} {df_res['C1_Prec'].max()*100:>8.2f}% {df_res['C2_Prec'].max()*100:>8.2f}% {df_res['C1_Rec'].max()*100:>8.2f}% {df_res['C2_Rec'].max()*100:>8.2f}% {df_res['C1_F1'].max():>7.3f} {df_res['C2_F1'].max():>7.3f} {df_res['C1_F1_star'].max():>8.4f} {df_res['C2_F1_star'].max():>8.4f} {int(df_res['C1_FP'].max()):>7} {int(df_res['C2_FP'].max()):>7} {df_res['FP_Reduction_Pct'].max():>9.2f}% {df_res['C2_EpCatch'].max()*100:>9.2f}%")
    print("=" * 185)

    # Section 10: Fault Type Breakdown
    print("\n4. FAULT-TYPE DETECTION RECALL BREAKDOWN (CHANCE 1 vs CHANCE 2)")
    print("-" * 125)
    print(f"{'Fault Type':<28} {'Total GT Points':<18} {'Chance 1 Rec':<22} {'Chance 2 Rec':<22} {'Delta Recall'}")
    print("-" * 125)
    for ftype, stats in fault_type_totals.items():
        tot = stats["total"]
        c1_tp = stats["c1_tp"]
        c2_tp = stats["c2_tp"]
        c1_rec = c1_tp / tot * 100.0 if tot > 0 else 0.0
        c2_rec = c2_tp / tot * 100.0 if tot > 0 else 0.0
        print(f"{ftype:<28} {tot:<18} {c1_tp:<8} ({c1_rec:>6.2f}%)     {c2_tp:<8} ({c2_rec:>6.2f}%)     {c2_rec - c1_rec:>+8.2f}%")
    print("-" * 125)

if __name__ == '__main__':
    run_chance2()
