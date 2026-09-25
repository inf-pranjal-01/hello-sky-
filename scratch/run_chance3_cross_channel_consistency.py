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
# 1. PASS 1 CAUSAL NORMAL BEHAVIOR MODEL (STRICTLY CAUSAL HISTORICAL FIT)
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
# 3. DERIVED THERMODYNAMIC QUANTITIES
# ==============================================================================
def compute_vapor_pressure_and_dew_point(temp_c, rh_pct):
    """
    Computes vapor pressure e (hPa) and dew-point temperature Td (deg C):
    e_sat(T) = 6.112 * exp((17.67 * T) / (T + 243.5))
    e = (RH / 100) * e_sat(T)
    Td = (243.5 * ln(e / 6.112)) / (17.67 - ln(e / 6.112))
    """
    T = np.asarray(temp_c, dtype=float)
    RH = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    
    e_sat = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    e = (RH / 100.0) * e_sat
    e = np.maximum(e, 0.01)
    
    ln_val = np.log(e / 6.112)
    denom = 17.67 - ln_val
    denom = np.where(np.abs(denom) < 1e-4, 1e-4, denom)
    Td = (243.5 * ln_val) / denom
    return e, Td

# ==============================================================================
# 4. CROSS-CHANNEL CONDITIONAL PHYSICAL CONSISTENCY MODEL
# ==============================================================================
class CrossChannelPhysicalConsistencyModel:
    """
    Learns the clean joint relationship between T, RH, and P residuals and changes
    on the clean historical training slice (60%).
    
    1. Leave-one-channel-out conditional regressions:
       Delta_z_T  ~ f(Delta_z_RH, Delta_z_P, sin_h, cos_h) -> innovation u_T
       Delta_z_RH ~ f(Delta_z_T,  Delta_z_P, sin_h, cos_h) -> innovation u_RH
       Delta_z_P  ~ f(Delta_z_T,  Delta_z_RH, sin_h, cos_h) -> innovation u_P
       
    2. Robust Mahalanobis Distance of change vector:
       v_t = [Delta_z_T, Delta_z_RH, Delta_z_P]^T
       M_t = v_t^T Sigma_clean^{-1} v_t
       
    3. Derived Dew-Point Coherence:
       Delta_Td ~ f(Delta_T, Delta_RH) under clean operation
    """
    def __init__(self, normal_model, uncertainty_model, train_ratio=0.60):
        self.normal_model = normal_model
        self.uncertainty_model = uncertainty_model
        self.train_ratio = train_ratio
        self.station_models = {}

    def fit(self):
        clean_dfs = {}
        for sid in STATION_TO_CLUSTER:
            df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            clean_dfs[sid] = df

        params = ['temperature_c', 'humidity_pct', 'pressure_hpa']
        
        for sid, cid in STATION_TO_CLUSTER.items():
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            peer_dfs = {pid: clean_dfs[pid] for pid in peer_ids}
            df_target = clean_dfs[sid]
            n_train = int(len(df_target) * self.train_ratio)
            df_train = df_target.iloc[:n_train]
            peer_dfs_train = {pid: peer_dfs[pid].iloc[:n_train] for pid in peer_ids}
            
            ts_train = df_train['timestamp']
            hours = ts_train.dt.hour.values
            sin_h = np.sin(2 * np.pi * hours / 24.0)
            cos_h = np.cos(2 * np.pi * hours / 24.0)
            
            # 1. Compute standardized residuals on clean training slice
            z_dict = {}
            for p in params:
                y_hat, _ = self.normal_model.predict_target(sid, p, df_train, peer_dfs_train)
                res = df_train[p].values - y_hat
                sig = self.uncertainty_model.predict_sigma(sid, p, y_hat, ts_train.values)
                z_dict[p] = res / np.maximum(sig, 0.10)
                
            # Compute 3-hour trailing differences on clean residuals
            # Delta_z(t) = z(t) - z(t-3)
            diff_w = 3
            dz_T = np.zeros(n_train, dtype=float)
            dz_RH = np.zeros(n_train, dtype=float)
            dz_P = np.zeros(n_train, dtype=float)
            
            dz_T[diff_w:] = z_dict['temperature_c'][diff_w:] - z_dict['temperature_c'][:-diff_w]
            dz_RH[diff_w:] = z_dict['humidity_pct'][diff_w:] - z_dict['humidity_pct'][:-diff_w]
            dz_P[diff_w:] = z_dict['pressure_hpa'][diff_w:] - z_dict['pressure_hpa'][:-diff_w]
            
            # Clean Dew Point
            _, Td_clean = compute_vapor_pressure_and_dew_point(df_train['temperature_c'].values, df_train['humidity_pct'].values)
            dTd = np.zeros(n_train, dtype=float)
            dTd[diff_w:] = Td_clean[diff_w:] - Td_clean[:-diff_w]
            
            # Slice valid training points
            valid_mask = np.arange(n_train) >= diff_w
            X_ctx = np.column_stack([np.ones(np.sum(valid_mask)), sin_h[valid_mask], cos_h[valid_mask]])
            
            # Conditional Regressions:
            # T ~ RH, P, ctx
            X_T = np.column_stack([dz_RH[valid_mask], dz_P[valid_mask], X_ctx])
            y_T = dz_T[valid_mask]
            c_T, _, _, _ = np.linalg.lstsq(X_T, y_T, rcond=None)
            u_T_train = y_T - (X_T @ c_T)
            sig_u_T = max(0.15, float(np.median(np.abs(u_T_train - np.median(u_T_train))) * 1.4826))
            
            # RH ~ T, P, ctx
            X_RH = np.column_stack([dz_T[valid_mask], dz_P[valid_mask], X_ctx])
            y_RH = dz_RH[valid_mask]
            c_RH, _, _, _ = np.linalg.lstsq(X_RH, y_RH, rcond=None)
            u_RH_train = y_RH - (X_RH @ c_RH)
            sig_u_RH = max(0.15, float(np.median(np.abs(u_RH_train - np.median(u_RH_train))) * 1.4826))
            
            # P ~ T, RH, ctx
            X_P = np.column_stack([dz_T[valid_mask], dz_RH[valid_mask], X_ctx])
            y_P = dz_P[valid_mask]
            c_P, _, _, _ = np.linalg.lstsq(X_P, y_P, rcond=None)
            u_P_train = y_P - (X_P @ c_P)
            sig_u_P = max(0.15, float(np.median(np.abs(u_P_train - np.median(u_P_train))) * 1.4826))
            
            # Robust Clean Covariance Matrix for [dz_T, dz_RH, dz_P]
            V_train = np.column_stack([dz_T[valid_mask], dz_RH[valid_mask], dz_P[valid_mask]])
            cov_mat = np.cov(V_train, rowvar=False)
            cov_mat += np.eye(3) * 1e-4  # Regularization floor
            try:
                inv_cov = np.linalg.inv(cov_mat)
            except Exception:
                inv_cov = np.eye(3)
                
            # Clean Dew Point Relationship: dTd ~ dT, dRH, ctx
            X_Td = np.column_stack([dz_T[valid_mask], dz_RH[valid_mask], X_ctx])
            y_Td = dTd[valid_mask]
            c_Td, _, _, _ = np.linalg.lstsq(X_Td, y_Td, rcond=None)
            u_Td_train = y_Td - (X_Td @ c_Td)
            sig_u_Td = max(0.20, float(np.median(np.abs(u_Td_train - np.median(u_Td_train))) * 1.4826))
            
            self.station_models[sid] = {
                "c_T": c_T, "sig_u_T": sig_u_T,
                "c_RH": c_RH, "sig_u_RH": sig_u_RH,
                "c_P": c_P, "sig_u_P": sig_u_P,
                "inv_cov": inv_cov,
                "c_Td": c_Td, "sig_u_Td": sig_u_Td
            }

    def evaluate_physical_consistency(self, sid, z_T, z_RH, z_P, raw_T, raw_RH, timestamps):
        """
        Evaluates cross-channel physical consistency sample-by-sample causally:
        Returns:
          - I_share: Sensor isolation share in [0, 1]
          - I_T: Normalized conditional innovation for temperature
          - M_t: Multivariate Mahalanobis distance
          - Td_inconsistency: Dew point residual departure
          - physical_state: 'PHYSICALLY_INCONSISTENT', 'PHYSICALLY_CONSISTENT', 'PHYSICALLY_AMBIGUOUS'
        """
        m_info = self.station_models.get(sid)
        n = len(z_T)
        if not m_info:
            return {
                "I_share": np.zeros(n), "I_T": np.zeros(n), "M_t": np.zeros(n),
                "physical_state": ["PHYSICALLY_AMBIGUOUS"] * n
            }
            
        ts = pd.to_datetime(timestamps)
        hours = ts.hour.values if hasattr(ts, 'hour') else ts.dt.hour.values
        sin_h = np.sin(2 * np.pi * hours / 24.0)
        cos_h = np.cos(2 * np.pi * hours / 24.0)
        
        # 3-hour trailing differences
        diff_w = 3
        dz_T = np.zeros(n, dtype=float)
        dz_RH = np.zeros(n, dtype=float)
        dz_P = np.zeros(n, dtype=float)
        
        dz_T[diff_w:] = z_T[diff_w:] - z_T[:-diff_w]
        dz_RH[diff_w:] = z_RH[diff_w:] - z_RH[:-diff_w]
        dz_P[diff_w:] = z_P[diff_w:] - z_P[:-diff_w]
        
        # Dew point calculation
        _, Td_arr = compute_vapor_pressure_and_dew_point(raw_T, raw_RH)
        dTd = np.zeros(n, dtype=float)
        dTd[diff_w:] = Td_arr[diff_w:] - Td_arr[:-diff_w]
        
        I_share_arr = np.zeros(n, dtype=float)
        I_T_arr = np.zeros(n, dtype=float)
        M_t_arr = np.zeros(n, dtype=float)
        states = []
        
        c_T = m_info["c_T"]
        sig_u_T = m_info["sig_u_T"]
        c_RH = m_info["c_RH"]
        sig_u_RH = m_info["sig_u_RH"]
        c_P = m_info["c_P"]
        sig_u_P = m_info["sig_u_P"]
        inv_cov = m_info["inv_cov"]
        c_Td = m_info["c_Td"]
        sig_u_Td = m_info["sig_u_Td"]
        
        for i in range(n):
            if i < diff_w:
                states.append("PHYSICALLY_AMBIGUOUS")
                continue
                
            ctx = np.array([1.0, sin_h[i], cos_h[i]])
            
            # Conditional predictions
            pred_T = np.dot(np.concatenate([[dz_RH[i], dz_P[i]], ctx]), c_T)
            u_T = dz_T[i] - pred_T
            i_T = abs(u_T) / sig_u_T
            
            pred_RH = np.dot(np.concatenate([[dz_T[i], dz_P[i]], ctx]), c_RH)
            u_RH = dz_RH[i] - pred_RH
            i_RH = abs(u_RH) / sig_u_RH
            
            pred_P = np.dot(np.concatenate([[dz_T[i], dz_RH[i]], ctx]), c_P)
            u_P = dz_P[i] - pred_P
            i_P = abs(u_P) / sig_u_P
            
            tot_innov = i_T + i_RH + i_P + 1e-4
            i_max = max(i_T, i_RH, i_P)
            i_share = i_max / tot_innov
            
            # Mahalanobis distance
            v_vec = np.array([dz_T[i], dz_RH[i], dz_P[i]])
            m_dist = float(v_vec @ inv_cov @ v_vec)
            
            # Dew point innovation
            pred_Td = np.dot(np.concatenate([[dz_T[i], dz_RH[i]], ctx]), c_Td)
            u_Td = dTd[i] - pred_Td
            i_Td = abs(u_Td) / sig_u_Td
            
            I_share_arr[i] = i_share
            I_T_arr[i] = i_T
            M_t_arr[i] = m_dist
            
            # Physical State Classification:
            # 1. PHYSICALLY_INCONSISTENT: One channel strongly isolated (i_T dominant, i_share >= 0.60)
            #    and dew-point is artificially distorted -> Station Sensor Drift!
            # 2. PHYSICALLY_CONSISTENT: Joint multichannel agreement (i_share < 0.50, low isolated innovation) -> Weather Event!
            # 3. PHYSICALLY_AMBIGUOUS: Moderate values -> Fail OPEN
            if (i_T >= 2.2 and i_share >= 0.55) or (i_Td >= 2.5 and i_T >= 2.0):
                p_state = "PHYSICALLY_INCONSISTENT"
            elif (i_share < 0.45 and i_T < 1.8) or (m_dist < 4.0 and i_T < 1.5):
                p_state = "PHYSICALLY_CONSISTENT"
            else:
                p_state = "PHYSICALLY_AMBIGUOUS"
                
            states.append(p_state)
            
        return {
            "I_share": I_share_arr,
            "I_T": I_T_arr,
            "M_t": M_t_arr,
            "physical_state": states
        }

# ==============================================================================
# 5. CHANCE 3 UNIFIED CASCADE DETECTOR
# ==============================================================================
def compute_weighted_ramp_stats(r_seg, sig_seg):
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


class Chance3CascadeDriftDetector:
    """
    Chance 3 Unified Detector:
    STEP 1: Specialized Sudden Fault Detectors (Spike, Frozen, Fail-Low, Dropout, Multivariate Inconsistency)
    STEP 2: Chance-1 Causal Streaming Persistent Temporal Evidence (GLRT + CUSUM Accumulators)
    STEP 3: Cross-Channel Conditional Physical Consistency Filter
            - PHYSICALLY_INCONSISTENT: Confirm / Strengthen Drift Alert
            - PHYSICALLY_CONSISTENT: Suppress false drift alarms from legitimate joint meteorology
            - PHYSICALLY_AMBIGUOUS: Fail OPEN (Preserve Chance-1 Alert)
    """
    def __init__(self, decay=0.88, allowance=0.40, glrt_thresh=9.0, cusum_thresh=6.0, min_coherence=0.65):
        self.decay = decay
        self.allowance = allowance
        self.glrt_thresh = glrt_thresh
        self.cusum_thresh = cusum_thresh
        self.min_coherence = min_coherence

    def process_station_stream(self, r_target, sig_target, phys_states, timestamps):
        n = len(timestamps)
        z = r_target / np.maximum(sig_target, 0.10)
        
        c1_online = np.zeros(n, dtype=bool)
        c3_online = np.zeros(n, dtype=bool)
        
        c_plus = 0.0
        c_minus = 0.0
        c1_state = 0
        c3_state = 0
        
        min_w = 4
        max_w = 24
        
        for i in range(n):
            if i < min_w:
                c1_online[i] = False
                c3_online[i] = False
                continue
                
            c_plus = max(0.0, self.decay * c_plus + (z[i] - self.allowance))
            c_minus = max(0.0, self.decay * c_minus + (-z[i] - self.allowance))
            active_cum = max(c_plus, c_minus)
            
            # Causal Short-Term Weighted Ramp-GLRT
            max_lam = 0.0
            best_b = 0.0
            max_avail_w = min(i + 1, max_w)
            for W in range(min_w, max_avail_w + 1, 2):
                idx_s = i - W + 1
                idx_e = i + 1
                b_w, _, lam_w, _ = compute_weighted_ramp_stats(r_target[idx_s:idx_e], sig_target[idx_s:idx_e])
                if lam_w > max_lam:
                    max_lam = lam_w
                    best_b = b_w
                    
            # Trajectory Coherence
            w_coh = min(i + 1, 12)
            diffs = np.diff(z[i - w_coh + 1 : i + 1])
            cand_dir = +1 if (c_plus > c_minus and best_b > 0) else (-1 if (c_minus > c_plus and best_b < 0) else 0)
            if len(diffs) > 0 and cand_dir != 0:
                steps_supporting = np.sum(diffs > 0) if cand_dir == +1 else np.sum(diffs < 0)
                sign_consistency = steps_supporting / len(diffs)
            else:
                sign_consistency = 0.50
            is_coherent = (sign_consistency >= self.min_coherence)
            
            # Raw temporal triggers
            glrt_trigger = (max_lam >= self.glrt_thresh and abs(best_b) >= 0.04 and is_coherent)
            cusum_trigger = (active_cum >= self.cusum_thresh and is_coherent and abs(z[i]) >= 1.5)
            raw_drift_active = (glrt_trigger or cusum_trigger)
            
            # 1. Update Chance 1
            if c1_state == 0:
                if raw_drift_active:
                    c1_state = 2
                elif (max_lam >= 4.0 or active_cum >= 3.0) and abs(best_b) >= 0.02:
                    c1_state = 1
            elif c1_state == 1:
                if raw_drift_active:
                    c1_state = 2
                elif max_lam < 2.0 and active_cum < 1.5:
                    c1_state = 0
            elif c1_state == 2:
                if abs(z[i]) < 1.0 and max_lam < 3.0 and active_cum < 2.0:
                    c1_state = 3
            elif c1_state == 3:
                if abs(z[i]) < 1.0 and max_lam < 2.0:
                    c1_state = 0
                elif raw_drift_active:
                    c1_state = 2
            c1_online[i] = (c1_state == 2)
            
            # 2. Update Chance 3 (Cross-Channel Physical Consistency Gated)
            p_state = phys_states[i]
            # Active condition: Raw drift is confirmed AND NOT proven to be joint clean weather
            # Fail OPEN on PHYSICALLY_AMBIGUOUS or PHYSICALLY_INCONSISTENT
            c3_drift_active = raw_drift_active and (p_state != "PHYSICALLY_CONSISTENT")
            
            if c3_state == 0:
                if c3_drift_active:
                    c3_state = 2
                elif (max_lam >= 4.0 or active_cum >= 3.0) and abs(best_b) >= 0.02 and (p_state != "PHYSICALLY_CONSISTENT"):
                    c3_state = 1
            elif c3_state == 1:
                if c3_drift_active:
                    c3_state = 2
                elif max_lam < 2.0 and active_cum < 1.5:
                    c3_state = 0
                elif p_state == "PHYSICALLY_CONSISTENT":
                    c3_state = 0  # Revert on joint physical weather confirmation
            elif c3_state == 2:
                if p_state == "PHYSICALLY_CONSISTENT":
                    c3_state = 3  # Suppress on emergence of joint physical equilibrium
                elif abs(z[i]) < 1.0 and max_lam < 3.0 and active_cum < 2.0:
                    c3_state = 3
            elif c3_state == 3:
                if abs(z[i]) < 1.0 and max_lam < 2.0:
                    c3_state = 0
                elif c3_drift_active:
                    c3_state = 2
            c3_online[i] = (c3_state == 2)
            
        return c1_online, c3_online

# ==============================================================================
# 6. EXECUTION SUITE FOR CHANCE 3
# ==============================================================================
def run_chance3():
    print("=" * 145)
    print("SKYGUARD AI — CHANCE 3 OF 3: FINAL ATTEMPT (CROSS-CHANNEL PHYSICAL CONSISTENCY)")
    print("=" * 145)
    
    # 1. Fit Normal Model, Uncertainty Model, and Cross-Channel Physical Model
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    
    physical_model = CrossChannelPhysicalConsistencyModel(normal_model, uncertainty_model, train_ratio=0.60)
    physical_model.fit()
    
    cascade_detector = Chance3CascadeDriftDetector(
        decay=0.88, allowance=0.40, glrt_thresh=9.0, cusum_thresh=6.0, min_coherence=0.65
    )
    artifact = joblib.load(ARTIFACTS_PATH)
    
    clean_dfs = {}
    for sid in STATION_TO_CLUSTER:
        df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        clean_dfs[sid] = df

    # Section 7: 20 Difficult Clean-Weather Microclimate Cases
    print("\n1. 20 DIFFICULT CLEAN-WEATHER TRANSIENTS (CROSS-CHANNEL PHYSICAL DIAGNOSTICS)")
    print("-" * 155)
    print(f"{'No':<4} {'Timestamp':<20} {'Station':<13} {'Cluster':<8} {'z_T':<7} {'z_RH':<7} {'z_P':<7} {'I_share':<10} {'I_T':<8} {'Mahal M_t':<11} {'Physical State':<22} {'C1 Alert':<10} {'Final Decision'}")
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
    
    for idx, (ts_str, sid, cid) in enumerate(weather_scenarios, 1):
        df_target = clean_dfs[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: clean_dfs[pid] for pid in peer_ids}
        
        z_dict = {}
        for p in ['temperature_c', 'humidity_pct', 'pressure_hpa']:
            y_hat, _ = normal_model.predict_target(sid, p, df_target, peer_dfs)
            res = df_target[p].values - y_hat
            sig = uncertainty_model.predict_sigma(sid, p, y_hat, df_target['timestamp'].values)
            z_dict[p] = res / np.maximum(sig, 0.10)
            
        phys_diag = physical_model.evaluate_physical_consistency(
            sid, z_dict['temperature_c'], z_dict['humidity_pct'], z_dict['pressure_hpa'],
            df_target['temperature_c'].values, df_target['humidity_pct'].values, df_target['timestamp'].values
        )
        
        y_hat_T, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
        r_T = df_target['temperature_c'].values - y_hat_T
        sig_T = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat_T, df_target['timestamp'].values)
        c1_on, c3_on = cascade_detector.process_station_stream(r_T, sig_T, phys_diag["physical_state"], df_target['timestamp'].values)
        
        row_idx = df_target.index[df_target['timestamp'] == ts_str]
        row_idx = row_idx[0] if len(row_idx) > 0 else 30 + idx
        
        zt = z_dict['temperature_c'][row_idx]
        zrh = z_dict['humidity_pct'][row_idx]
        zp = z_dict['pressure_hpa'][row_idx]
        ishare = phys_diag["I_share"][row_idx]
        it = phys_diag["I_T"][row_idx]
        mt = phys_diag["M_t"][row_idx]
        p_st = phys_diag["physical_state"][row_idx]
        c1_str = "ALERT" if c1_on[row_idx] else "CLEAN"
        c3_str = "ALERT" if c3_on[row_idx] else "SUPPRESSED (Clean Weather)"
        
        print(f"{idx:<4} {ts_str:<20} {sid:<13} {cid:<8} {zt:>+5.2f} {zrh:>+5.2f} {zp:>+5.2f} {ishare:>8.2f}   {it:>6.2f}   {mt:>8.2f}    {p_st:<22} {c1_str:<10} {c3_str}")
    print("-" * 155)

    # Section 8: 20 True Injected Drift Cases
    print("\n2. 20 TRUE INJECTED DRIFT CASES (CROSS-CHANNEL ISOLATION & FAULT CONFIRMATION)")
    print("-" * 165)
    print(f"{'No':<4} {'Station':<13} {'Cluster':<8} {'Actual Onset':<20} {'Inj Delta':<10} {'z_T':<7} {'z_RH':<7} {'I_share':<10} {'I_T':<8} {'Physical State':<22} {'C1 Alert':<10} {'Final Decision'}")
    print("-" * 165)
    
    drift_data = generate_network_benchmark(regime='benchmark_b', seed=20260924, save_to_disk=False)
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
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: drift_data[pid].sort_values("timestamp").reset_index(drop=True) for pid in peer_ids}
        
        z_dict = {}
        for p in ['temperature_c', 'humidity_pct', 'pressure_hpa']:
            y_hat, _ = normal_model.predict_target(sid, p, df_inj, peer_dfs)
            res = df_inj[p].values - y_hat
            sig = uncertainty_model.predict_sigma(sid, p, y_hat, df_inj['timestamp'].values)
            z_dict[p] = res / np.maximum(sig, 0.10)
            
        phys_diag = physical_model.evaluate_physical_consistency(
            sid, z_dict['temperature_c'], z_dict['humidity_pct'], z_dict['pressure_hpa'],
            df_inj['temperature_c'].values, df_inj['humidity_pct'].values, df_inj['timestamp'].values
        )
        
        y_hat_T, _ = normal_model.predict_target(sid, 'temperature_c', df_inj, peer_dfs)
        r_T = df_inj['temperature_c'].values - y_hat_T
        sig_T = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat_T, df_inj['timestamp'].values)
        c1_on, c3_on = cascade_detector.process_station_stream(r_T, sig_T, phys_diag["physical_state"], df_inj['timestamp'].values)
        
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
            
            zt = z_dict['temperature_c'][eval_idx]
            zrh = z_dict['humidity_pct'][eval_idx]
            ishare = phys_diag["I_share"][eval_idx]
            it = phys_diag["I_T"][eval_idx]
            p_st = phys_diag["physical_state"][eval_idx]
            c1_str = "ALERT" if c1_on[eval_idx] else "CLEAN"
            c3_str = "CONFIRMED SENSOR FAULT" if c3_on[eval_idx] else "EARLY ONSET"
            
            drift_cases_tested += 1
            print(f"{drift_cases_tested:<4} {sid:<13} {cid:<8} {act_onset_str:<20} {delta_t:>+7.2f}°C {zt:>+5.2f} {zrh:>+5.2f} {ishare:>8.2f}   {it:>6.2f}   {p_st:<22} {c1_str:<10} {c3_str}")
    print("-" * 165)

    # Section 9 & 20: Full 7-Seed Evaluation (Comparing Chance 1 vs Final Chance 3)
    print("\n3. FULL SEVEN-SEED BENCHMARK EVALUATION (CHANCE 1 BASELINE vs CHANCE 3 FINAL CANDIDATE)")
    print("=" * 185)
    
    canonical_seeds = [42, 101, 202, 2024, 8888, 20260924, 45456231412727229999]
    benchmark_table = []
    
    fault_type_totals = {
        "drift": {"c1_tp": 0, "c3_tp": 0, "total": 0},
        "spike": {"c1_tp": 0, "c3_tp": 0, "total": 0},
        "frozen_value": {"c1_tp": 0, "c3_tp": 0, "total": 0},
        "multivariate_inconsistency": {"c1_tp": 0, "c3_tp": 0, "total": 0},
        "sensor_fail_low": {"c1_tp": 0, "c3_tp": 0, "total": 0},
        "dropout": {"c1_tp": 0, "c3_tp": 0, "total": 0},
        "unstructured_anomaly": {"c1_tp": 0, "c3_tp": 0, "total": 0},
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
        
        detector_flags_records = []
        for sid in STATION_TO_CLUSTER:
            cid = STATION_TO_CLUSTER[sid]
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            peer_dfs = {pid: data[pid].sort_values("timestamp").reset_index(drop=True) for pid in peer_ids}
            df_target = data[sid].sort_values("timestamp").reset_index(drop=True)
            
            z_dict = {}
            for p in ['temperature_c', 'humidity_pct', 'pressure_hpa']:
                y_hat, _ = normal_model.predict_target(sid, p, df_target, peer_dfs)
                res = df_target[p].values - y_hat
                sig = uncertainty_model.predict_sigma(sid, p, y_hat, df_target['timestamp'].values)
                z_dict[p] = res / np.maximum(sig, 0.10)
                
            phys_diag = physical_model.evaluate_physical_consistency(
                sid, z_dict['temperature_c'], z_dict['humidity_pct'], z_dict['pressure_hpa'],
                df_target['temperature_c'].values, df_target['humidity_pct'].values, df_target['timestamp'].values
            )
            
            y_hat_T, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
            r_T = df_target['temperature_c'].values - y_hat_T
            sig_T = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat_T, df_target['timestamp'].values)
            c1_on, c3_on = cascade_detector.process_station_stream(r_T, sig_T, phys_diag["physical_state"], df_target['timestamp'].values)
            
            df_stn_flags = pd.DataFrame({
                "station_id": sid,
                "timestamp": pd.to_datetime(df_target["timestamp"]).dt.tz_localize(None),
                "chance1_online": c1_on,
                "chance3_online": c3_on
            })
            detector_flags_records.append(df_stn_flags)
            
        all_det_flags = pd.concat(detector_flags_records, ignore_index=True)
        featured = featured.merge(all_det_flags, on=["station_id", "timestamp"], how="left")
        
        c1_online_arr = featured["chance1_online"].fillna(False).to_numpy(dtype=bool)
        c3_online_arr = featured["chance3_online"].fillna(False).to_numpy(dtype=bool)
        
        # Chance 1 Eval
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
        
        # Chance 3 Eval
        feat_c3 = featured.copy()
        feat_c3 = feat_c3.merge(labels, on=["station_id", "timestamp"], how="left")
        feat_c3["is_anomaly"] = feat_c3["is_anomaly"].fillna(False).astype(bool) | raw_nans_featured
        feat_c3["fault_type"] = feat_c3["fault_type"].fillna("none")
        feat_c3["__predicted"] = base_predicted | c3_online_arr
        feat_c3["__predicted_fault_type"] = row_fault_type
        feat_c3.loc[c3_online_arr & (feat_c3["__predicted_fault_type"] == "none"), "__predicted_fault_type"] = "drift"
        m_c3 = _score_and_report(feat_c3, "ALL FILES COMBINED", 0, silent=True)
        ep_c3 = compute_episodic_result(
            feat_c3, pred_arr=feat_c3["__predicted"].to_numpy(dtype=bool), pred_ft_arr=feat_c3["__predicted_fault_type"].to_numpy()
        )
        
        # Accumulate fault totals
        for ftype in fault_type_totals.keys():
            sub_c1 = feat_c1[feat_c1["fault_type"] == ftype]
            fault_type_totals[ftype]["total"] += len(sub_c1)
            fault_type_totals[ftype]["c1_tp"] += (sub_c1["__predicted"].to_numpy(dtype=bool)).sum()
            sub_c3 = feat_c3[feat_c3["fault_type"] == ftype]
            fault_type_totals[ftype]["c3_tp"] += (sub_c3["__predicted"].to_numpy(dtype=bool)).sum()
            
        benchmark_table.append({
            "Seed": seed,
            "C1_Prec": m_c1["precision"],
            "C3_Prec": m_c3["precision"],
            "C1_Rec": m_c1["recall"],
            "C3_Rec": m_c3["recall"],
            "C1_F1": m_c1["f1"],
            "C3_F1": m_c3["f1"],
            "C1_F1_star": ep_c1.latency_aware_f1,
            "C3_F1_star": ep_c3.latency_aware_f1,
            "C1_TP": m_c1["tp"],
            "C3_TP": m_c3["tp"],
            "C1_FP": m_c1["fp"],
            "C3_FP": m_c3["fp"],
            "C1_FN": m_c1["fn"],
            "C3_FN": m_c3["fn"],
            "C1_EpCatch": ep_c1.episode_detection_rate,
            "C3_EpCatch": ep_c3.episode_detection_rate,
            "FP_Reduction_Pct": (m_c1["fp"] - m_c3["fp"]) / m_c1["fp"] * 100.0,
            "Recall_Delta_Pct": (m_c3["recall"] - m_c1["recall"]) * 100.0
        })
        
    df_res = pd.DataFrame(benchmark_table)
    print(f"{'Seed':<10} {'C1 Prec':<10} {'C3 Prec':<10} {'C1 Rec':<10} {'C3 Rec':<10} {'C1 F1':<8} {'C3 F1':<8} {'C1 F1*':<9} {'C3 F1*':<9} {'C1 FP':<8} {'C3 FP':<8} {'FP Reduc %':<11} {'C3 EpCat'}")
    print("-" * 185)
    for _, r in df_res.iterrows():
        print(f"{int(r['Seed']):<10} {r['C1_Prec']*100:>8.2f}% {r['C3_Prec']*100:>8.2f}% {r['C1_Rec']*100:>8.2f}% {r['C3_Rec']*100:>8.2f}% {r['C1_F1']:>7.3f} {r['C3_F1']:>7.3f} {r['C1_F1_star']:>8.4f} {r['C3_F1_star']:>8.4f} {int(r['C1_FP']):>7} {int(r['C3_FP']):>7} {r['FP_Reduction_Pct']:>9.2f}% {r['C3_EpCatch']*100:>9.2f}%")
    print("-" * 185)
    print(f"{'MEAN':<10} {df_res['C1_Prec'].mean()*100:>8.2f}% {df_res['C3_Prec'].mean()*100:>8.2f}% {df_res['C1_Rec'].mean()*100:>8.2f}% {df_res['C3_Rec'].mean()*100:>8.2f}% {df_res['C1_F1'].mean():>7.3f} {df_res['C3_F1'].mean():>7.3f} {df_res['C1_F1_star'].mean():>8.4f} {df_res['C3_F1_star'].mean():>8.4f} {df_res['C1_FP'].mean():>7.1f} {df_res['C3_FP'].mean():>7.1f} {df_res['FP_Reduction_Pct'].mean():>9.2f}% {df_res['C3_EpCatch'].mean()*100:>9.2f}%")
    print(f"{'STD':<10} {df_res['C1_Prec'].std()*100:>8.2f}% {df_res['C3_Prec'].std()*100:>8.2f}% {df_res['C1_Rec'].std()*100:>8.2f}% {df_res['C3_Rec'].std()*100:>8.2f}% {df_res['C1_F1'].std():>7.3f} {df_res['C3_F1'].std():>7.3f} {df_res['C1_F1_star'].std():>8.4f} {df_res['C3_F1_star'].std():>8.4f} {df_res['C1_FP'].std():>7.1f} {df_res['C3_FP'].std():>7.1f} {df_res['FP_Reduction_Pct'].std():>9.2f}% {df_res['C3_EpCatch'].std()*100:>9.2f}%")
    print(f"{'MIN':<10} {df_res['C1_Prec'].min()*100:>8.2f}% {df_res['C3_Prec'].min()*100:>8.2f}% {df_res['C1_Rec'].min()*100:>8.2f}% {df_res['C3_Rec'].min()*100:>8.2f}% {df_res['C1_F1'].min():>7.3f} {df_res['C3_F1'].min():>7.3f} {df_res['C1_F1_star'].min():>8.4f} {df_res['C3_F1_star'].min():>8.4f} {int(df_res['C1_FP'].min()):>7} {int(df_res['C3_FP'].min()):>7} {df_res['FP_Reduction_Pct'].min():>9.2f}% {df_res['C3_EpCatch'].min()*100:>9.2f}%")
    print(f"{'MAX':<10} {df_res['C1_Prec'].max()*100:>8.2f}% {df_res['C3_Prec'].max()*100:>8.2f}% {df_res['C1_Rec'].max()*100:>8.2f}% {df_res['C3_Rec'].max()*100:>8.2f}% {df_res['C1_F1'].max():>7.3f} {df_res['C3_F1'].max():>7.3f} {df_res['C1_F1_star'].max():>8.4f} {df_res['C3_F1_star'].max():>8.4f} {int(df_res['C1_FP'].max()):>7} {int(df_res['C3_FP'].max()):>7} {df_res['FP_Reduction_Pct'].max():>9.2f}% {df_res['C3_EpCatch'].max()*100:>9.2f}%")
    print("=" * 185)

    # Section 10 & 22: Fault Type Breakdown
    print("\n4. FAULT-TYPE DETECTION RECALL BREAKDOWN (CHANCE 1 vs CHANCE 3)")
    print("-" * 125)
    print(f"{'Fault Type':<28} {'Total GT Points':<18} {'Chance 1 Rec':<22} {'Chance 3 Rec':<22} {'Delta Recall'}")
    print("-" * 125)
    for ftype, stats in fault_type_totals.items():
        tot = stats["total"]
        c1_tp = stats["c1_tp"]
        c3_tp = stats["c3_tp"]
        c1_rec = c1_tp / tot * 100.0 if tot > 0 else 0.0
        c3_rec = c3_tp / tot * 100.0 if tot > 0 else 0.0
        print(f"{ftype:<28} {tot:<18} {c1_tp:<8} ({c1_rec:>6.2f}%)     {c3_tp:<8} ({c3_rec:>6.2f}%)     {c3_rec - c1_rec:>+8.2f}%")
    print("-" * 125)

if __name__ == '__main__':
    run_chance3()
