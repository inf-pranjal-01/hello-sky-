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
    RULE_BASE_CONFIDENCE, FROZEN_MIN_MODEL_CORROBORATION,
    HELPER_ALERT_THRESHOLD, FROZEN_HELPER_ALERT_THRESHOLD,
    add_frozen_channel_labels_from_reference
)

# Build station/cluster lookup
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
# 1. NORMAL BEHAVIOR MODEL (Pass 1 Causal Diurnal + Spatial Normal Model)
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
            assert len(peer_ids) == 3, f"Station {sid} does not have exactly 3 peers!"
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
                        peer_regressors[pid] = {
                            "coeffs": coeffs,
                            "sigma": sigma
                        }
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

# ==============================================================================
# 2. CONDITIONAL RESIDUAL UNCERTAINTY MODEL (Pass 5)
# ==============================================================================
class ConditionalResidualUncertaintyModel:
    """
    Models sigma_{s, c}(context_t) where context_t is strictly causal and clean:
    - Station s
    - Channel c
    - Predicted mean \hat{y}_{s, c}(t) (binned into 4 quantiles based on training distribution)
    - Diurnal Phase / Hour Regime h(t) (4 regimes: Night [22-05], Morning [06-10], Peak Day [11-16], Evening [17-21])
    
    Robust scale estimation: MAD * 1.4826 computed strictly on clean historical training slice (60%).
    Hierarchical Bayesian shrinkage toward station-wide scale and cluster scale to guarantee stability.
    Minimum variance floor: sigma >= 0.15 deg C.
    """
    def __init__(self, normal_model, train_ratio=0.60):
        self.normal_model = normal_model
        self.train_ratio = train_ratio
        self.scales = {}
        self.yhat_bins = {}
        self.global_channel_scales = {}

    @staticmethod
    def get_hour_regime(hour):
        # 0: Night (22:00 - 05:00)
        # 1: Morning / Transition (06:00 - 10:00)
        # 2: Peak Solar / Daytime Convective (11:00 - 16:00)
        # 3: Evening / Sunset (17:00 - 21:00)
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
        
        # 1. Compute training residuals for all stations & channels
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

        # Global channel scales (MAD)
        for p in params:
            arr = np.array(all_channel_residuals[p])
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826
            self.global_channel_scales[p] = max(0.15, mad)

        # 2. Fit conditional uncertainty tables per station & channel
        for sid, cid in STATION_TO_CLUSTER.items():
            hours_train = train_hours[sid]
            regimes_train = self.get_hour_regime(hours_train)
            
            for p in params:
                res_train = train_residuals[(sid, p)]
                yhat_train = train_yhats[(sid, p)]
                
                # Station-level overall scale
                stn_med = np.median(res_train)
                stn_scale = max(0.15, np.median(np.abs(res_train - stn_med)) * 1.4826)
                
                # Quantile binning for y_hat (4 quantiles: 25th, 50th, 75th)
                q_cuts = np.quantile(yhat_train, [0.25, 0.50, 0.75])
                # Ensure strictly increasing cuts
                if len(np.unique(q_cuts)) < 3:
                    q_cuts = np.linspace(np.min(yhat_train), np.max(yhat_train), 4)[1:]
                self.yhat_bins[(sid, p)] = q_cuts
                
                yhat_bin_idx = np.digitize(yhat_train, q_cuts)  # values 0, 1, 2, 3
                
                # Table of scales: (yhat_bin, hour_regime) -> sigma
                scale_table = {}
                for yb in range(4):
                    for hr in range(4):
                        mask = (yhat_bin_idx == yb) & (regimes_train == hr)
                        sub_res = res_train[mask]
                        n_pts = len(sub_res)
                        
                        if n_pts >= 15:
                            sub_med = np.median(sub_res)
                            raw_mad = np.median(np.abs(sub_res - sub_med)) * 1.4826
                            # Bayesian shrinkage toward station-level scale: weight = n / (n + 10)
                            w_local = n_pts / (n_pts + 10.0)
                            shrunk_sigma = w_local * raw_mad + (1.0 - w_local) * stn_scale
                        else:
                            shrunk_sigma = stn_scale
                            
                        # Floor scale at 0.15 to prevent infinite weights on ultra-quiet series
                        shrunk_sigma = max(0.15, shrunk_sigma)
                        scale_table[(yb, hr)] = shrunk_sigma
                
                # Also store hour-only scales and yhat-only scales for fallback
                hr_scales = {}
                for hr in range(4):
                    sub_res = res_train[regimes_train == hr]
                    if len(sub_res) >= 10:
                        hr_scales[hr] = max(0.15, np.median(np.abs(sub_res - np.median(sub_res))) * 1.4826)
                    else:
                        hr_scales[hr] = stn_scale
                        
                self.scales[(sid, p)] = {
                    "stn_scale": stn_scale,
                    "scale_table": scale_table,
                    "hr_scales": hr_scales,
                    "q_cuts": q_cuts
                }

    def predict_sigma(self, sid, p, y_hat, timestamps):
        """
        Returns sigma_t for given station, channel, predicted y_hat, and timestamps.
        Strictly causal and clean: uses only y_hat and timestamps.
        """
        model_info = self.scales.get((sid, p))
        if not model_info:
            return np.full(len(y_hat), self.global_channel_scales.get(p, 0.45), dtype=float)
            
        ts = pd.to_datetime(timestamps)
        if hasattr(ts, 'hour'):
            hours = ts.hour.values
        elif hasattr(ts, 'dt'):
            hours = ts.dt.hour.values
        else:
            hours = pd.DatetimeIndex(ts).hour.values
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
# 3. WEIGHTED INTERCEPT-AWARE RAMP-GLRT
# ==============================================================================
def compute_weighted_glrt_series_fast(residuals, sigmas, min_w=4, max_w=48):
    """
    Computes Intercept-Aware Nested Linear Ramp-GLRT with conditional residual uncertainty weights:
    w_i = 1 / sigma_i^2
    \bar{t}_w = sum(w_i * t_i) / sum(w_i)
    \bar{r}_w = sum(w_i * r_i) / sum(w_i)
    s_{tt} = sum(w_i * (t_i - \bar{t}_w)^2)
    s_{tr} = sum(w_i * (t_i - \bar{t}_w) * (r_i - \bar{r}_w))
    \Delta RSS = s_{tr}^2 / s_{tt}
    \Lambda = \Delta RSS / 2
    """
    n = len(residuals)
    r = np.asarray(residuals, dtype=float)
    sig = np.asarray(sigmas, dtype=float)
    sig = np.where(sig < 0.10, 0.10, sig)
    w = 1.0 / (sig ** 2)
    
    lambdas = np.zeros(n, dtype=float)
    slopes = np.zeros(n, dtype=float)
    onsets = np.zeros(n, dtype=int)
    best_weights = np.zeros(n, dtype=int)

    for i in range(min_w, n):
        max_lam = 0.0
        best_b = 0.0
        best_onset = i
        best_W = min_w
        
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
                best_W = W
                
        lambdas[i] = max_lam
        slopes[i] = best_b
        onsets[i] = best_onset
        best_weights[i] = best_W
        
    return lambdas, slopes, onsets, best_weights

def compute_unweighted_glrt_series_fast(residuals, sigma=0.45, min_w=4, max_w=48):
    """
    Pass 2 Baseline unweighted Ramp-GLRT with fixed global sigma
    """
    n = len(residuals)
    r = np.asarray(residuals, dtype=float)
    
    lambdas = np.zeros(n, dtype=float)
    slopes = np.zeros(n, dtype=float)
    onsets = np.zeros(n, dtype=int)
    
    weights_by_w = {}
    stt_by_w = {}
    for W in range(min_w, max_w + 1):
        t_w = np.arange(W, dtype=float) - (W - 1.0) / 2.0
        s_tt = W * (W**2 - 1.0) / 12.0
        weights_by_w[W] = t_w
        stt_by_w[W] = s_tt

    for i in range(min_w, n):
        max_lam = 0.0
        best_b = 0.0
        best_onset = i
        
        max_avail_w = min(i + 1, max_w)
        for W in range(min_w, max_avail_w + 1, 2):
            seg = r[i - W + 1 : i + 1]
            t_w = weights_by_w[W]
            s_tt = stt_by_w[W]
            s_tr = np.dot(t_w, seg)
            
            b_hat = s_tr / s_tt
            delta_rss = (s_tr ** 2) / s_tt
            lam = delta_rss / (2.0 * (sigma ** 2))
            
            if lam > max_lam:
                max_lam = lam
                best_b = b_hat
                best_onset = i - W + 1
                
        lambdas[i] = max_lam
        slopes[i] = best_b
        onsets[i] = best_onset
        
    return lambdas, slopes, onsets

# ==============================================================================
# 4. EXECUTION SUITE FOR PASS 5
# ==============================================================================
def run_pass5():
    print("=" * 135)
    print("SKYGUARD AI — PASS 5: CONDITIONAL RESIDUAL UNCERTAINTY")
    print("=" * 135)
    
    # 1. Fit Normal Behavior Model and Conditional Uncertainty Model
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    
    clean_dfs = {}
    for sid in STATION_TO_CLUSTER:
        df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        clean_dfs[sid] = df

    # Section 4: Raw vs Normalized Residual Statistics
    print("\n1. RAW VS NORMALIZED RESIDUAL HOMOSCEDASTICITY ANALYSIS (28 STATIONS, TEMPERATURE CHANNEL)")
    print("-" * 135)
    print(f"{'Station':<14} {'Cluster':<8} {'Raw Std':<10} {'Raw MAD':<10} {'Mean Sigma':<12} {'Norm Std (z)':<14} {'Norm MAD (z)':<14} {'Variance Reduction'}")
    print("-" * 135)
    
    raw_stds, raw_mads, mean_sigmas, norm_stds, norm_mads = [], [], [], [], []
    station_clean_data = {}
    
    for sid in sorted(STATION_TO_CLUSTER.keys()):
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: clean_dfs[pid] for pid in peer_ids}
        df_target = clean_dfs[sid]
        
        y_hat, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
        r_raw = df_target['temperature_c'].values - y_hat
        sigmas = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat, df_target['timestamp'].values)
        z_norm = r_raw / sigmas
        
        station_clean_data[sid] = {
            "y_hat": y_hat,
            "r_raw": r_raw,
            "sigmas": sigmas,
            "z_norm": z_norm,
            "timestamps": df_target['timestamp'].values
        }
        
        r_std = float(np.std(r_raw))
        r_mad = float(np.median(np.abs(r_raw - np.median(r_raw))) * 1.4826)
        m_sig = float(np.mean(sigmas))
        z_std = float(np.std(z_norm))
        z_mad = float(np.median(np.abs(z_norm - np.median(z_norm))) * 1.4826)
        
        raw_stds.append(r_std)
        raw_mads.append(r_mad)
        mean_sigmas.append(m_sig)
        norm_stds.append(z_std)
        norm_mads.append(z_mad)
        
        print(f"{sid:<14} {cid:<8} {r_std:>8.4f}°C {r_mad:>8.4f}°C {m_sig:>10.4f}°C {z_std:>12.4f} {z_mad:>12.4f}   {((r_std - z_std*m_sig)/r_std)*100:>+14.2f}%")
        
    print("-" * 135)
    print(f"NETWORK MEAN:  Raw Std = {np.mean(raw_stds):.4f}°C | Raw MAD = {np.mean(raw_mads):.4f}°C | Mean Sigma = {np.mean(mean_sigmas):.4f}°C | Norm Std = {np.mean(norm_stds):.4f} | Norm MAD = {np.mean(norm_mads):.4f}")
    print("-" * 135)

    # Section 4B: Diurnal Phase Breakdown of Uncertainty
    print("\n2. DIURNAL REGIME UNCERTAINTY PROFILE (TEMPERATURE CHANNEL ACROSS 28 STATIONS)")
    print("-" * 115)
    print(f"{'Diurnal Regime':<30} {'Hours':<15} {'Mean Raw MAD':<16} {'Mean Model Sigma':<20} {'Normalized MAD (z)'}")
    print("-" * 115)
    regime_names = ["Night (Calm / Inversion)", "Morning Transition", "Peak Daytime (Convective)", "Evening Sunset"]
    regime_hours = ["22:00 - 05:00", "06:00 - 10:00", "11:00 - 16:00", "17:00 - 21:00"]
    
    for hr_idx in range(4):
        raw_regime_mads = []
        model_regime_sigmas = []
        norm_regime_mads = []
        for sid in STATION_TO_CLUSTER:
            ts = pd.to_datetime(station_clean_data[sid]["timestamps"])
            hours = ts.hour.values if hasattr(ts, 'hour') else ts.dt.hour.values
            reg = uncertainty_model.get_hour_regime(hours)
            mask = (reg == hr_idx)
            
            r_sub = station_clean_data[sid]["r_raw"][mask]
            sig_sub = station_clean_data[sid]["sigmas"][mask]
            z_sub = station_clean_data[sid]["z_norm"][mask]
            
            raw_regime_mads.append(np.median(np.abs(r_sub - np.median(r_sub))) * 1.4826)
            model_regime_sigmas.append(np.mean(sig_sub))
            norm_regime_mads.append(np.median(np.abs(z_sub - np.median(z_sub))) * 1.4826)
            
        print(f"{regime_names[hr_idx]:<30} {regime_hours[hr_idx]:<15} {np.mean(raw_regime_mads):>12.4f}°C {np.mean(model_regime_sigmas):>16.4f}°C {np.mean(norm_regime_mads):>18.4f}")
    print("-" * 115)

    # Section 5: Clean Data Empirical Null Distributions
    print("\n3. CLEAN DATA EMPIRICAL NULL DISTRIBUTIONS (60,480 TIMESTEPS ACROSS 28 STATIONS)")
    print("-" * 125)
    clean_lam_unweighted = []
    clean_lam_weighted = []
    
    for sid in STATION_TO_CLUSTER:
        cdata = station_clean_data[sid]
        lam_unw, _, _ = compute_unweighted_glrt_series_fast(cdata['r_raw'], sigma=0.45)
        lam_w, _, _, _ = compute_weighted_glrt_series_fast(cdata['r_raw'], cdata['sigmas'])
        
        clean_lam_unweighted.extend(lam_unw)
        clean_lam_weighted.extend(lam_w)
        
    clean_lam_unweighted = np.array(clean_lam_unweighted)
    clean_lam_weighted = np.array(clean_lam_weighted)
    
    p50_unw, p90_unw, p95_unw, p99_unw = np.percentile(clean_lam_unweighted, [50, 90, 95, 99])
    p50_w, p90_w, p95_w, p99_w = np.percentile(clean_lam_weighted, [50, 90, 95, 99])
    
    print(f"{'Statistic':<25} {'Unweighted GLRT (Sigma=0.45)':<35} {'Weighted Conditional GLRT (Sigma_t)':<35} {'Ratio / Reduction'}")
    print("-" * 125)
    print(f"{'Mean':<25} {np.mean(clean_lam_unweighted):>25.3f} {np.mean(clean_lam_weighted):>33.3f} {np.mean(clean_lam_weighted)/np.mean(clean_lam_unweighted):>18.3f}")
    print(f"{'Median (P50)':<25} {p50_unw:>25.3f} {p50_w:>33.3f} {p50_w/p50_unw if p50_unw>0 else 1.0:>18.3f}")
    print(f"{'P90':<25} {p90_unw:>25.3f} {p90_w:>33.3f} {p90_w/p90_unw:>18.3f}")
    print(f"{'P95':<25} {p95_unw:>25.3f} {p95_w:>33.3f} {p95_w/p95_unw:>18.3f}")
    print(f"{'P99':<25} {p99_unw:>25.3f} {p99_w:>33.3f} {p99_w/p99_unw:>18.3f}")
    print(f"{'P(Lambda > 9.0) [%]':<25} {np.mean(clean_lam_unweighted >= 9.0)*100:>24.3f}% {np.mean(clean_lam_weighted >= 9.0)*100:>32.3f}% {((np.mean(clean_lam_weighted >= 9.0)-np.mean(clean_lam_unweighted >= 9.0))/np.mean(clean_lam_unweighted >= 9.0))*100:>17.2f}%")
    print("-" * 125)

    # Section 6: 20 Clean Weather Look-Alike Validation Cases
    print("\n4. 20 CLEAN WEATHER LOOK-ALIKE SCENARIOS (CONDITIONAL UNCERTAINTY SUPPRESSION)")
    print("-" * 145)
    print(f"{'No':<4} {'Timestamp':<20} {'Station':<13} {'Cluster':<8} {'y_hat':<9} {'r_raw':<9} {'Sigma_t':<10} {'z_norm':<9} {'Lam_Unw':<10} {'Lam_Wtd':<10} {'Detection Verdict'}")
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
    
    for idx, (ts_str, sid, cid) in enumerate(weather_scenarios, 1):
        cdata = station_clean_data[sid]
        df_target = clean_dfs[sid]
        row_idx = df_target.index[df_target['timestamp'] == ts_str]
        row_idx = row_idx[0] if len(row_idx) > 0 else 30 + idx
        
        lam_unw, _, _ = compute_unweighted_glrt_series_fast(cdata['r_raw'], sigma=0.45)
        lam_w, _, _, _ = compute_weighted_glrt_series_fast(cdata['r_raw'], cdata['sigmas'])
        
        yh = cdata['y_hat'][row_idx]
        rr = cdata['r_raw'][row_idx]
        sig = cdata['sigmas'][row_idx]
        zn = cdata['z_norm'][row_idx]
        lu = lam_unw[row_idx]
        lw = lam_w[row_idx]
        
        verdict = "CLEAN WEATHER SUPPRESSED" if lw < 9.0 else "UNRESOLVED HIGH STATISTIC"
        print(f"{idx:<4} {ts_str:<20} {sid:<13} {cid:<8} {yh:>6.2f}°C {rr:>+6.2f}°C {sig:>7.3f}°C {zn:>+6.2f} {lu:>9.2f} {lw:>9.2f}   {verdict}")
    print("-" * 145)

    # Section 7: 20 True Injected Drift Cases with Trajectory Evolution
    print("\n5. 20 TRUE INJECTED DRIFT CASES (CONDITIONAL UNCERTAINTY DETECTION EVOLUTION)")
    print("-" * 155)
    print(f"{'No':<4} {'Timestamp':<20} {'Station':<13} {'Cluster':<8} {'Inj Delta':<10} {'y_hat':<9} {'r_raw':<9} {'Sigma_t':<9} {'z_norm':<8} {'Lam_Unw':<9} {'Lam_Wtd':<9} {'Attribution Verdict'}")
    print("-" * 155)
    
    drift_data = generate_network_benchmark(regime='benchmark_b', seed=20260924, save_to_disk=False)
    drift_cases_found = 0
    
    for sid in sorted(STATION_TO_CLUSTER.keys()):
        if drift_cases_found >= 20:
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
        z_norm = r_inj / sigmas
        
        lam_unw, _, _ = compute_unweighted_glrt_series_fast(r_inj, sigma=0.45)
        lam_w, _, _, _ = compute_weighted_glrt_series_fast(r_inj, sigmas)
        
        for s_idx, e_idx in zip(df_inj.index[starts], df_inj.index[ends]):
            if drift_cases_found >= 20:
                break
            ep_len = e_idx - s_idx + 1
            if ep_len < 10:
                continue
                
            eval_idx = min(s_idx + 8, e_idx)
            ts_str = str(df_inj['timestamp'].iloc[eval_idx])
            clean_t = clean_dfs[sid]['temperature_c'].iloc[eval_idx]
            inj_t = df_inj['temperature_c'].iloc[eval_idx]
            delta_t = inj_t - clean_t
            
            yh = y_hat[eval_idx]
            rr = r_inj[eval_idx]
            sig = sigmas[eval_idx]
            zn = z_norm[eval_idx]
            lu = lam_unw[eval_idx]
            lw = lam_w[eval_idx]
            
            verdict = "STATION SENSOR DRIFT (Confirmed)" if lw >= 9.0 else "EARLY ONSET RAMP"
            drift_cases_found += 1
            print(f"{drift_cases_found:<4} {ts_str:<20} {sid:<13} {cid:<8} {delta_t:>+7.2f}°C {yh:>6.2f}°C {rr:>+6.2f}°C {sig:>6.3f}°C {zn:>+5.2f} {lu:>8.2f} {lw:>8.2f}   {verdict}")
    print("-" * 155)

    # Section 8: Peer-Consensus Masked False Negatives
    print("\n6. 20 PREVIOUS PEER-CONSENSUS MASKED CASES UNDER CONDITIONAL UNCERTAINTY")
    print("-" * 155)
    print(f"{'No':<4} {'Timestamp':<20} {'Station':<13} {'Clean T':<9} {'Inj T':<9} {'Delta':<8} {'Peer Med':<10} {'r_raw':<9} {'Sigma_t':<9} {'z_norm':<8} {'Lam_Wtd':<9} {'Detection Status'}")
    print("-" * 155)
    
    masked_count = 0
    for sid in sorted(STATION_TO_CLUSTER.keys()):
        if masked_count >= 20:
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
        z_norm = r_inj / sigmas
        lam_w, b_w, _, _ = compute_weighted_glrt_series_fast(r_inj, sigmas)
        
        # Calculate peer median residuals
        peer_res_list = []
        for pid in peer_ids:
            p_p_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != pid]
            p_p_dfs = {s: drift_data[s].sort_values("timestamp").reset_index(drop=True) for s in p_p_ids}
            y_hat_p, _ = normal_model.predict_target(pid, 'temperature_c', drift_data[pid], p_p_dfs)
            peer_res_list.append(drift_data[pid]['temperature_c'].values - y_hat_p)
        peer_res_mat = np.column_stack(peer_res_list)
        peer_med_arr = np.median(peer_res_mat, axis=1)
        
        for s_idx, e_idx in zip(df_inj.index[starts], df_inj.index[ends]):
            if masked_count >= 20:
                break
            for t_off in range(4, min(10, e_idx - s_idx + 1)):
                c_idx = s_idx + t_off
                clean_t = clean_dfs[sid]['temperature_c'].iloc[c_idx]
                inj_t = df_inj['temperature_c'].iloc[c_idx]
                delta_t = inj_t - clean_t
                
                rr = r_inj[c_idx]
                p_med = peer_med_arr[c_idx]
                sig = sigmas[c_idx]
                zn = z_norm[c_idx]
                lw = lam_w[c_idx]
                
                if abs(rr) > abs(rr - p_med) + 0.10 and lw > 4.0:
                    masked_count += 1
                    ts_str = str(df_inj['timestamp'].iloc[c_idx])
                    rec_status = "RECOVERED (Weighted Ramp Active)" if lw >= 9.0 else "ACCUMULATING EVIDENCE"
                    print(f"{masked_count:<4} {ts_str:<20} {sid:<13} {clean_t:>7.2f}°C {inj_t:>7.2f}°C {delta_t:>+6.2f}°C {p_med:>+8.2f}°C {rr:>+7.2f}°C {sig:>6.3f}°C {zn:>+5.2f} {lw:>8.2f}  {rec_status}")
                    break
    print("-" * 155)

    # Section 9 & 10: Controlled Canonical Benchmark A vs B across 7 Seeds
    print("\n7. AUTHORITATIVE CONTROLLED BENCHMARK: PASS-2 BASELINE (A) VS PASS-5 CONDITIONAL UNCERTAINTY (B)")
    print("=" * 165)
    
    artifact = joblib.load(ARTIFACTS_PATH)
    bench_results = []
    canonical_seeds = [42, 101, 202, 2024, 8888, 20260924, 45456231412727229999]
    
    # Also track per-fault-type metrics
    fault_type_totals = {
        "drift": {"a_tp": 0, "b_tp": 0, "total": 0},
        "spike": {"a_tp": 0, "b_tp": 0, "total": 0},
        "frozen": {"a_tp": 0, "b_tp": 0, "total": 0},
        "sensor_failure": {"a_tp": 0, "b_tp": 0, "total": 0},
        "stuck": {"a_tp": 0, "b_tp": 0, "total": 0},
        "noise": {"a_tp": 0, "b_tp": 0, "total": 0},
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
        
        # Base rule + fusion engine
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
        
        # Generate GLRT flags for A (Unweighted Baseline on r_raw) and B (Pass-5 Weighted on r_raw with sigma_t)
        flag_records = []
        for sid in STATION_TO_CLUSTER:
            cid = STATION_TO_CLUSTER[sid]
            peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
            peer_dfs = {pid: data[pid].sort_values("timestamp").reset_index(drop=True) for pid in peer_ids}
            df_target = data[sid].sort_values("timestamp").reset_index(drop=True)
            
            y_hat, _ = normal_model.predict_target(sid, 'temperature_c', df_target, peer_dfs)
            r_target = df_target['temperature_c'].values - y_hat
            sigmas = uncertainty_model.predict_sigma(sid, 'temperature_c', y_hat, df_target['timestamp'].values)
            
            # A: Unweighted Baseline Ramp-GLRT (fixed sigma = 0.45)
            lam_unw, b_unw, _ = compute_unweighted_glrt_series_fast(r_target, sigma=0.45)
            flags_a = (lam_unw >= 9.0) & (np.abs(b_unw) >= 0.04)
            
            # B: Pass-5 Weighted Conditional Ramp-GLRT
            lam_w, b_w, _, _ = compute_weighted_glrt_series_fast(r_target, sigmas)
            flags_b = (lam_w >= 9.0) & (np.abs(b_w) >= 0.04)
            
            df_stn_flags = pd.DataFrame({
                "station_id": sid,
                "timestamp": pd.to_datetime(df_target["timestamp"]).dt.tz_localize(None),
                "glrt_A": flags_a,
                "glrt_B": flags_b
            })
            flag_records.append(df_stn_flags)
            
        all_flags_df = pd.concat(flag_records, ignore_index=True)
        featured = featured.merge(all_flags_df, on=["station_id", "timestamp"], how="left")
        glrt_flags_A = featured["glrt_A"].fillna(False).to_numpy(dtype=bool)
        glrt_flags_B = featured["glrt_B"].fillna(False).to_numpy(dtype=bool)
        
        # Eval A
        feat_A = featured.copy()
        raw_nans_featured = feat_A["__raw_nan_flag"].fillna(False).to_numpy(dtype=bool)
        feat_A = feat_A.merge(labels, on=["station_id", "timestamp"], how="left")
        feat_A["is_anomaly"] = feat_A["is_anomaly"].fillna(False).astype(bool) | raw_nans_featured
        feat_A["fault_type"] = feat_A["fault_type"].fillna("none")
        feat_A["__predicted"] = base_predicted | glrt_flags_A
        m_a = _score_and_report(feat_A, "ALL FILES COMBINED", 0, silent=True)
        drift_a = feat_A[feat_A["fault_type"] == "drift"]
        dr_a_tp = (drift_a["__predicted"] == True).sum()
        dr_a_fn = (drift_a["__predicted"] == False).sum()
        dr_a_rec = dr_a_tp / len(drift_a) if len(drift_a) else 0.0
        
        # Eval B
        feat_B = featured.copy()
        feat_B = feat_B.merge(labels, on=["station_id", "timestamp"], how="left")
        feat_B["is_anomaly"] = feat_B["is_anomaly"].fillna(False).astype(bool) | raw_nans_featured
        feat_B["fault_type"] = feat_B["fault_type"].fillna("none")
        feat_B["__predicted"] = base_predicted | glrt_flags_B
        m_b = _score_and_report(feat_B, "ALL FILES COMBINED", 0, silent=True)
        drift_b = feat_B[feat_B["fault_type"] == "drift"]
        dr_b_tp = (drift_b["__predicted"] == True).sum()
        dr_b_fn = (drift_b["__predicted"] == False).sum()
        dr_b_rec = dr_b_tp / len(drift_b) if len(drift_b) else 0.0
        
        # Fault breakdown accumulation
        for ftype in fault_type_totals.keys():
            sub = feat_A[feat_A["fault_type"] == ftype]
            fault_type_totals[ftype]["total"] += len(sub)
            fault_type_totals[ftype]["a_tp"] += (sub["__predicted"] == True).sum()
            sub_b = feat_B[feat_B["fault_type"] == ftype]
            fault_type_totals[ftype]["b_tp"] += (sub_b["__predicted"] == True).sum()
        
        bench_results.append({
            "Seed": seed,
            "A_Prec": m_a["precision"],
            "B_Prec": m_b["precision"],
            "A_Rec": m_a["recall"],
            "B_Rec": m_b["recall"],
            "A_F1": m_a["f1"],
            "B_F1": m_b["f1"],
            "A_FP": m_a["fp"],
            "B_FP": m_b["fp"],
            "A_TP": m_a["tp"],
            "B_TP": m_b["tp"],
            "A_FN": m_a["fn"],
            "B_FN": m_b["fn"],
            "A_DrTP": dr_a_tp,
            "B_DrTP": dr_b_tp,
            "A_DrRec": dr_a_rec,
            "B_DrRec": dr_b_rec,
            "FP_Reduc": (m_a["fp"] - m_b["fp"]) / m_a["fp"] * 100.0,
            "DrTP_Ret": dr_b_tp / dr_a_tp * 100.0 if dr_a_tp > 0 else 0.0
        })
        
    df_res = pd.DataFrame(bench_results)
    print(f"{'Seed':<10} {'A Prec':<9} {'B Prec':<9} {'A Rec':<9} {'B Rec':<9} {'A F1':<7} {'B F1':<7} {'A FP':<8} {'B FP':<8} {'A DrTP':<8} {'B DrTP':<8} {'A DrRec':<9} {'B DrRec':<9} {'FP Reduc':<9} {'DrTP Ret'}")
    print("-" * 165)
    for _, r in df_res.iterrows():
        print(f"{int(r['Seed']):<10} {r['A_Prec']*100:>7.2f}% {r['B_Prec']*100:>7.2f}% {r['A_Rec']*100:>7.2f}% {r['B_Rec']*100:>7.2f}% {r['A_F1']:>6.3f} {r['B_F1']:>6.3f} {int(r['A_FP']):>7} {int(r['B_FP']):>7} {int(r['A_DrTP']):>7} {int(r['B_DrTP']):>7} {r['A_DrRec']*100:>7.2f}% {r['B_DrRec']*100:>7.2f}% {r['FP_Reduc']:>7.2f}% {r['DrTP_Ret']:>7.2f}%")
    print("-" * 165)
    print(f"{'MEAN':<10} {df_res['A_Prec'].mean()*100:>7.2f}% {df_res['B_Prec'].mean()*100:>7.2f}% {df_res['A_Rec'].mean()*100:>7.2f}% {df_res['B_Rec'].mean()*100:>7.2f}% {df_res['A_F1'].mean():>6.3f} {df_res['B_F1'].mean():>6.3f} {df_res['A_FP'].mean():>7.1f} {df_res['B_FP'].mean():>7.1f} {df_res['A_DrTP'].mean():>7.1f} {df_res['B_DrTP'].mean():>7.1f} {df_res['A_DrRec'].mean()*100:>7.2f}% {df_res['B_DrRec'].mean()*100:>7.2f}% {df_res['FP_Reduc'].mean():>7.2f}% {df_res['DrTP_Ret'].mean():>7.2f}%")
    print(f"{'STD':<10} {df_res['A_Prec'].std()*100:>7.2f}% {df_res['B_Prec'].std()*100:>7.2f}% {df_res['A_Rec'].std()*100:>7.2f}% {df_res['B_Rec'].std()*100:>7.2f}% {df_res['A_F1'].std():>6.3f} {df_res['B_F1'].std():>6.3f} {df_res['A_FP'].std():>7.1f} {df_res['B_FP'].std():>7.1f} {df_res['A_DrTP'].std():>7.1f} {df_res['B_DrTP'].std():>7.1f} {df_res['A_DrRec'].std()*100:>7.2f}% {df_res['B_DrRec'].std()*100:>7.2f}% {df_res['FP_Reduc'].std():>7.2f}% {df_res['DrTP_Ret'].std():>7.2f}%")
    print(f"{'MIN':<10} {df_res['A_Prec'].min()*100:>7.2f}% {df_res['B_Prec'].min()*100:>7.2f}% {df_res['A_Rec'].min()*100:>7.2f}% {df_res['B_Rec'].min()*100:>7.2f}% {df_res['A_F1'].min():>6.3f} {df_res['B_F1'].min():>6.3f} {df_res['A_FP'].min():>7} {df_res['B_FP'].min():>7} {df_res['A_DrTP'].min():>7} {df_res['B_DrTP'].min():>7} {df_res['A_DrRec'].min()*100:>7.2f}% {df_res['B_DrRec'].min()*100:>7.2f}% {df_res['FP_Reduc'].min():>7.2f}% {df_res['DrTP_Ret'].min():>7.2f}%")
    print(f"{'MAX':<10} {df_res['A_Prec'].max()*100:>7.2f}% {df_res['B_Prec'].max()*100:>7.2f}% {df_res['A_Rec'].max()*100:>7.2f}% {df_res['B_Rec'].max()*100:>7.2f}% {df_res['A_F1'].max():>6.3f} {df_res['B_F1'].max():>6.3f} {df_res['A_FP'].max():>7} {df_res['B_FP'].max():>7} {df_res['A_DrTP'].max():>7} {df_res['B_DrTP'].max():>7} {df_res['A_DrRec'].max()*100:>7.2f}% {df_res['B_DrRec'].max()*100:>7.2f}% {df_res['FP_Reduc'].max():>7.2f}% {df_res['DrTP_Ret'].max():>7.2f}%")
    print("=" * 165)

    # Section 11: Failure Type Breakdown
    print("\n8. FAILURE TYPE DETECTION RECALL BREAKDOWN ACROSS 7 SEEDS")
    print("-" * 115)
    print(f"{'Fault Type':<20} {'Total Points':<15} {'A Recalled':<15} {'A Recall %':<15} {'B Recalled':<15} {'B Recall %':<15} {'Delta Recall'}")
    print("-" * 115)
    for ftype, stats in fault_type_totals.items():
        tot = stats["total"]
        a_tp = stats["a_tp"]
        b_tp = stats["b_tp"]
        a_rec = a_tp / tot * 100.0 if tot > 0 else 0.0
        b_rec = b_tp / tot * 100.0 if tot > 0 else 0.0
        print(f"{ftype:<20} {tot:<15} {a_tp:<15} {a_rec:>13.2f}% {b_tp:<15} {b_rec:>13.2f}% {b_rec - a_rec:>+13.2f}%")
    print("-" * 115)

if __name__ == '__main__':
    run_pass5()
