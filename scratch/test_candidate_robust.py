import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.linalg as la
import joblib

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
from scratch.run_chance1_causal_audit import (
    STATION_TO_CLUSTER, CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel
)

def run():
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()

    clean_dfs = {sid: pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp']) for sid in STATION_TO_CLUSTER}
    for sid in clean_dfs:
        clean_dfs[sid]['timestamp'] = pd.to_datetime(clean_dfs[sid]['timestamp']).dt.tz_localize(None)

    params = ['temperature_c', 'humidity_pct', 'pressure_hpa']
    joint_models = {}
    for sid, cid in STATION_TO_CLUSTER.items():
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        df_target = clean_dfs[sid]
        n_train = int(len(df_target) * 0.60)
        df_train = df_target.iloc[:n_train]
        peer_tr = {pid: clean_dfs[pid].iloc[:n_train] for pid in peer_ids}
        hr_tr = uncertainty_model.get_hour_regime(df_train['timestamp'].dt.hour.values)
        
        Z_tr_list = []
        for p in params:
            y_tr, _ = normal_model.predict_target(sid, p, df_train, peer_tr)
            r_tr = df_train[p].values - y_tr
            sig_tr = uncertainty_model.predict_sigma(sid, p, y_tr, df_train['timestamp'].values)
            Z_tr_list.append(r_tr / np.maximum(0.1, sig_tr))
        Z_tr = np.column_stack(Z_tr_list)
        
        stn_joint = {}
        for regime in range(4):
            mask = (hr_tr == regime)
            if mask.sum() < 20: continue
            Z_sub = Z_tr[mask]
            mu = np.mean(Z_sub, axis=0)
            Z_c = Z_sub - mu
            S = (Z_c.T @ Z_c) / (len(Z_c) - 1)
            Sigma = 0.5 * (S + S.T) + 0.05 * np.eye(3)
            
            reg_info = {'mu': mu, 'cond': {}}
            for c in range(3):
                other = [i for i in range(3) if i != c]
                sig_cc = Sigma[c, c]
                sig_c_other = Sigma[c:c+1, other]
                sig_other_other = Sigma[np.ix_(other, other)]
                W = la.solve(sig_other_other, sig_c_other.T, assume_a='pos').T
                cond_var = sig_cc - (W @ sig_c_other.T)[0, 0]
                reg_info['cond'][c] = {
                    'W': W,
                    'other': other,
                    'cond_var': cond_var,
                    'cond_std': np.sqrt(cond_var)
                }
            stn_joint[regime] = reg_info
        joint_models[sid] = stn_joint

    def predict_target_robust(sid, p, target_df, peer_dfs_dict):
        model_info = normal_model.models.get((sid, p))
        if not model_info or not model_info['peer_regressors']:
            return target_df[p].values, np.ones(len(target_df))
        ts = pd.to_datetime(target_df['timestamp'])
        hours = ts.hour.values if hasattr(ts, 'hour') else ts.dt.hour.values
        sin_h = np.sin(2 * np.pi * hours / 24.0)
        cos_h = np.cos(2 * np.pi * hours / 24.0)
        n = len(target_df)
        target_ts = pd.to_datetime(target_df['timestamp']).dt.tz_localize(None)
        preds = []
        for pid, reg in model_info['peer_regressors'].items():
            if pid in peer_dfs_dict:
                coeffs = reg['coeffs']
                p_df = peer_dfs_dict[pid].copy()
                p_df['timestamp'] = pd.to_datetime(p_df['timestamp']).dt.tz_localize(None)
                p_df = p_df.set_index('timestamp')
                p_val = p_df.reindex(target_ts)[p].ffill().bfill().values
                if p == 'temperature_c': p_val = np.clip(p_val, -10.0, 55.0)
                elif p == 'humidity_pct': p_val = np.clip(p_val, 0.0, 100.0)
                elif p == 'pressure_hpa': p_val = np.clip(p_val, 800.0, 1100.0)
                X = np.column_stack([np.ones(n), p_val, sin_h, cos_h])
                preds.append(X @ coeffs)
        if preds:
            y_med = np.median(np.column_stack(preds), axis=1)
        else:
            y_med = target_df[p].values
        return y_med, model_info['comb_sigma'] * np.ones(n, dtype=float)

    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='benchmark_b', seed=42, save_to_disk=False)

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

    labels = df_full[['station_id', 'timestamp', 'is_anomaly', 'fault_type']].copy()
    labels['is_anomaly'] = labels['is_anomaly'].fillna(False).astype(bool)
    labels['fault_type'] = labels['fault_type'].fillna('none')

    df_in = df_full.drop(columns=['is_anomaly', 'fault_type'], errors='ignore')
    featured, _ = _featurize(df_in)
    featured['timestamp'] = pd.to_datetime(featured['timestamp']).dt.tz_localize(None)

    featured_base, row_hard, row_rule_conf, row_fault_type, _, _ = run_rule_engine_and_health(featured.copy(), artifact)
    row_rule_conf, row_fault_type, _ = apply_spatial_corroboration(
        featured_base, row_hard, row_rule_conf, row_fault_type, artifact, gate_mode='new'
    )
    model_pct = vectorized_model_scores(featured_base, artifact)
    overall_confidence = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
    base_predicted = (
        row_hard
        | ((overall_confidence > FUSION_ANOMALY_THRESHOLD) & (row_rule_conf > 0))
        | (model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD)
        | (row_rule_conf > RULE_CONFIDENCE_BYPASS)
    )

    theta_glrt = 24.0
    theta_cusum = 18.0
    min_coherence = 0.65

    det_flags_list = []
    for sid in STATION_TO_CLUSTER:
        df = data[sid].sort_values('timestamp').reset_index(drop=True)
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: data[pid].sort_values('timestamp').reset_index(drop=True) for pid in peer_ids}
        hr = uncertainty_model.get_hour_regime(pd.to_datetime(df['timestamp']).dt.hour.values)
        
        Z_list = []
        for p in params:
            y_hat, _ = predict_target_robust(sid, p, df, peer_dfs)
            r = df[p].values - y_hat
            sig = uncertainty_model.predict_sigma(sid, p, y_hat, df['timestamp'].values)
            Z_list.append(r / np.maximum(0.1, sig))
        Z = np.column_stack(Z_list)
        n = len(Z)
        
        u_T = np.zeros(n)
        sig_cond_T = np.ones(n)
        for i in range(n):
            rg = hr[i]
            reg_info = joint_models[sid].get(rg, joint_models[sid][0])
            cond_T = reg_info['cond'][0]
            mu = reg_info['mu']
            other = cond_T['other']
            W = cond_T['W']
            pred_z_T = mu[0] + (W @ (Z[i, other] - mu[other]))[0]
            u_T[i] = Z[i, 0] - pred_z_T
            sig_cond_T[i] = cond_T['cond_std']
            
        z_u = u_T / sig_cond_T
        w = 1.0 / (sig_cond_T ** 2)
        
        c_p = 0.0
        c_m = 0.0
        state = 0
        online_preds = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if i < 4: continue
            c_p = max(0.0, 0.88 * c_p + (z_u[i] - 0.40))
            c_m = max(0.0, 0.88 * c_m + (-z_u[i] - 0.40))
            
            max_lam = 0.0
            best_b = 0.0
            max_w = min(i + 1, 24)
            for W in range(4, max_w + 1, 2):
                idx_s = i - W + 1
                w_seg = w[idx_s:i+1]
                u_seg = u_T[idx_s:i+1]
                t_seg = np.arange(W, dtype=float)
                sw = np.sum(w_seg)
                t_bar = np.sum(w_seg * t_seg) / sw
                u_bar = np.sum(w_seg * u_seg) / sw
                t_dev = t_seg - t_bar
                u_dev = u_seg - u_bar
                s_tt = np.sum(w_seg * (t_dev**2))
                s_tu = np.sum(w_seg * t_dev * u_dev)
                lam = (s_tu**2) / (2.0 * s_tt)
                if lam > max_lam:
                    max_lam = lam
                    best_b = s_tu / s_tt
                    
            w_coh = min(i + 1, 12)
            diffs = np.diff(z_u[i - w_coh + 1 : i + 1])
            cand_dir = +1 if (c_p > c_m and best_b > 0) else (-1 if (c_m > c_p and best_b < 0) else 0)
            if len(diffs) > 0 and cand_dir != 0:
                steps_supporting = np.sum(diffs > 0) if cand_dir == +1 else np.sum(diffs < 0)
                sign_consistency = steps_supporting / len(diffs)
            else:
                sign_consistency = 0.50
                
            active_cum = max(c_p, c_m)
            is_coherent = (sign_consistency >= min_coherence)
            
            glrt_trig = (max_lam >= theta_glrt and abs(best_b) >= 0.04 and is_coherent)
            cusum_trig = (active_cum >= theta_cusum and is_coherent and abs(z_u[i]) >= 1.5)
            
            if state == 0:
                if glrt_trig or cusum_trig: state = 2
                elif (max_lam >= 10.0 or active_cum >= 8.0) and abs(best_b) >= 0.02: state = 1
            elif state == 1:
                if glrt_trig or cusum_trig: state = 2
                elif max_lam < 5.0 and active_cum < 4.0: state = 0
            elif state == 2:
                if abs(z_u[i]) < 1.0 and max_lam < 5.0 and active_cum < 4.0: state = 3
            elif state == 3:
                if abs(z_u[i]) < 1.0 and max_lam < 3.0:
                    state = 0; c_p = 0.0; c_m = 0.0
                elif glrt_trig or cusum_trig: state = 2
                
            online_preds[i] = (state == 2)
            
        df_stn = pd.DataFrame({
            'station_id': sid,
            'timestamp': pd.to_datetime(df['timestamp']).dt.tz_localize(None),
            'detector_online': online_preds
        })
        det_flags_list.append(df_stn)

    all_det_flags = pd.concat(det_flags_list, ignore_index=True)
    featured = featured.merge(all_det_flags, on=['station_id', 'timestamp'], how='left')
    det_online_arr = featured['detector_online'].fillna(False).to_numpy(dtype=bool)

    cand_predicted = base_predicted | det_online_arr

    feat_cand = featured.copy()
    feat_cand = feat_cand.merge(labels, on=['station_id', 'timestamp'], how='left')
    feat_cand['is_anomaly'] = feat_cand['is_anomaly'].fillna(False).astype(bool) | feat_cand['__raw_nan_flag'].fillna(False).to_numpy(dtype=bool)
    feat_cand['fault_type'] = feat_cand['fault_type'].fillna('none')
    feat_cand['__predicted'] = cand_predicted
    feat_cand['__predicted_fault_type'] = row_fault_type
    feat_cand.loc[det_online_arr & (feat_cand['__predicted_fault_type'] == 'none'), '__predicted_fault_type'] = 'drift'

    m_cand = _score_and_report(feat_cand, 'ALL FILES COMBINED', 0, silent=True)
    ep_cand = compute_episodic_result(
        feat_cand,
        pred_arr=feat_cand['__predicted'].to_numpy(dtype=bool),
        pred_ft_arr=feat_cand['__predicted_fault_type'].to_numpy()
    )

    print(f"Robust Candidate Precision: {m_cand['precision']*100:.2f}% | Recall: {m_cand['recall']*100:.2f}% | F1: {m_cand['f1']:.4f} | F1*: {ep_cand.latency_aware_f1:.4f}")
    print(f"TP: {m_cand['tp']} | FP: {m_cand['fp']} | FN: {m_cand['fn']} | Episode Catch: {ep_cand.episode_detection_rate*100:.2f}%")
    drift_sub = feat_cand[feat_cand['fault_type'] == 'drift']
    dr_recalled = (drift_sub['__predicted'].to_numpy(dtype=bool)).sum()
    print(f"Drift point recall: {dr_recalled} / {len(drift_sub)} ({dr_recalled/len(drift_sub)*100:.2f}%)")

if __name__ == '__main__':
    run()
