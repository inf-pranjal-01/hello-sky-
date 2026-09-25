import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib, pandas as pd, numpy as np
from config import MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD, MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    ARTIFACTS_PATH, vectorized_model_scores, run_rule_engine_and_health,
    apply_spatial_corroboration, _featurize
)
from scratch.run_benchmark_o_evaluation import (
    STATION_TO_CLUSTER, CausalNormalBehaviorModel,
    ConditionalResidualUncertaintyModel, ConditionalMultivariateJointModel,
    ConditionalMultivariateDriftDetector
)

artifact = joblib.load(ARTIFACTS_PATH)
data = generate_network_benchmark(regime='operational_v1', seed=42, save_to_disk=False)
frames = [d.assign(station_id=sid) for sid, d in data.items()]
df_full = pd.concat(frames, ignore_index=True)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp']).dt.tz_localize(None)

label_cols = ['station_id', 'timestamp', 'is_anomaly', 'fault_type']
labels = df_full[label_cols].copy()
labels['is_anomaly'] = labels['is_anomaly'].fillna(False).astype(bool)

df_in = df_full.drop(columns=['is_anomaly', 'fault_type', 'injected_delta', 'injected_start', 'injected_end'], errors='ignore')
featured, _ = _featurize(df_in)
featured['timestamp'] = pd.to_datetime(featured['timestamp']).dt.tz_localize(None)

# Merge labels to aligned featured frame
featured = featured.merge(labels, on=['station_id', 'timestamp'], how='left')
featured_base, row_hard, row_rule_conf, row_fault_type, _, _ = run_rule_engine_and_health(featured.copy(), artifact)
row_rule_conf, row_fault_type, _ = apply_spatial_corroboration(featured_base, row_hard, row_rule_conf, row_fault_type, artifact, gate_mode='new')
model_pct = vectorized_model_scores(featured_base, artifact)

is_clean = ~featured['is_anomaly'].to_numpy(dtype=bool)
is_fault = featured['is_anomaly'].to_numpy(dtype=bool)

print('Total featured clean rows:', is_clean.sum(), 'Total featured fault rows:', is_fault.sum())
print('Clean rows with row_hard:', (row_hard & is_clean).sum())
print('Clean rows with model_pct > 95:', ((model_pct > 95.0) & is_clean).sum())
print('Clean rows with model_pct > 90:', ((model_pct > 90.0) & is_clean).sum())
print('Clean rows with model_pct > 85:', ((model_pct > 85.0) & is_clean).sum())
print('Clean rows with model_pct > 70:', ((model_pct > 70.0) & is_clean).sum())
print('Clean rows with row_rule_conf > 0:', ((row_rule_conf > 0) & is_clean).sum())
print('Clean rows with row_rule_conf >= 90:', ((row_rule_conf >= 90) & is_clean).sum())

fusion_score = MODEL_WEIGHT * model_pct + RULE_WEIGHT * row_rule_conf
print('Clean rows with fusion > 50 and rule_conf > 0:', ((fusion_score > 50.0) & (row_rule_conf > 0) & is_clean).sum())
print('Fault rows with fusion > 50 and rule_conf > 0:', ((fusion_score > 50.0) & (row_rule_conf > 0) & is_fault).sum())

# Drift GLRT on clean rows:
normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
normal_model.fit()
uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
uncertainty_model.fit()
joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
joint_model.fit()

detector = ConditionalMultivariateDriftDetector(glrt_thresh=26.0, cusum_thresh=22.0, allowance=0.45)
det_flags_list = []
for sid in STATION_TO_CLUSTER:
    df = data[sid].sort_values('timestamp').reset_index(drop=True)
    cid = STATION_TO_CLUSTER[sid]
    peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
    peer_dfs = {pid: data[pid].sort_values('timestamp').reset_index(drop=True) for pid in peer_ids}
    hr = uncertainty_model.get_hour_regime(pd.to_datetime(df['timestamp']).dt.hour.values)

    Z_list = []
    for p in joint_model.channels:
        y_hat, _ = normal_model.predict_target_robust(sid, p, df, peer_dfs)
        r = df[p].values - y_hat
        sig = uncertainty_model.predict_sigma(sid, p, y_hat, df['timestamp'].values)
        Z_list.append(r / np.maximum(0.1, sig))
    Z = np.column_stack(Z_list)
    n_rows = len(Z)

    u_T = np.zeros(n_rows)
    sig_cond_T = np.ones(n_rows)
    for i in range(n_rows):
        u, s = joint_model.get_conditional_innovation(sid, hr[i], Z[i], 0)
        u_T[i] = u
        sig_cond_T[i] = s

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

print('Drift detector triggers on clean rows:', (det_online & is_clean).sum())
print('Drift detector triggers on fault rows:', (det_online & is_fault).sum())

# What rules are firing on clean rows?
for ft in np.unique(row_fault_type):
    cnt_clean = ((row_fault_type == ft) & is_clean).sum()
    cnt_fault = ((row_fault_type == ft) & is_fault).sum()
    print(f'Rule fault type {ft:<28}: clean={cnt_clean} | fault={cnt_fault}')
