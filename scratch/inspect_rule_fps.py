import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

# Let's inspect where the FPs in test_all_fixes_seed42 came from
# Is it frozen, spike, or GLRT?
from scratch.test_all_fixes_seed42 import (
    normal_model, uncertainty_model, joint_model, detector,
    run_calibrated_rules, artifact, thresholds, STATION_TO_CLUSTER
)
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import (
    vectorized_model_scores, _featurize,
    add_frozen_channel_labels_from_reference
)

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

featured_base, row_hard, row_rule_conf, row_fault_type = run_calibrated_rules(featured.copy(), data)

feat_final = featured.copy()
feat_final = feat_final.merge(labels, on=['station_id', 'timestamp'], how='left')
feat_final['is_anomaly'] = feat_final['is_anomaly'].fillna(False).astype(bool) | raw_nans

is_gt = np.asarray(feat_final['is_anomaly'], dtype=bool)

print("Rule-alone FP breakdown:")
for ft in ["frozen_value", "spike", "multivariate_inconsistency", "sensor_fail_low"]:
    rule_fires = (row_fault_type == ft) & (row_rule_conf >= 90.0)
    fp_for_rule = rule_fires & (~is_gt)
    tp_for_rule = rule_fires & is_gt
    print(f"  Rule '{ft}': Fires={rule_fires.sum()}, TP={tp_for_rule.sum()}, FP={fp_for_rule.sum()}")
