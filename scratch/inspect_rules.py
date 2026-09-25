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
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel, ConditionalMultivariateDriftDetector,
    STATION_TO_CLUSTER
)
from scratch.test_injector_detector_sync import generate_network_benchmark_v5

def inspect_rules():
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()
    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()
    joint_model = ConditionalMultivariateJointModel(normal_model, uncertainty_model, train_ratio=0.60)
    joint_model.fit()

    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark_v5(seed=42)

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

    spatial_z_dict = defaultdict(dict)
    for sid in STATION_TO_CLUSTER:
        df_clean = data[sid].copy()
        df_clean[['temperature_c', 'pressure_hpa', 'humidity_pct']] = df_clean[['temperature_c', 'pressure_hpa', 'humidity_pct']].ffill().bfill()
        cid = STATION_TO_CLUSTER[sid]
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: data[pid].sort_values('timestamp').reset_index(drop=True) for pid in peer_ids}
        for p in joint_model.channels:
            y_hat, _ = normal_model.predict_target_robust(sid, p, df_clean, peer_dfs)
            r = df_clean[p].values - y_hat
            sig = uncertainty_model.predict_sigma(sid, p, y_hat, df_clean['timestamp'].values)
            z_p = np.nan_to_num(r / np.maximum(0.1, sig), nan=0.0)
            spatial_z_dict[sid][p] = z_p

    # Inspect spatial_z distributions on normal rows vs GT MV rows
    feat_m = featured.merge(labels, on=['station_id', 'timestamp'], how='left')
    is_gt_mv = (feat_m['fault_type'] == 'multivariate_inconsistency').to_numpy()
    is_gt_normal = (feat_m['fault_type'] == 'none').to_numpy()

    print(f"Total GT MV rows: {is_gt_mv.sum()}, Total GT normal rows: {is_gt_normal.sum()}")

    # Collect z_T * z_RH for MV vs Normal
    all_prod_mv = []
    all_prod_norm = []
    all_vapor_mv = []
    all_vapor_norm = []

    for sid in STATION_TO_CLUSTER:
        stn_sub = feat_m[feat_m['station_id'] == sid]
        m = len(stn_sub)
        z_T = spatial_z_dict[sid]['temperature_c'][-m:]
        z_RH = spatial_z_dict[sid]['humidity_pct'][-m:]
        prod = z_T * z_RH
        mv_mask = (stn_sub['fault_type'] == 'multivariate_inconsistency').to_numpy()
        norm_mask = (stn_sub['fault_type'] == 'none').to_numpy()
        all_prod_mv.extend(prod[mv_mask])
        all_prod_norm.extend(prod[norm_mask])

    all_prod_mv = np.array(all_prod_mv)
    all_prod_norm = np.array(all_prod_norm)

    print("\nz_T * z_RH statistics:")
    print(f"  GT MV:     min={np.min(all_prod_mv):.2f}, p10={np.percentile(all_prod_mv, 10):.2f}, mean={np.mean(all_prod_mv):.2f}, p90={np.percentile(all_prod_mv, 90):.2f}")
    print(f"  GT Normal: min={np.min(all_prod_norm):.2f}, p90={np.percentile(all_prod_norm, 90):.2f}, p99={np.percentile(all_prod_norm, 99):.2f}, max={np.max(all_prod_norm):.2f}")
    print(f"  Normal rows with prod >= 8.0: {(all_prod_norm >= 8.0).sum()}/{len(all_prod_norm)}")
    print(f"  Normal rows with prod >= 15.0: {(all_prod_norm >= 15.0).sum()}/{len(all_prod_norm)}")
    print(f"  MV rows with prod >= 15.0: {(all_prod_mv >= 15.0).sum()}/{len(all_prod_mv)}")

if __name__ == '__main__':
    inspect_rules()
