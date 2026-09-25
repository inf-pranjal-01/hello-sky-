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
    add_frozen_channel_labels_from_reference
)
from scratch.run_benchmark_o_evaluation import (
    CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel,
    ConditionalMultivariateJointModel, STATION_TO_CLUSTER
)

def evaluate_model_on_multivariate(seed=42):
    print(f"=== PHASE 3: CRITICAL ML MODEL TEST FOR MULTIVARIATE (SEED {seed}) ===")
    artifact = joblib.load(ARTIFACTS_PATH)
    data = generate_network_benchmark(regime='observable_v1', seed=seed, save_to_disk=False)
    
    frames = []
    for sid, df_raw in data.items():
        d = df_raw.copy()
        d['station_id'] = sid
        frames.append(d)
    df_full = pd.concat(frames, ignore_index=True)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp']).dt.tz_localize(None)
    df_full = add_frozen_channel_labels_from_reference(df_full)

    raw_nans = df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']].isna().any(axis=1).to_numpy(dtype=bool)
    df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']] = df_full[['temperature_c', 'pressure_hpa', 'humidity_pct']].ffill().bfill()

    label_cols = ['station_id', 'timestamp', 'is_anomaly', 'fault_type']
    labels = df_full[label_cols].copy()
    labels['is_anomaly'] = labels['is_anomaly'].fillna(False).astype(bool)
    labels['fault_type'] = labels['fault_type'].fillna('none')

    df_in = df_full.drop(columns=['is_anomaly', 'fault_type', 'injected_delta', 'injected_start', 'injected_end'], errors='ignore')
    featured, _ = _featurize(df_in)
    featured['timestamp'] = pd.to_datetime(featured['timestamp']).dt.tz_localize(None)

    featured = featured.merge(labels, on=['station_id', 'timestamp'], how='left')
    model_pct = vectorized_model_scores(featured, artifact)
    featured['model_score'] = model_pct

    # Test ML model score distributions across categories
    mv_df = featured[featured['fault_type'] == 'multivariate_inconsistency']
    unstr_df = featured[featured['fault_type'] == 'unstructured_anomaly']
    drift_df = featured[featured['fault_type'] == 'drift']
    spike_df = featured[featured['fault_type'] == 'spike']
    frozen_df = featured[featured['fault_type'] == 'frozen_value']
    normal_df = featured[featured['fault_type'] == 'none']

    print(f"\n--- MODEL SCORE PERCENTILE DISTRIBUTIONS ---")
    cats = [
        ("Multivariate Inconsistency", mv_df),
        ("Unstructured Anomaly", unstr_df),
        ("Drift", drift_df),
        ("Spike", spike_df),
        ("Frozen Value", frozen_df),
        ("Clean Normal Weather", normal_df)
    ]
    for cname, cdf in cats:
        scores = cdf['model_score'].dropna().values
        print(f"{cname:<30} (N={len(cdf):>6}): Mean={scores.mean():>5.1f} | Med={np.median(scores):>5.1f} | 25th={np.percentile(scores, 25):>5.1f} | 75th={np.percentile(scores, 75):>5.1f} | >60%: {(scores > 60).sum()/len(scores)*100:>5.1f}% | >75%: {(scores > 75).sum()/len(scores)*100:>5.1f}%")

    print("\n--- MULTIVARIATE RECALL & PRECISION vs MODEL SCORE THRESHOLDS ---")
    for thresh in [40, 50, 60, 70, 75, 80, 85]:
        mv_caught = (mv_df['model_score'] >= thresh).sum()
        norm_fp = (normal_df['model_score'] >= thresh).sum()
        print(f"  Threshold >= {thresh:>2}% -> MV Recall: {mv_caught}/{len(mv_df)} ({mv_caught/len(mv_df)*100:>5.1f}%) | Normal Weather FPs: {norm_fp:>5} ({norm_fp/len(normal_df)*100:>5.2f}%)")

if __name__ == '__main__':
    evaluate_model_on_multivariate(seed=42)
