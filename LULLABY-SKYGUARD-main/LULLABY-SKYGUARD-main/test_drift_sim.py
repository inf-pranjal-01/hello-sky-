import pandas as pd
import numpy as np
from model.features import build_feature_matrix
from model.seasonal_baseline import get_expected_roc
from config import CUSUM_DRIFT_ALLOWANCE, CUSUM_THRESHOLD, EWMA_DRIFT_ALPHA, EWMA_DRIFT_THRESHOLD

df = pd.read_csv('data/AWS-DEL-011_labeled.csv')
featured = build_feature_matrix(df)

splus = 0.0
sminus = 0.0
ewma_val = 0.0
allowance = 0.05
hours = pd.to_datetime(featured['timestamp']).dt.hour.to_numpy()
raw_rocs = featured['temp_roc_1h'].to_numpy()
scales = featured['temp_robust_scale'].to_numpy()
is_drift = (df['fault_type'] == 'drift').to_numpy()

cusum_hits = 0
ewma_hits = 0
total_drift = is_drift.sum()

for i in range(len(featured)):
    raw_roc = raw_rocs[i]
    scale_val = max(float(scales[i]), 1.0) if pd.notna(scales[i]) else 1.0
    h = hours[i]
    if pd.notna(raw_roc):
        expected = get_expected_roc('AWS-DEL-011', 'temp', int(h))
        residual = float(np.clip((raw_roc - expected) / scale_val, -3.0, 3.0))
        splus = max(0.0, splus + residual - allowance)
        sminus = max(0.0, sminus - residual - allowance)
        ewma_val = EWMA_DRIFT_ALPHA * residual + (1.0 - EWMA_DRIFT_ALPHA) * ewma_val
    
    if splus > CUSUM_THRESHOLD or sminus > CUSUM_THRESHOLD:
        if is_drift[i]:
            cusum_hits += 1
    if abs(ewma_val) > EWMA_DRIFT_THRESHOLD:
        if is_drift[i]:
            ewma_hits += 1

print(f'Total drift: {total_drift}, CUSUM hits: {cusum_hits} ({cusum_hits/total_drift:.1%}), EWMA hits: {ewma_hits} ({ewma_hits/total_drift:.1%})')
