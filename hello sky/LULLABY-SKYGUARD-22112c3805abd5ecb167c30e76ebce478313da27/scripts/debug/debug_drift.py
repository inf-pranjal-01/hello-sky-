"""Debug why CUSUM fires on normal diurnal warming."""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from model.features import add_temporal_features
from model.seasonal_baseline import get_expected_roc

station = "AWS-CHN-101"
df = pd.read_csv(f"data/{station}.csv", parse_dates=["timestamp"])
df["station_id"] = station
df = df.head(200)
df = add_temporal_features(df)

# Show the mismatch: normalized_roc_1h is in z-score units, expected_roc is in raw C/h
print("=== UNIT MISMATCH PROOF ===")
print("normalized_roc_1h is roc_1h / rolling_std (dimensionless z-score)")
print("get_expected_roc returns raw C/h")
print()

morning = df[(df["timestamp"].dt.hour >= 6) & (df["timestamp"].dt.hour <= 12)].dropna(subset=["temp_normalized_roc_1h"])
for _, row in morning.head(15).iterrows():
    h = row["timestamp"].hour
    nroc = row["temp_normalized_roc_1h"]
    roc = row["temp_roc_1h"]
    std = row["temp_rolling_std"]
    expected = get_expected_roc(station, "temp", h)
    residual = nroc - expected  # THIS IS THE BUG: z-score minus C/h
    correct_residual = (roc - expected) / std if std > 0 else 0  # What it SHOULD be
    
    print(f"  {row['timestamp']}  h={h:2d}  "
          f"roc_1h={roc:+7.3f}C/h  "
          f"std={std:.3f}  "
          f"norm_roc={nroc:+7.3f}  "
          f"expected_roc={expected:+7.3f}C/h  "
          f"BUGGY_residual={residual:+7.3f}  "
          f"CORRECT_residual={correct_residual:+7.3f}")
