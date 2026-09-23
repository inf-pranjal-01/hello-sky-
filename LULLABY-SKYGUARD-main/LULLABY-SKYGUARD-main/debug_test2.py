
import pandas as pd
import sys
sys.path.insert(0, '.')
from model.features import build_features_for_latest
target_history = pd.DataFrame([
    {'station_id': 'AWS-CHN-024', 'timestamp': pd.Timestamp('2024-06-01T11:00:00Z'), 'temperature_c': 30.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0},
    {'station_id': 'AWS-CHN-024', 'timestamp': pd.Timestamp('2024-06-01T12:00:00Z'), 'temperature_c': 45.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0},
])
features = build_features_for_latest(target_history)
print(features[['temperature_c', 'temp_deviation', 'temp_roc_1h']])

