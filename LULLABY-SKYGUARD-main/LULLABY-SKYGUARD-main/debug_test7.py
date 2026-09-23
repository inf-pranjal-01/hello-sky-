
import pandas as pd
import sys
sys.path.insert(0, '.')
from model.features import add_temporal_features

times = [pd.Timestamp('2024-06-01T00:00:00Z') + pd.Timedelta(hours=i) for i in range(30)]
target_history = pd.DataFrame([
    {'station_id': 'AWS-CHN-024', 'timestamp': t, 'temperature_c': 30.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0}
    for t in times[:-1]
] + [{'station_id': 'AWS-CHN-024', 'timestamp': times[-1], 'temperature_c': 45.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0}])

features = add_temporal_features(target_history)
print(features[['temperature_c', 'temp_deviation', 'temp_roc_1h']].tail(2))

