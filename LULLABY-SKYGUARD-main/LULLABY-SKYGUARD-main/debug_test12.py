
import pandas as pd
import sys
sys.path.insert(0, '.')
from model.detect import _corroborate_network
import numpy as np

np.random.seed(42)
times = [pd.Timestamp('2024-06-01T00:00:00Z') + pd.Timedelta(hours=i) for i in range(30)]
target_history = pd.DataFrame([
    {'station_id': 'AWS-CHN-024', 'timestamp': t, 'temperature_c': 30.0 + np.random.normal(0, 0.5), 'pressure_hpa': 1005.0, 'humidity_pct': 70.0}
    for t in times[:-1]
] + [{'station_id': 'AWS-CHN-024', 'timestamp': times[-1], 'temperature_c': 45.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0}])

raw_reading = {'station_id': 'AWS-CHN-024', 'timestamp': times[-1], 'temperature_c': 45.0}

swing_neighbors = {
    'AWS-CHN-101': pd.DataFrame([
        {'station_id': 'AWS-CHN-101', 'timestamp': t, 'temperature_c': 30.0 + np.random.normal(0, 0.5), 'pressure_hpa': 1005.0, 'humidity_pct': 70.0}
        for t in times[:-1]
    ] + [{'station_id': 'AWS-CHN-101', 'timestamp': times[-1], 'temperature_c': 45.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0}]),
    'AWS-CHN-103': pd.DataFrame([
        {'station_id': 'AWS-CHN-103', 'timestamp': t, 'temperature_c': 30.0 + np.random.normal(0, 0.5), 'pressure_hpa': 1005.0, 'humidity_pct': 70.0}
        for t in times[:-1]
    ] + [{'station_id': 'AWS-CHN-103', 'timestamp': times[-1], 'temperature_c': 45.5, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0}]),
}

res_drift = _corroborate_network(raw_reading, target_history, swing_neighbors, fault_type='drift', implicated_params=['temperature_c'])
print(res_drift)

