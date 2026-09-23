
import pandas as pd
import sys
sys.path.insert(0, '.')
from model.detect import _corroborate_network
target_history = pd.DataFrame([
    {'station_id': 'AWS-CHN-024', 'timestamp': pd.Timestamp('2024-06-01T11:00:00Z'), 'temperature_c': 30.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0},
    {'station_id': 'AWS-CHN-024', 'timestamp': pd.Timestamp('2024-06-01T12:00:00Z'), 'temperature_c': 45.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0},
])
raw_reading = {'station_id': 'AWS-CHN-024', 'timestamp': '2024-06-01T12:00:00Z', 'temperature_c': 45.0}

swing_neighbors = {
    'AWS-CHN-101': pd.DataFrame([
        {'station_id': 'AWS-CHN-101', 'timestamp': pd.Timestamp('2024-06-01T11:00:00Z'), 'temperature_c': 30.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0},
        {'station_id': 'AWS-CHN-101', 'timestamp': pd.Timestamp('2024-06-01T12:00:00Z'), 'temperature_c': 45.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0},
    ]),
    'AWS-CHN-103': pd.DataFrame([
        {'station_id': 'AWS-CHN-103', 'timestamp': pd.Timestamp('2024-06-01T11:00:00Z'), 'temperature_c': 30.0, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0},
        {'station_id': 'AWS-CHN-103', 'timestamp': pd.Timestamp('2024-06-01T12:00:00Z'), 'temperature_c': 45.5, 'pressure_hpa': 1005.0, 'humidity_pct': 70.0},
    ]),
}
res_drift = _corroborate_network(raw_reading, target_history, swing_neighbors, fault_type='drift', implicated_params=['temperature_c'])
print(res_drift)

