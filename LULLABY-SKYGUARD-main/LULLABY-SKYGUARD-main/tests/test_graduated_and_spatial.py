import unittest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    graduated_confidence_frozen,
    graduated_confidence_drift,
    graduated_confidence_spike,
    graduated_confidence_fail_low,
    graduated_confidence_multivariate,
    RULE_CONFIDENCE_BYPASS,
)
from model.detect import _corroborate_network

class TestGraduatedAndSpatial(unittest.TestCase):
    def test_graduated_confidence_marginal_vs_extreme(self):
        # 1. Frozen Value
        conf_frozen_marginal = graduated_confidence_frozen(streak=4, req=4)
        conf_frozen_mid = graduated_confidence_frozen(streak=6, req=4)
        conf_frozen_extreme = graduated_confidence_frozen(streak=10, req=4)
        self.assertEqual(conf_frozen_marginal, 80.0)
        self.assertTrue(80.0 < conf_frozen_mid < 95.0)
        self.assertEqual(conf_frozen_extreme, 95.0)

        # 2. Drift (CUSUM)
        conf_drift_marginal = graduated_confidence_drift(accumulator_val=7.1, threshold=7.0)
        conf_drift_mid = graduated_confidence_drift(accumulator_val=10.5, threshold=7.0)
        conf_drift_extreme = graduated_confidence_drift(accumulator_val=14.5, threshold=7.0)
        self.assertTrue(85.0 <= conf_drift_marginal < 86.0)
        self.assertEqual(conf_drift_mid, 87.0)
        self.assertEqual(conf_drift_extreme, 89.0)

        # 3. Fail Low
        conf_faillow_marginal = graduated_confidence_fail_low(val=-8.0, floor=-8.0, streak=2, req=2)
        conf_faillow_extreme = graduated_confidence_fail_low(val=-35.0, floor=-8.0, streak=6, req=2)
        self.assertEqual(conf_faillow_marginal, 92.0)
        self.assertEqual(conf_faillow_extreme, 98.0)

        # 4. Multivariate
        conf_mv_single_marginal = graduated_confidence_multivariate(joint_z=3.1, threshold=3.0, confirmed=False)
        conf_mv_single_high = graduated_confidence_multivariate(joint_z=6.0, threshold=3.0, confirmed=False)
        conf_mv_conf_marginal = graduated_confidence_multivariate(joint_z=3.1, threshold=3.0, confirmed=True)
        conf_mv_conf_high = graduated_confidence_multivariate(joint_z=6.0, threshold=3.0, confirmed=True)
        self.assertLess(conf_mv_single_high, conf_mv_conf_marginal)

        # 5. Spike
        conf_spike_clean = graduated_confidence_spike(abs_dev=6.0, spike_threshold=3.0, reversion_cleanliness=1.0)
        conf_spike_partial = graduated_confidence_spike(abs_dev=4.0, spike_threshold=3.0, reversion_cleanliness=0.6)
        self.assertGreater(conf_spike_clean, conf_spike_partial)

    def test_spike_and_multivariate_untouched_by_network_corroboration(self):
        raw_reading = {'station_id': 'AWS-KOL-015', 'timestamp': '2024-06-01T12:00:00Z', 'temperature_c': 35.0}
        history_df = pd.DataFrame([{
            'station_id': 'AWS-KOL-015',
            'timestamp': pd.Timestamp('2024-06-01T12:00:00Z'),
            'temperature_c': 35.0,
            'pressure_hpa': 1005.0,
            'humidity_pct': 70.0,
        }])
        empty_neighbors = {}
        swing_neighbors = {
            'AWS-KOL-101': pd.DataFrame([{
                'station_id': 'AWS-KOL-101',
                'timestamp': pd.Timestamp('2024-06-01T12:00:00Z'),
                'temperature_c': 40.0,
                'pressure_hpa': 1005.0,
                'humidity_pct': 50.0,
            }]),
        }
        res_spike_empty = _corroborate_network(raw_reading, history_df, empty_neighbors, fault_type='spike', implicated_params=['temperature_c'])
        res_spike_swing = _corroborate_network(raw_reading, history_df, swing_neighbors, fault_type='spike', implicated_params=['temperature_c'])
        self.assertEqual(res_spike_empty['confidence_bonus'], 0.0)
        self.assertEqual(res_spike_swing['confidence_bonus'], 0.0)
        self.assertIsNone(res_spike_swing['relabel_fault_type'])

        res_mv_empty = _corroborate_network(raw_reading, history_df, empty_neighbors, fault_type='multivariate_inconsistency', implicated_params=['temperature_c', 'humidity_pct'])
        res_mv_swing = _corroborate_network(raw_reading, history_df, swing_neighbors, fault_type='multivariate_inconsistency', implicated_params=['temperature_c', 'humidity_pct'])
        self.assertEqual(res_mv_empty['confidence_bonus'], 0.0)
        self.assertEqual(res_mv_swing['confidence_bonus'], 0.0)
        self.assertIsNone(res_mv_swing['relabel_fault_type'])

    def test_is_anomaly_never_flipped_by_network_corroboration(self):
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
        self.assertTrue('is_anomaly' not in res_drift or res_drift.get('is_anomaly') is not False)
        self.assertEqual(res_drift['confidence_bonus'], 3.0)
        self.assertEqual(res_drift['relabel_fault_type'], 'REGIONAL_EVENT')

if __name__ == '__main__':
    unittest.main()
