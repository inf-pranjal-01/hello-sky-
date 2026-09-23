import unittest
import sys
import os
import pandas as pd
import numpy as np

# Ensure we can import from model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.detect import _cusum_evidence

class TestCusumDrift(unittest.TestCase):
    def test_positive_drift(self):
        # Continuous climbing temperature residuals well above seasonal expectations
        dates = pd.date_range('2024-01-01 00:00', periods=12, freq='h')
        temps = [20.0 + i * 1.5 for i in range(12)]
        df = pd.DataFrame({
            'timestamp': dates,
            'temperature_c': temps,
            'temp_roc_1h': [1.5] * 12,
            'temp_robust_scale': [0.5] * 12
        })
        res = _cusum_evidence(df, 'temp', 'temperature_c', station_id='FAKE-TEST-000', current_hour=12)
        self.assertIsNotNone(res, "CUSUM drift should trigger on persistent climbing residuals")
        self.assertEqual(res["type"], "drift")
        self.assertEqual(res["parameter"], "temperature_c")
        self.assertGreaterEqual(res["confidence"], 85.0)

    def test_normal_weather_no_drift(self):
        # Stable temperatures without rate of change
        dates = pd.date_range('2024-01-01 00:00', periods=12, freq='h')
        temps = [25.0] * 12
        df = pd.DataFrame({
            'timestamp': dates,
            'temperature_c': temps,
            'temp_roc_1h': [0.0] * 12,
            'temp_robust_scale': [1.0] * 12
        })
        res = _cusum_evidence(df, 'temp', 'temperature_c', station_id='FAKE-TEST-000', current_hour=12)
        self.assertIsNone(res, "CUSUM should not trigger on normal stable weather")

    def test_direction_streak_required(self):
        # Oscillating residuals (no consistent positive or negative streak)
        dates = pd.date_range('2024-01-01 00:00', periods=12, freq='h')
        temps = [20.0 + (1.5 if i % 2 == 1 else 0.0) for i in range(12)]
        rocs = [1.5 if i % 2 == 1 else -1.5 for i in range(12)]
        df = pd.DataFrame({
            'timestamp': dates,
            'temperature_c': temps,
            'temp_roc_1h': rocs,
            'temp_robust_scale': [0.5] * 12
        })
        res = _cusum_evidence(df, 'temp', 'temperature_c', station_id='FAKE-TEST-000', current_hour=12)
        self.assertIsNone(res, "CUSUM requires persistent direction streak and should not fire on oscillating noise")

if __name__ == '__main__':
    unittest.main()
