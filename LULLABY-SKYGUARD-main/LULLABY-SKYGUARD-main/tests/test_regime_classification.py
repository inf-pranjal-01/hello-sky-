import unittest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.detect import _classify_regime

class TestRegimeClassification(unittest.TestCase):
    def test_high_heat(self):
        feature_row = pd.Series({
            "temperature_c": 36.0,
            "humidity_pct": 50.0,
            "temp_deviation": 1.0,
            "humidity_deviation": 0.5,
            "temp_roc_1h": 0.0,
            "temp_roc_3h": 0.0,
            "pressure_roc_3h": 0.0,
            "temp_volatility_z": 0.0,
            "pressure_volatility_z": 0.0,
            "humidity_volatility_z": 0.0
        })
        raw_reading = {"timestamp": "2023-01-01T12:00:00Z"}
        history_df = pd.DataFrame()
        regime = _classify_regime(feature_row, raw_reading, history_df)
        self.assertEqual(regime, "HIGH_HEAT")

    def test_pressure_shift(self):
        feature_row = pd.Series({
            "temperature_c": 20.0,
            "humidity_pct": 50.0,
            "temp_deviation": 0.0,
            "humidity_deviation": 0.0,
            "temp_roc_1h": 0.0,
            "temp_roc_3h": 0.0,
            "pressure_roc_3h": 2.5,  # > 2.0 triggers PRESSURE_SHIFT
            "temp_volatility_z": 0.0,
            "pressure_volatility_z": 0.0,
            "humidity_volatility_z": 0.0
        })
        raw_reading = {"timestamp": "2023-01-01T12:00:00Z"}
        history_df = pd.DataFrame()
        regime = _classify_regime(feature_row, raw_reading, history_df)
        self.assertEqual(regime, "PRESSURE_SHIFT")

if __name__ == '__main__':
    unittest.main()
