import unittest
import sys
import os
import pandas as pd

# Ensure we can import from model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.detect import _rule_checks, PHYSICAL_BOUNDS

class TestRuleBoundaries(unittest.TestCase):
    def test_physical_limits(self):
        artifact = {"rule_thresholds": {}}
        feature_row = pd.Series({
            "temp_frozen_streak": 0,
            "pressure_frozen_streak": 0,
            "humidity_frozen_streak": 0,
        })
        
        # Test just below upper threshold (normal)
        raw_reading = {
            "station_id": "AWS-CHN-024",
            "timestamp": "2024-01-01T12:00:00Z",
            "temperature_c": PHYSICAL_BOUNDS["temperature_c"][1] - 0.5,
            "pressure_hpa": 1000.0,
            "humidity_pct": 50.0,
        }
        history_df = pd.DataFrame([raw_reading])
        rules = _rule_checks(raw_reading, feature_row, history_df, artifact)
        self.assertFalse(any(r["type"] == "physical_bounds" for r in rules["fired"]))
        
        # Test just above upper threshold (out of bounds)
        raw_reading_hi = {
            "station_id": "AWS-CHN-024",
            "timestamp": "2024-01-01T12:00:00Z",
            "temperature_c": PHYSICAL_BOUNDS["temperature_c"][1] + 5.0,
            "pressure_hpa": 1000.0,
            "humidity_pct": 50.0,
        }
        history_df_hi = pd.DataFrame([raw_reading_hi])
        rules_hi = _rule_checks(raw_reading_hi, feature_row, history_df_hi, artifact)
        self.assertTrue(any(r["type"] == "physical_bounds" for r in rules_hi["fired"]))

    def test_dropout_check(self):
        artifact = {"rule_thresholds": {}}
        feature_row = pd.Series({
            "temp_frozen_streak": 0,
            "pressure_frozen_streak": 0,
            "humidity_frozen_streak": 0,
        })
        # Missing/NaN temperature
        raw_reading = {
            "station_id": "AWS-CHN-024",
            "timestamp": "2024-01-01T12:00:00Z",
            "temperature_c": None,
            "pressure_hpa": 1000.0,
            "humidity_pct": 50.0,
        }
        history_df = pd.DataFrame([raw_reading])
        rules = _rule_checks(raw_reading, feature_row, history_df, artifact)
        self.assertTrue(any(r["type"] == "dropout" for r in rules["fired"]))

if __name__ == '__main__':
    unittest.main()
