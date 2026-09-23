import os
import sys
import unittest
from pathlib import Path
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.detect import score_reading
from model.features import add_temporal_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestRanchiRegression(unittest.TestCase):
    def test_ranchi_diurnal_drift_robustness(self):
        csv_path = PROJECT_ROOT / "data" / "AWS-RAN-067_labeled.csv"
        self.assertTrue(os.path.exists(csv_path), f"{csv_path} should exist")
            
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        end_time = pd.to_datetime("2025-01-02 12:00:00")
        start_time = end_time - pd.Timedelta(days=5)
        
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        test_df = df[mask].copy().sort_values("timestamp").reset_index(drop=True)
        
        artifact_path = PROJECT_ROOT / "model_artifacts" / "isolation_forest.pkl"
        self.assertTrue(os.path.exists(artifact_path), f"{artifact_path} should exist")
        artifact = joblib.load(artifact_path)
        
        target_times = [
            "2025-01-02 08:00:00",
            "2025-01-02 09:00:00",
            "2025-01-02 10:00:00",
            "2025-01-02 11:00:00",
            "2025-01-02 12:00:00",
        ]
        
        for target in target_times:
            target_dt = pd.to_datetime(target)
            hist = test_df[test_df["timestamp"] <= target_dt].copy()
            self.assertGreater(len(hist), 0)
                
            raw_reading = hist.iloc[-1].to_dict()
            raw_reading["timestamp"] = str(raw_reading["timestamp"])
            
            verdict = score_reading(raw_reading, hist, artifact)
            self.assertIn("is_anomaly", verdict)
            
            # Crucial verification: Sunrise warming (hours 8-11) must NOT falsely trigger temperature drift!
            rules_fired = verdict.get("rules_fired", [])
            temp_drift_rules = [
                r for r in rules_fired
                if (isinstance(r, dict) and r.get("type") == "drift" and r.get("parameter") == "temperature_c")
            ]
            self.assertEqual(len(temp_drift_rules), 0, f"Temperature sunrise warming at {target} must not trigger drift rule!")

if __name__ == "__main__":
    unittest.main()
