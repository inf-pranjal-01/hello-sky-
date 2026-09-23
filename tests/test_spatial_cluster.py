import unittest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.detect import _corroborate_network

class TestSpatialCluster(unittest.TestCase):
    def test_insufficient_corroboration(self):
        # Empty peers should return INSUFFICIENT_CORROBORATION
        raw_reading = {"station_id": "AWS-CHN-024", "timestamp": "2023-01-01T12:00:00Z"}
        history_df = pd.DataFrame([raw_reading])
        result = _corroborate_network(raw_reading, history_df, {}, {}, pd.Series({"temp_roc_1h": 0.0}), "drift", ["temperature_c"])
        self.assertEqual(result["state"], "INSUFFICIENT_CORROBORATION")
        self.assertEqual(result["eligible_peer_count"], 0)

    def test_stale_peer_filtering(self):
        # A peer with stale data (>1hr old) should be skipped
        raw_reading = {"station_id": "AWS-CHN-024", "timestamp": "2023-01-01T12:00:00Z"}
        history_df = pd.DataFrame([raw_reading])
        
        stale_peer_df = pd.DataFrame([
            {"station_id": "AWS-CHN-101", "timestamp": "2023-01-01T10:00:00Z", "temperature_c": 25.0} # 2 hours old
        ])
        
        result = _corroborate_network(raw_reading, history_df, {"peer_1": stale_peer_df}, {"peer_1": pd.Series({"temp_roc_1h": 0.0})}, pd.Series({"temp_roc_1h": 0.0}), "drift", ["temperature_c"])
        self.assertEqual(result["state"], "INSUFFICIENT_CORROBORATION")
        self.assertEqual(result["eligible_peer_count"], 0)

if __name__ == '__main__':
    unittest.main()
