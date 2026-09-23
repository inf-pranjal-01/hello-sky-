import unittest

import pandas as pd

from model.evaluate import episodic_metrics


class EpisodicMetricsTests(unittest.TestCase):
    def make_frame(self, gt, pred, pred_type=None):
        timestamps = pd.date_range("2026-01-01", periods=len(gt), freq="h")
        return pd.DataFrame({
            "station_id": ["AWS-TEST-001"] * len(gt),
            "timestamp": timestamps,
            "fault_type_gt": ["drift" if value else "none" for value in gt],
            "fault_type_pred": pred_type or ["drift" if value else None for value in pred],
            "is_anomaly_pred": pred,
        })

    def test_delayed_alert_overlaps_one_episode(self):
        frame = self.make_frame([False, True, True, True, True, False],
                                [False, False, True, True, False, False])
        result = episodic_metrics(frame, "drift")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))

    def test_unmatched_alert_counts_as_false_episode(self):
        frame = self.make_frame([False, True, True, False, False, False],
                                [False, False, False, False, True, False])
        result = episodic_metrics(frame, "drift")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_consecutive_alerts_count_as_one_episode(self):
        frame = self.make_frame([False] * 6, [False, True, True, True, False, False])
        result = episodic_metrics(frame, "drift")
        self.assertEqual(result["pred_episodes"], 1)
        self.assertEqual(result["fp"], 1)

    def test_fault_types_are_matched_separately(self):
        frame = self.make_frame([False, True, True, False],
                                [False, True, True, False],
                                [None, "frozen_value", "frozen_value", None])
        result = episodic_metrics(frame, "drift")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
