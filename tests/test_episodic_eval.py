"""
tests/test_episodic_eval.py
===========================
Automated tests for evaluation/episodic_eval.py strictly covering
Tests 1 through 12 from the Final Evaluation Change specification.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_episodic_eval.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.episodic_eval import (
    extract_gt_episodes,
    evaluate_episodes,
    compute_episodic_result,
    FaultEpisode,
    EpisodicResult,
    LATENCY_CREDIT_FAULT_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(gt: list, ft: list, pred: list, pred_ft: list = None, station: str = "S1") -> pd.DataFrame:
    """Build a minimal evaluation dataframe from row-by-row lists."""
    if pred_ft is None:
        pred_ft = [ft[i] if pred[i] else "none" for i in range(len(pred))]
    assert len(gt) == len(ft) == len(pred) == len(pred_ft)
    n = len(gt)
    ts = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "station_id": [station] * n,
        "timestamp": ts,
        "is_anomaly": [bool(x) for x in gt],
        "fault_type": ft,
        "__pred": [bool(x) for x in pred],
        "__pred_ft": pred_ft,
    })


def run(df: pd.DataFrame) -> EpisodicResult:
    return compute_episodic_result(
        df,
        pred_arr=df["__pred"].to_numpy(),
        pred_ft_arr=df["__pred_ft"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Required Tests 1 through 12
# ---------------------------------------------------------------------------

class TestRequiredSuite:

    def test_test1_perfect_frozen_detection(self):
        # GT:    F F F F
        # Model: X X X X
        df = make_df(
            gt   = [True, True, True, True],
            ft   = ["frozen"] * 4,
            pred = [True, True, True, True],
        )
        r = run(df)
        assert r.tp_star == 4
        assert r.fn_star == 0
        assert r.raw_tp == 4
        assert r.raw_fn == 0
        assert r.latency_aware_recall == pytest.approx(1.0)

    def test_test2_frozen_detection_lag(self):
        # GT:    F F F F F F
        # Model: . . X X X X
        # Expected: TP* = 6, FN* = 0 (first two are lag credit)
        df = make_df(
            gt   = [True, True, True, True, True, True],
            ft   = ["frozen"] * 6,
            pred = [False, False, True, True, True, True],
        )
        r = run(df)
        assert r.raw_tp == 4
        assert r.raw_fn == 2
        assert r.tp_star == 6
        assert r.fn_star == 0
        assert r.latency_aware_recall == pytest.approx(1.0)
        assert r.episodes[0].latency_hours == pytest.approx(2.0)

    def test_test3_frozen_detects_then_misses(self):
        # GT:    F F F F F F F F
        # Model: . . X X . . . .
        # First det at pos 2 -> rows 0,1 get lag credit (2), rows 2,3 are actual TPs (2), rows 4-7 are FNs (4)
        # Expected: TP* = 4, FN* = 4 (NOT TP* = 8, FN* = 0)
        df = make_df(
            gt   = [True] * 8,
            ft   = ["frozen"] * 8,
            pred = [False, False, True, True, False, False, False, False],
        )
        r = run(df)
        assert r.tp_star == 4
        assert r.fn_star == 4
        assert r.raw_tp == 2
        assert r.raw_fn == 6
        assert r.latency_aware_recall == pytest.approx(4 / 8)

    def test_test4_completely_missed_frozen(self):
        # GT:    F F F F
        # Model: . . . .
        # Expected: TP* = 0, FN* = 4 (NO lag credit)
        df = make_df(
            gt   = [True, True, True, True],
            ft   = ["frozen"] * 4,
            pred = [False, False, False, False],
        )
        r = run(df)
        assert r.tp_star == 0
        assert r.fn_star == 4
        assert r.detected_episodes == 0
        assert r.missed_episodes == 1
        assert r.latency_aware_recall == pytest.approx(0.0)

    def test_test5_drift_detection_lag(self):
        # Same principle for drift
        df = make_df(
            gt   = [True, True, True, True, True, True],
            ft   = ["drift"] * 6,
            pred = [False, False, True, True, True, True],
        )
        r = run(df)
        assert r.tp_star == 6
        assert r.fn_star == 0
        assert r.raw_tp == 4
        assert r.raw_fn == 2

    def test_test6_spike_must_not_receive_latency_credit(self):
        # GT:    F F F F
        # Model: . . X X
        # Expected: TP = 2, FN = 2 (NOT TP* = 4, FN* = 0)
        df = make_df(
            gt   = [True, True, True, True],
            ft   = ["spike"] * 4,
            pred = [False, False, True, True],
        )
        r = run(df)
        assert r.tp_star == 2
        assert r.fn_star == 2
        assert r.raw_tp == 2
        assert r.raw_fn == 2
        assert r.latency_aware_recall == pytest.approx(2 / 4)

    def test_test7_bias_must_not_receive_latency_credit(self):
        # Bias receives NO latency credit
        df = make_df(
            gt   = [True, True, True, True],
            ft   = ["bias"] * 4,
            pred = [False, False, True, True],
        )
        r = run(df)
        assert r.tp_star == 2
        assert r.fn_star == 2
        assert r.raw_tp == 2
        assert r.raw_fn == 2

    def test_test8_fp_unchanged(self):
        # Introduce false positives outside fault episodes
        df = make_df(
            gt   = [False, True, True, False, False],
            ft   = ["none", "drift", "drift", "none", "none"],
            pred = [True,  False, True,  True,  False],
        )
        r = run(df)
        assert r.raw_fp == 2   # pos 0 and pos 3
        # Latency logic must NEVER forgive or alter false positives
        assert r.raw_fp == 2

    def test_test9_wrong_attribution_but_successful_detection(self):
        # GT type: frozen, Prediction: anomaly (True), Predicted type: drift
        # Expected: Detection = YES, Attribution = NO
        df = make_df(
            gt      = [True, True, True, True],
            ft      = ["frozen"] * 4,
            pred    = [False, False, True, True],
            pred_ft = ["none", "none", "drift", "drift"],
        )
        r = run(df)
        ep = r.episodes[0]
        assert ep.detected is True
        assert ep.attribution_correct is False
        assert ep.predicted_fault_type_at_first_detection == "drift"
        # Since it is a detected frozen episode, it still receives lag credit for detection
        assert r.tp_star == 4
        assert r.fn_star == 0

    def test_test10_two_separate_frozen_episodes(self):
        # Verify detection in episode 1 cannot give latency credit to episode 2
        # GT:    F F F F | N N | F F F F
        # Model: . . X . | . . | . . . .
        # Ep 1: detected at pos 2 -> rows 0,1 lag credit (2), row 2 actual TP (1), row 3 FN (1) -> TP*=3, FN*=1
        # Ep 2: missed -> TP*=0, FN*=4
        df = make_df(
            gt   = [True, True, True, True, False, False, True, True, True, True],
            ft   = ["frozen"] * 4 + ["none", "none"] + ["frozen"] * 4,
            pred = [False, False, True, False, False, False, False, False, False, False],
        )
        r = run(df)
        assert r.total_episodes == 2
        assert r.detected_episodes == 1
        assert r.missed_episodes == 1
        assert r.episodes[0].detected is True
        assert r.episodes[0].tp_star == 3
        assert r.episodes[0].fn_star == 1
        assert r.episodes[1].detected is False
        assert r.episodes[1].tp_star == 0
        assert r.episodes[1].fn_star == 4
        assert r.tp_star == 3
        assert r.fn_star == 5

    def test_test11_recovery_boundary(self):
        # Predictions after a frozen/drift episode ends cannot provide credit to the preceding episode
        # GT:    F F F F | N N N
        # Model: . . . . | X X .
        df = make_df(
            gt   = [True, True, True, True, False, False, False],
            ft   = ["frozen"] * 4 + ["none", "none", "none"],
            pred = [False, False, False, False, True, True, False],
        )
        r = run(df)
        assert r.episodes[0].detected is False
        assert r.tp_star == 0
        assert r.fn_star == 4
        assert r.raw_fp == 2

    def test_test12_non_frozen_non_drift_fault_regression(self):
        # Run dropout, noise, unstructured_anomaly, sensor_fail_low, multivariate_inconsistency
        for non_lag_ft in ["dropout", "noise", "unstructured_anomaly", "sensor_fail_low", "multivariate_inconsistency"]:
            df = make_df(
                gt   = [True, True, True, True, True, True],
                ft   = [non_lag_ft] * 6,
                pred = [False, False, True, True, False, False],
            )
            r = run(df)
            assert r.tp_star == r.raw_tp == 2
            assert r.fn_star == r.raw_fn == 4


# ---------------------------------------------------------------------------
# Strict Invariant Tests
# ---------------------------------------------------------------------------

class TestStrictInvariants:

    def test_sum_invariant(self):
        df = make_df(
            gt   = [False, True, True, False, True, True, True, False],
            ft   = ["none", "frozen", "frozen", "none", "spike", "spike", "spike", "none"],
            pred = [False, False, True, False, False, True, False, False],
        )
        r = run(df)
        total_gt = int(df["is_anomaly"].sum())
        assert r.tp_star + r.fn_star == total_gt == 5

    def test_predictions_immutable(self):
        pred_arr = np.array([False, False, True, True])
        pred_copy = pred_arr.copy()
        df = make_df(
            gt   = [True, True, True, True],
            ft   = ["drift"] * 4,
            pred = pred_arr,
        )
        run(df)
        assert np.array_equal(pred_arr, pred_copy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
