import math
import numpy as np
from typing import Optional, Tuple
import json

SPIKE_WINDOW_HOURS = 5
SPIKE_DECAY_RATIO = 0.50
SPIKE_NOISE_FLOOR = 0.2
SPIKE_NOISE_STD_MULTIPLIER = 1.5
SPIKE_VIOLATION_CONSECUTIVE_REQUIRED = 3

def init_spike_state() -> dict:
    return {
        "status": "IDLE",
        "t0_val": 0.0,
        "peak_val": 0.0,
        "jump": 0.0,
        "ticks": 0,
        "elapsed_hours": 0.0,
        "baseline_drift": 0.0,
        "best_progress": 0.0,
        "violation_run": 0,
        "prev_val": None,
        "candidate_dev": 0.0,
    }

def step_spike_state(
    val: float,
    dev: float,
    spike_thresh: float,
    multiplier: float,
    state: dict,
    graduated_conf_func=None,
    dt_hours: float = 1.0,
    noise_std: Optional[float] = None,
    expected_roc: float = 0.0,
    **kwargs
) -> Tuple[float, str, str]:

    if val is None or (isinstance(val, float) and math.isnan(val)):
        state["status"] = "IDLE"
        state["prev_val"] = val
        return 0.0, "IDLE", ""

    prev_val = state.get("prev_val")
    state["prev_val"] = val
    status = state.get("status", "IDLE")

    if status == "IDLE":
        if prev_val is not None and not (isinstance(prev_val, float) and math.isnan(prev_val)):
            step_diff = abs(val - prev_val)
            abs_dev = abs(dev) if dev is not None and not (isinstance(dev, float) and math.isnan(dev)) else step_diff
            if step_diff >= spike_thresh and abs_dev >= (spike_thresh * multiplier):
                state["status"] = "PROVISIONAL"
                state["t0_val"] = prev_val
                state["peak_val"] = val
                state["jump"] = val - prev_val
                state["ticks"] = 0
                state["elapsed_hours"] = 0.0
                state["baseline_drift"] = 0.0
                state["best_progress"] = abs(val - prev_val)
                state["violation_run"] = 0
                state["candidate_dev"] = abs_dev

                # PEAK-CONFIDENCE BOOST (Tier-2 fix):
                # When the jump is large enough (|dev| >= 2x effective threshold),
                # the peak row itself is high-certainty evidence — boost to 92.0
                # (> RULE_CONFIDENCE_BYPASS=90) so t0 bypasses fusion without
                # needing the recovery tick to confirm. Intentionally coupled with
                # the CONFIRMED_SPIKE suppression below; neither change works alone.
                effective_thresh = spike_thresh * multiplier
                if abs_dev >= 2.0 * effective_thresh:
                    return 92.0, "PROVISIONAL", f"Large spike at peak ({abs_dev:.1f} >= 2x{effective_thresh:.1f}); peak-confidence bypass."

                return 40.0, "PROVISIONAL", f"Candidate jump of {step_diff:.1f} detected."
        return 0.0, "IDLE", ""

    elif status == "PROVISIONAL":
        dt = dt_hours if (dt_hours is not None and dt_hours > 0) else 1.0
        state["ticks"] += 1
        state["elapsed_hours"] += dt
        state["baseline_drift"] += (expected_roc or 0.0) * dt

        jump = state["jump"]
        t0 = state["t0_val"]
        expected_now = t0 + state["baseline_drift"]
        resid = abs(val - expected_now)

        if resid <= abs(jump) * SPIKE_DECAY_RATIO:
            state["status"] = "IDLE"
            effective_thresh = spike_thresh * multiplier
            peak_was_boosted = state.get("candidate_dev", 0.0) >= 2.0 * effective_thresh

            if peak_was_boosted:
                # RECOVERY-TICK SUPPRESSION (coupled with peak boost at IDLE→PROVISIONAL):
                # Only suppress when the peak row already fired at conf=92.0 (bypass).
                # For those episodes the recovery tick is a clean-weather FP — suppress it.
                # Small spikes (candidate_dev < 2x threshold) did NOT get a boosted peak,
                # so they still need the recovery tick to carry the alert — do not suppress.
                return 0.0, "CONFIRMED_SPIKE", "Recovery tick suppressed (peak already alerted via bypass)."
            else:
                # Small spike: peak did not bypass fusion. Keep recovery tick so the
                # episode can still alert. conf from graduated_conf_func or fallback 95.
                conf = 95.0
                if graduated_conf_func:
                    conf = graduated_conf_func(state.get("candidate_dev", abs(jump)), effective_thresh)
                return max(90.0, conf), "CONFIRMED_SPIKE", "Spike confirmed (small-spike path, recovery tick retained)."

        tol = max(SPIKE_NOISE_FLOOR, SPIKE_NOISE_STD_MULTIPLIER * (noise_std or 0.0))
        
        pre_runs = state["violation_run"]
        if resid < state["best_progress"] - 1e-9:
            state["best_progress"] = resid
            state["violation_run"] = 0
        elif resid > state["best_progress"] + tol:
            state["violation_run"] += 1

        if state["elapsed_hours"] >= SPIKE_WINDOW_HOURS or state["violation_run"] >= SPIKE_VIOLATION_CONSECUTIVE_REQUIRED:
            state["status"] = "IDLE"
            return 0.0, "RECLASSIFIED", "Failed to revert."

        return 45.0, "PROVISIONAL", "Tracking candidate spike."

    return 0.0, "IDLE", ""
