import math
import numpy as np
from typing import Tuple

# The ground-truth injector uses tau=0.5 to 1.5, and a tail length of 2 to 5.
# At tau=1.5, it takes ~1 hour to hit 50%, and ~4.5 hours to hit 5% (complete decay).
# We set the window to safely cover the maximum tail.
SPIKE_WINDOW_HOURS = 5
SPIKE_DECAY_RATIO = 0.50

def init_spike_state() -> dict:
    return {
        "status": "IDLE",
        "t0_val": 0.0,
        "peak_val": 0.0,
        "jump": 0.0,
        "ticks": 0,
        "monotonic_violations": 0,
        "prev_val": None,
        "candidate_dev": 0.0
    }

def step_spike_state(
    val: float,
    dev: float,
    spike_thresh: float,
    multiplier: float,
    state: dict,
    graduated_conf_func=None
) -> Tuple[float, str, str]:
    """
    State machine for spike detection.
    Returns (confidence: float, status: str, reason: str)
    """
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
                # Jump detected -> Enter PROVISIONAL state
                state["status"] = "PROVISIONAL"
                state["t0_val"] = prev_val
                state["peak_val"] = val
                state["jump"] = val - prev_val
                state["ticks"] = 0
                state["monotonic_violations"] = 0
                state["candidate_dev"] = abs_dev
                return 40.0, "PROVISIONAL", f"Candidate jump of {step_diff:.1f} detected. Tracking for reversion/decay."
        return 0.0, "IDLE", ""

    elif status == "PROVISIONAL":
        state["ticks"] += 1
        jump = state["jump"]
        t0 = state["t0_val"]
        
        dist_from_t0 = abs(val - t0)

        # 1. Check for Reversion (Fast or Decay)
        if dist_from_t0 <= abs(jump) * SPIKE_DECAY_RATIO:
            state["status"] = "IDLE"
            conf = 95.0
            if graduated_conf_func:
                conf = graduated_conf_func(state.get("candidate_dev", abs(jump)), spike_thresh * multiplier)
            return max(90.0, conf), "CONFIRMED_SPIKE", f"Spike confirmed: mathematically reverted within {state['ticks']} hours."

        # 2. Track Monotonic Decay
        if jump > 0 and val > prev_val + 0.2:
            state["monotonic_violations"] += 1
        elif jump < 0 and val < prev_val - 0.2:
            state["monotonic_violations"] += 1

        # 3. Handle Timeouts and Violations (Reclassification)
        if state["ticks"] >= SPIKE_WINDOW_HOURS or state["monotonic_violations"] > 1:
            state["status"] = "IDLE"
            # Return 0 confidence so it gets handed off natively to CUSUM Drift / Sustained Rules
            return 0.0, "RECLASSIFIED", f"Failed to revert within window or plateaued. Reclassified."

        # 4. Still decaying/waiting within window
        return 45.0, "PROVISIONAL", f"Tracking candidate spike... (Hour {state['ticks']}/{SPIKE_WINDOW_HOURS})"

    return 0.0, "IDLE", ""
