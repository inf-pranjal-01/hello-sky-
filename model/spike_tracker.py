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
                
                # Log ENTRY
                if kwargs.get('ts'):
                    with open("scratch/spike_events.jsonl", "a") as f:
                        f.write(json.dumps({"event": "ENTER", "station": kwargs.get('station'), "param": kwargs.get('param'), "ts": kwargs.get('ts'), "jump": state["jump"], "t0_val": prev_val, "peak_val": val, "thresh": spike_thresh}) + "\n")
                        
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
            conf = 95.0
            if graduated_conf_func:
                conf = graduated_conf_func(state.get("candidate_dev", abs(jump)), spike_thresh * multiplier)
            
            if kwargs.get('ts'):
                with open("scratch/spike_events.jsonl", "a") as f:
                    f.write(json.dumps({"event": "CONFIRM", "station": kwargs.get('station'), "param": kwargs.get('param'), "ts": kwargs.get('ts'), "ticks": state["ticks"], "resid": resid, "elapsed_hours": state["elapsed_hours"]}) + "\n")
                    
            return max(90.0, conf), "CONFIRMED_SPIKE", f"Spike confirmed."

        tol = max(SPIKE_NOISE_FLOOR, SPIKE_NOISE_STD_MULTIPLIER * (noise_std or 0.0))
        
        pre_runs = state["violation_run"]
        if resid < state["best_progress"] - 1e-9:
            state["best_progress"] = resid
            state["violation_run"] = 0
        elif resid > state["best_progress"] + tol:
            state["violation_run"] += 1
            
        if kwargs.get('ts'):
            with open("scratch/spike_events.jsonl", "a") as f:
                f.write(json.dumps({"event": "TICK", "station": kwargs.get('station'), "param": kwargs.get('param'), "ts": kwargs.get('ts'), "ticks": state["ticks"], "val": val, "t0": t0, "exp_now": expected_now, "resid": resid, "best": state["best_progress"], "tol": tol, "runs": state["violation_run"], "eroc": expected_roc, "drift": state["baseline_drift"]}) + "\n")

        if state["elapsed_hours"] >= SPIKE_WINDOW_HOURS or state["violation_run"] >= SPIKE_VIOLATION_CONSECUTIVE_REQUIRED:
            reclass_reason = "TIMEOUT" if state["elapsed_hours"] >= SPIKE_WINDOW_HOURS else "VIOLATION"
            state["status"] = "IDLE"
            
            if kwargs.get('ts'):
                with open("scratch/spike_events.jsonl", "a") as f:
                    f.write(json.dumps({"event": "RECLASSIFY", "reason": reclass_reason, "station": kwargs.get('station'), "param": kwargs.get('param'), "ts": kwargs.get('ts'), "ticks": state["ticks"], "elapsed": state["elapsed_hours"]}) + "\n")
                    
            return 0.0, "RECLASSIFIED", "Failed to revert."

        return 45.0, "PROVISIONAL", "Tracking candidate spike."

    return 0.0, "IDLE", ""
