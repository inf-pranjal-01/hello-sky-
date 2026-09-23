"""Verification tests for bug audit fixes."""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

# 1. Import all backend modules
from model.detect import _classify_regime, _corroborate_network, score_reading
from model.state import StateManager, StationBuffer
import main as m

print("=== Backend import: OK ===")

# 2. Test _compute_decision_basis
cases = [
    (None,  [],                              "MODEL_UNAVAILABLE"),
    (72.0,  [("drift","temperature_c",85)],  "MODEL_AND_RULE_SUPPORTED"),
    (None,  [("physical_bounds","temp",100)],"PHYSICS_ONLY"),
    (None,  [("drift","temp",85)],           "RULE_ONLY_STATISTICAL"),
    (88.0,  [],                              "MODEL_CONFIRMED"),
]
all_ok = True
print("\n--- decision_basis ---")
for model_pct, rules, expected in cases:
    got = m._compute_decision_basis(model_pct, rules)
    ok = got == expected
    status = "OK" if ok else f"FAIL (expected {expected})"
    print(f"  {str([r[0] for r in rules]):40} model={str(model_pct):6} => {got:35} {status}")
    all_ok = all_ok and ok

# 3. Test _compute_model_status
print("\n--- model_status ---")
status_cases = [
    (72.0,  None, "AVAILABLE"),
    (None,  10,   "UNAVAILABLE_WARMUP"),
    (None,  100,  "UNAVAILABLE_MISSING_FEATURES"),
    (None,  None, "UNAVAILABLE_MISSING_FEATURES"),
]
for model_pct, hist, expected in status_cases:
    got = m._compute_model_status(model_pct, hist)
    ok = got == expected
    status = "OK" if ok else f"FAIL (expected {expected})"
    print(f"  model={str(model_pct):6} hist={str(hist):5} => {got:35} {status}")
    all_ok = all_ok and ok

# 4. Test _classify_regime
print("\n--- _classify_regime ---")

history = pd.DataFrame({
    "temperature_c": [24.0, 24.5, 25.0],
    "timestamp": pd.date_range("2025-01-01", periods=3, freq="1h"),
})

def make_row(**kw):
    defaults = {
        "temperature_c": 25.0, "humidity_pct": 60.0,
        "temp_deviation": 0.1,  "humidity_deviation": 0.1,
        "temp_roc_1h": 0.0,     "temp_roc_3h": 0.0,
        "pressure_roc_3h": 0.0,
        "temp_volatility_z": 0.0,
        "pressure_volatility_z": 0.0,
        "humidity_volatility_z": 0.0,
    }
    defaults.update(kw)
    return pd.Series(defaults)

regime_cases = [
    (make_row(temperature_c=38.0, temp_deviation=3.0), {"temperature_c": 38.0, "timestamp": "2025-01-01 12:00"}, "HIGH_HEAT"),
    (make_row(humidity_pct=90.0, humidity_deviation=2.5), {"humidity_pct": 90.0, "timestamp": "2025-01-01 14:00"}, "HIGH_HUMIDITY"),
    (make_row(temp_deviation=None, humidity_deviation=None), {"temperature_c": 25.0, "timestamp": "2025-01-01 10:00"}, "UNKNOWN_INSUFFICIENT_DATA"),
    (make_row(pressure_roc_3h=3.5), {"temperature_c": 25.0, "timestamp": "2025-01-01 11:00"}, "PRESSURE_SHIFT"),
    (make_row(temp_volatility_z=2.5), {"temperature_c": 25.0, "timestamp": "2025-01-01 08:00"}, "HIGH_VOLATILITY"),
    (make_row(temp_deviation=0.05, temp_roc_3h=0.1, pressure_roc_3h=0.1), {"temperature_c": 22.0, "timestamp": "2025-01-01 02:00"}, "STABLE"),
]
for row, raw, expected in regime_cases:
    got = _classify_regime(row, raw, history)
    ok = got == expected
    status = "OK" if ok else f"FAIL (expected {expected})"
    print(f"  temp={raw.get('temperature_c',25):5} hum={raw.get('humidity_pct',60):5} => {got:35} {status}")
    all_ok = all_ok and ok

# 5. Test network state returns None when not anomaly
print("\n--- network_state None when is_anomaly=False ---")
from model.detect import PARAM_PREFIXES
# Minimal check: verify the verdict dict has network_corroboration=None when score is low
# (We don't have a live model here, just check the logic is wired correctly in detect.py)
import ast, re
with open("model/detect.py", encoding="utf-8") as f:
    src = f.read()
has_none_check = "network_state = None" in src
print(f"  detect.py network_state=None guard: {'OK' if has_none_check else 'MISSING'}")
all_ok = all_ok and has_none_check

has_target_once = "target_features = build_features_for_latest(history_df)" in src
print(f"  detect.py target_features computed once: {'OK' if has_target_once else 'MISSING'}")
all_ok = all_ok and has_target_once

# 6. Frontend type checks (grep)
print("\n--- Frontend TypeScript types ---")
with open("FRONTEND/src/types/index.ts", encoding="utf-8") as f:
    ts = f.read()
for needle, label in [
    ("decision_basis?: string", "LatestAnomaly decision_basis"),
    ("model_status?: string", "LatestAnomaly model_status"),
    ("NetworkCorroborationState", "NetworkCorroborationState type"),
]:
    ok = needle in ts
    print(f"  {label}: {'OK' if ok else 'MISSING'}")
    all_ok = all_ok and ok

print()
print("=== ALL TESTS PASSED ===" if all_ok else "=== SOME TESTS FAILED ===")
