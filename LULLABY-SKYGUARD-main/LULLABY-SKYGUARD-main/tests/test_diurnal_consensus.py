import sys
sys.path.insert(0, '.')
from model.detect import _apply_diurnal_consensus_filter, PARAM_PREFIXES
import pandas as pd

def make_spike(param, confidence=87.0):
    return {"type": "spike", "parameter": param, "confidence": confidence,
            "observed_value": 15.9, "threshold": ">3.0", "reason": "Jump."}

def make_peer(t0, t1):
    return pd.DataFrame({
        "temperature_c": [t0, t1],
        "pressure_hpa": [1008.0, 1008.1],
        "humidity_pct": [80.0, 77.0],
        "timestamp": ["2025-01-04 09:00:00", "2025-01-04 10:00:00"],
    })

feat = pd.Series({"temp_roc_1h": 3.2, "pressure_roc_1h": 0.1, "humidity_roc_1h": -3.0})

# TEST 1: All 3 peers warm up → spike dampened
fired = [make_spike("temperature_c", 87.0)]
nb = {"A": make_peer(11.8, 14.9), "B": make_peer(12.1, 15.4), "C": make_peer(11.5, 14.2)}
res, smap = _apply_diurnal_consensus_filter(fired, nb, feat)
r = res[0]
print(f"TEST1: conf={r['confidence']} expect<40, diurnal_consensus={r.get('diurnal_consensus')}")
assert r["confidence"] < 40.0, f"FAIL {r['confidence']}"
assert r.get("diurnal_consensus") is True
print("TEST1 PASS")

# TEST 2: Only 1/3 peers agree → no dampening
fired2 = [make_spike("temperature_c", 87.0)]
nb2 = {"A": make_peer(15.0, 11.5), "B": make_peer(12.1, 15.4), "C": make_peer(13.0, 9.8)}
res2, smap2 = _apply_diurnal_consensus_filter(fired2, nb2, feat)
r2 = res2[0]
print(f"TEST2: conf={r2['confidence']} expect=87.0, diurnal_consensus={r2.get('diurnal_consensus')}")
assert r2["confidence"] == 87.0, f"FAIL {r2['confidence']}"
print("TEST2 PASS")

# TEST 3: Flat peers below min_roc → only 1 eligible → no suppression
fired3 = [make_spike("temperature_c", 87.0)]
nb3 = {"A": make_peer(12.1, 12.4), "B": make_peer(13.0, 13.2), "C": make_peer(12.5, 15.3)}
res3, smap3 = _apply_diurnal_consensus_filter(fired3, nb3, feat)
r3 = res3[0]
print(f"TEST3: conf={r3['confidence']} expect=87.0 (not enough eligible peers)")
assert r3["confidence"] == 87.0, f"FAIL {r3['confidence']}"
print("TEST3 PASS")

# TEST 4: drift and frozen untouched when spike is dampened
fired4 = [
    make_spike("temperature_c", 87.0),
    {"type": "drift", "parameter": "temperature_c", "confidence": 87.0, "reason": "cusum"},
    {"type": "frozen_value", "parameter": "humidity_pct", "confidence": 80.0, "reason": "frozen"},
]
nb4 = {"A": make_peer(11.8, 14.9), "B": make_peer(12.1, 15.4), "C": make_peer(11.5, 14.2)}
res4, smap4 = _apply_diurnal_consensus_filter(fired4, nb4, feat)
spike_r = next(r for r in res4 if r["type"] == "spike")
drift_r = next(r for r in res4 if r["type"] == "drift")
frozen_r = next(r for r in res4 if r["type"] == "frozen_value")
print(f"TEST4: spike={spike_r['confidence']}<40, drift={drift_r['confidence']}=87, frozen={frozen_r['confidence']}=80")
assert spike_r["confidence"] < 40.0
assert drift_r["confidence"] == 87.0
assert frozen_r["confidence"] == 80.0
print("TEST4 PASS")

# Math check
suppressed = 87.0 * 0.38
fused_clean = 0.6 * 20 + 0.4 * suppressed
fused_fault = 0.6 * 89 + 0.4 * suppressed
print()
print(f"=== MATH CHECK ===")
print(f"Suppressed spike conf: {suppressed:.1f}")
print(f"Fused @ model=20 (clean weather): {fused_clean:.1f} (threshold=45 => anomaly={fused_clean>45})")
print(f"Fused @ model=89 (real sensor fault): {fused_fault:.1f} (threshold=45 => anomaly={fused_fault>45})")
print()
print("ALL TESTS PASSED")
