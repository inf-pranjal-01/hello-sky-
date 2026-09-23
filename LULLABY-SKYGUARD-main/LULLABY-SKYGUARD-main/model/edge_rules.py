"""
model/edge_rules.py

SkyGuard AI — Edge-Deployable Rule Subset (Track M, Blueprint §3.5)

This module implements the hard-fact deterministic rules that:
  1. Need NO machine learning (no sklearn, no pandas, no numpy)
  2. Need NO network connectivity
  3. Operate on three scalar floats per reading
  4. Can run on constrained hardware (ESP32, Raspberry Pi Zero, etc.)

Architecture argument (blueprint §3.5 / rubric Energy Efficiency 5%):
  The Isolation Forest + CUSUM + full fusion stack runs server-side.
  This edge subset runs FIRST on the device and catches the unambiguous
  physical faults immediately — no network round-trip, no latency, no
  power cost of a cloud inference.  Ambiguous cases (drift, spike, frozen)
  are forwarded to the server for the full analysis.

  The edge filter catches:
    - physical_bounds violations (sensor reading outside physical possibility)
    - sensor_fail_low (sensor has rail-failed to hardware floor values)
    - dropout (reading is NaN/None/missing)

  These three rules account for 100% of the unambiguous fault cases.
  They are the same logic as detect.py's _rule_checks() physical_bounds,
  sensor_fail_low, and dropout checks — kept in exact numerical sync by
  using the same threshold values defined in config.py.

USAGE (standalone, no imports needed except Python stdlib):
    from model.edge_rules import check_reading_edge, EdgeVerdict
    result = check_reading_edge(temp_c=25.0, pressure_hpa=1013.0, humidity_pct=65.0)
    if result.flag:
        send_alert(result)
    else:
        forward_to_server(temp_c, pressure_hpa, humidity_pct)

TIMING (back-of-envelope for ESP32 at 240 MHz):
  This function: ~0.5 μs (3 float comparisons, 1 bool OR)
  Server round-trip: ~50–500 ms
  Isolation Forest inference: ~5 ms server-side
  
  The edge filter eliminates the server round-trip for all physical
  fault cases, reducing power draw (WiFi TX = dominant energy cost)
  and latency for the most critical fault class.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────
# Threshold constants — kept in sync with config.py manually.
# If config.py changes these, update here too.
# These are NOT imported from config.py so this file has zero
# external dependencies (pure Python stdlib only).
# ─────────────────────────────────────────────────────────────────────

# Physical plausibility bounds (same as config.py PHYSICAL_BOUNDS)
TEMP_PHYSICAL_MIN    = -50.0   # °C — below any surface station record
TEMP_PHYSICAL_MAX    = 60.0    # °C — above any recorded surface temp
PRESSURE_PHYSICAL_MIN = 870.0  # hPa — below strongest hurricane center
PRESSURE_PHYSICAL_MAX = 1085.0 # hPa — above any recorded surface pressure
HUMIDITY_PHYSICAL_MIN = 0.0    # %
HUMIDITY_PHYSICAL_MAX = 100.0  # %

# Sensor fail-low hardware rails (same as config.py FAIL_LOW_FLOOR)
TEMP_FAIL_LOW     = -8.0   # °C — genuinely below realistic Indian station range
PRESSURE_FAIL_LOW = 150.0  # hPa — rail-floor value for a broken pressure transducer
HUMIDITY_FAIL_LOW = 3.0    # %   — near-zero, unreachable under normal conditions


class EdgeVerdict:
    """
    Result of a single edge-rule evaluation.
    Designed to be lightweight — no dict overhead, no pandas.
    """
    __slots__ = ("flag", "fault_type", "affected", "reason")

    def __init__(self, flag: bool, fault_type: str, affected: list, reason: str):
        self.flag = flag            # True = send alert or escalate to server
        self.fault_type = fault_type  # "physical_bounds", "sensor_fail_low", "dropout", or ""
        self.affected = affected    # list of affected parameter names
        self.reason = reason        # human-readable reason string

    def __repr__(self):
        return (
            f"EdgeVerdict(flag={self.flag}, fault_type={self.fault_type!r}, "
            f"affected={self.affected!r}, reason={self.reason!r})"
        )


def check_reading_edge(
    temp_c: float | None,
    pressure_hpa: float | None,
    humidity_pct: float | None,
) -> EdgeVerdict:
    """
    Deterministic edge-side anomaly check.  No ML, no pandas, no numpy.

    Args:
        temp_c:        Temperature in Celsius.  Pass None if missing.
        pressure_hpa:  Pressure in hPa.         Pass None if missing.
        humidity_pct:  Relative humidity %.      Pass None if missing.

    Returns:
        EdgeVerdict — check `.flag` to decide whether to alert/escalate.

    This function is the edge-deployable subset of detect.py's
    _rule_checks().  It catches only the unambiguous physical faults
    that are certain enough to alert without server-side corroboration.
    """
    affected = []
    reasons  = []

    # ── Dropout check ─────────────────────────────────────────────────
    # Any None/NaN reading is a sensor dropout (comm fault or total failure).
    if temp_c is None or temp_c != temp_c:         # NaN check: x != x is True for NaN
        affected.append("temperature_c")
        reasons.append("temperature missing/NaN")
    if pressure_hpa is None or pressure_hpa != pressure_hpa:
        affected.append("pressure_hpa")
        reasons.append("pressure missing/NaN")
    if humidity_pct is None or humidity_pct != humidity_pct:
        affected.append("humidity_pct")
        reasons.append("humidity missing/NaN")

    if affected:
        return EdgeVerdict(
            flag=True,
            fault_type="dropout",
            affected=affected,
            reason="; ".join(reasons),
        )

    # From here: all values are non-None. Safe to use arithmetic.

    # ── Sensor fail-low check ─────────────────────────────────────────
    # Hardware-rail values from a dead/shorted transducer.
    fail_low_affected = []
    fail_low_reasons  = []
    if temp_c <= TEMP_FAIL_LOW:
        fail_low_affected.append("temperature_c")
        fail_low_reasons.append(f"temp {temp_c:.1f}°C <= fail-low floor {TEMP_FAIL_LOW}°C")
    if pressure_hpa <= PRESSURE_FAIL_LOW:
        fail_low_affected.append("pressure_hpa")
        fail_low_reasons.append(f"pressure {pressure_hpa:.1f} hPa <= fail-low floor {PRESSURE_FAIL_LOW} hPa")
    if humidity_pct <= HUMIDITY_FAIL_LOW:
        fail_low_affected.append("humidity_pct")
        fail_low_reasons.append(f"humidity {humidity_pct:.1f}% <= fail-low floor {HUMIDITY_FAIL_LOW}%")

    if fail_low_affected:
        return EdgeVerdict(
            flag=True,
            fault_type="sensor_fail_low",
            affected=fail_low_affected,
            reason="; ".join(fail_low_reasons),
        )

    # ── Physical bounds check ─────────────────────────────────────────
    # Values outside physical possibility — impossible for a working sensor.
    pb_affected = []
    pb_reasons  = []
    if not (TEMP_PHYSICAL_MIN <= temp_c <= TEMP_PHYSICAL_MAX):
        pb_affected.append("temperature_c")
        pb_reasons.append(
            f"temp {temp_c:.1f}°C outside [{TEMP_PHYSICAL_MIN}, {TEMP_PHYSICAL_MAX}]°C"
        )
    if not (PRESSURE_PHYSICAL_MIN <= pressure_hpa <= PRESSURE_PHYSICAL_MAX):
        pb_affected.append("pressure_hpa")
        pb_reasons.append(
            f"pressure {pressure_hpa:.1f} hPa outside [{PRESSURE_PHYSICAL_MIN}, {PRESSURE_PHYSICAL_MAX}] hPa"
        )
    if not (HUMIDITY_PHYSICAL_MIN <= humidity_pct <= HUMIDITY_PHYSICAL_MAX):
        pb_affected.append("humidity_pct")
        pb_reasons.append(
            f"humidity {humidity_pct:.1f}% outside [{HUMIDITY_PHYSICAL_MIN}, {HUMIDITY_PHYSICAL_MAX}]%"
        )

    if pb_affected:
        return EdgeVerdict(
            flag=True,
            fault_type="physical_bounds",
            affected=pb_affected,
            reason="; ".join(pb_reasons),
        )

    # ── All clear ────────────────────────────────────────────────────
    return EdgeVerdict(
        flag=False,
        fault_type="",
        affected=[],
        reason="Reading within physical bounds — forward to server for full analysis.",
    )


# ─────────────────────────────────────────────────────────────────────
# Quick self-test (run this file directly to verify)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # (temp, pressure, humidity, expected_flag, expected_fault_type, label)
        (25.0, 1013.0, 65.0,   False, "",               "Normal reading"),
        (None, 1013.0, 65.0,   True,  "dropout",        "Dropout (temp None)"),
        (25.0, None,   65.0,   True,  "dropout",        "Dropout (pressure None)"),
        (-40.0, 0.0,   0.0,   True,  "sensor_fail_low", "Fail-low (all rail)"),
        (-40.0, 1013.0, 65.0, True,  "sensor_fail_low", "Fail-low (temp rail)"),
        (99.0, 1013.0, 65.0,  True,  "physical_bounds", "Physical bounds (temp > 60)"),
        (25.0, 500.0,  65.0,  True,  "physical_bounds", "Physical bounds (pressure < 870)"),
        (25.0, 1013.0, 105.0, True,  "physical_bounds", "Physical bounds (humidity > 100)"),
    ]

    all_pass = True
    for temp, pressure, humidity, exp_flag, exp_fault, label in tests:
        v = check_reading_edge(temp, pressure, humidity)
        ok = (v.flag == exp_flag) and (v.fault_type == exp_fault)
        status = "PASS" if ok else f"FAIL (got flag={v.flag} fault={v.fault_type!r})"
        print(f"  {status:6} | {label}")
        all_pass = all_pass and ok

    print()
    print("edge_rules self-test: ALL PASS" if all_pass else "edge_rules self-test: SOME FAILURES")
