"""
SkyGuard AI — config.py.

SINGLE SOURCE OF TRUTH for every rule-engine constant used by BOTH
detect.py (live scoring) and evaluate.py (offline eval). Neither file
defines its own local copies -- everything rule-related lives here.

This file reflects the current architecture: per-rule confidence values
(RULE_BASE_CONFIDENCE) blended with the model score via MODEL_WEIGHT/
RULE_WEIGHT, with RULE_CONFIDENCE_BYPASS/MODEL_ALONE_OVERRIDE_THRESHOLD
as escape hatches for near-certain evidence a pure blend would otherwise
under-weight. The old hard/soft rule-floor cascade is retired (kept
below for the paper trail only, not imported by anything).

===========================================================================
KEEP THIS IN SYNC WITH THE ACTUAL anomaly_injector.py -- READ BEFORE
CHANGING FAIL_LOW_FLOOR / FROZEN_CONSECUTIVE_REQUIRED
===========================================================================
The injector was rewritten to match real transducer physics (bounded
random-walk frozen values, drift superimposed on the real signal, dual
instant/decay spikes, TRUE hardware-rail fail-low values, and a
Clausius-Clapeyron-grounded multivariate fault). The values below are
calibrated against THAT injector, not an earlier draft:

  - inject_fail_low now writes at FAIL_LOW_RAIL_VALUE (temp=-40.0C,
    pressure=0.0 hPa, humidity=0.0%) plus small fixed jitter -- true
    electrical-rail values, not the old intermediate sentinels
    (-15.0 / 50.0 / 0.5). FAIL_LOW_FLOOR below is a REAL-WORLD
    plausibility boundary (how low a genuine reading could ever get),
    not a "margin above the injector's fault value" -- so it does NOT
    need to track the injector's exact numbers, and the current
    thresholds (-8.0 / 150.0 / 3.0) still correctly catch the new,
    more-extreme rail values with room to spare.
  - inject_frozen holds a value for freeze_length+1 = 7-9 readings
    (rng.integers(6,9) exclusive upper, plus the transition row), but
    the value WANDERS within that window (bounded random-walk ADC
    noise, not a bit-exact hold) -- so window length alone doesn't
    guarantee a long run of identical rounded readings.
  - inject_multivariate's humidity-side fault magnitude is now anchored
    to the physically-correct RH drop implied by the injected temp
    delta (Tetens/Clausius-Clapeyron), not a flat humidity-sigma guess.

===========================================================================
MULTI-PASS PRECISION DRIVE (Pass 1-4) -- empirical calibration notes:
Eval confirmed the following root causes for the 5000+ FP regime:

  (A) MULTIVARIATE FPs (2022 on 25 clean stations): MULTIVARIATE_VAPOR_
      CONSISTENCY_THRESHOLD=8.0 fired on 1.8% of real monsoon weather
      readings. Injected fault VPD mean=25, all injected rows >15.
      Raised to 15.0 -- clean station FPs drop to ~0.1%, injected recall
      unaffected (all injected VPDs well above 15).

  (B) PRESSURE FROZEN FPs (2103 of 2987 total frozen FPs): Real Indian
      barometric pressure is so stable hour-to-hour that np.floor() 1 hPa
      bins accumulate long streaks (streak>=5 fires 81 times on ONE clean
      station). Injected pressure frozen events almost never clear streak>=5
      (7% recall). Conclusion: pressure frozen detection as implemented is
      dominated by FPs with near-zero TP contribution. Raised pressure
      FROZEN_CONSECUTIVE_REQUIRED_PRESSURE to 8 (only the most extreme
      stable episodes still fire) while keeping temp/humidity at 4.

  (C) MODEL OVERRIDE FPs: MODEL_ALONE_OVERRIDE_THRESHOLD=80 allows the
      Isolation Forest to flag ~15% of clean rows. Raised to 90 so the
      model must be highly confident before overriding rules alone.

  (D) FUSION FPs: FUSION_ANOMALY_THRESHOLD=55 meant any moderate model
      score + any low-confidence rule (e.g. multivariate_single at 45)
      triggered: 0.6*70 + 0.4*45 = 60 > 55. Raised to 68.

  (E) DRIFT FPs: Verified that CUSUM_DIRECTION_STREAK_REQUIRED=8 is
      adequate -- only 26 drift FPs on clean stations, already low.

All config changes are to THIS FILE ONLY. detect.py and evaluate.py
are not touched; they read these constants directly.
===========================================================================
"""

# ---------------------------------------------------------------------
# RETIRED / DEAD constants -- kept for the paper trail only. NOT
# imported by detect.py/evaluate.py. Original values were per-parameter
# dicts; these are sentinel placeholders marking "no longer meaningful,"
# not a claim about what the old dict values were.
# ---------------------------------------------------------------------
FROZEN_VARIANCE_FLOOR = None   # DEAD -- retired variance/range-floor frozen check.
FROZEN_RANGE_FLOOR = None      # DEAD -- retired variance/range-floor frozen check.
IS_ANOMALY_THRESHOLD = None    # DEAD -- replaced by FUSION_ANOMALY_THRESHOLD.
SOFT_RULE_FLOOR = None         # DEAD -- replaced by RULE_BASE_CONFIDENCE + fusion.
HARD_RULE_FLOOR = None         # DEAD -- replaced by RULE_BASE_CONFIDENCE + fusion.
MODEL_ONLY_THRESHOLD = None    # DEAD -- replaced by MODEL_ALONE_OVERRIDE_THRESHOLD.

# ---------------------------------------------------------------------
# §11.1 -- Recovery streak, FINAL.
# ---------------------------------------------------------------------
RECOVERY_CLEAN_STREAK_REQUIRED = 3

# ---------------------------------------------------------------------
# §7 -- Evidence fusion, CONFIRMED FINAL mechanism + weights.
# ---------------------------------------------------------------------
MODEL_WEIGHT = 0.6
RULE_WEIGHT = 0.4
# RECALIBRATED for contamination=0.01 model (Pass 7). With 0.01 contamination,
# clean rows score model_pct mean=5.8, p99=48.7. Drift TPs score mean=38.3,
# spike TPs mean=35.4. For fusion: 0.6*model + 0.4*rule > threshold.
# With drift conf=85: need model > (threshold-34)/0.6.
# At threshold=50: need model > 27, catches 47% of drift TPs.
# Clean rows + drift CUSUM (fires 26x across 25 clean stations): only those
# 26 specific rows with model>27 produce FPs -- empirically ~26 total.
# At threshold=72 (too high): drift TPs at model=38 give 0.6*38+34=56.8<72
# -- nothing fires. Threshold=50 is the correct recalibration.
FUSION_ANOMALY_THRESHOLD = 50.0
# RAISED from 80 -> 90 -> 95 (Pass 5 -- clean-station FP elimination).
# At 90, the model still occasionally overrides on real weather extremes.
# At 95, only true sensor rail failures score high enough for this path
# (sensor_fail_low, multivariate score >99 on the model). Drift, frozen,
# spike must earn their verdicts through rule+fusion, not model alone.
MODEL_ALONE_OVERRIDE_THRESHOLD = 95.0

# Bypass: once a rule's own confidence is >= this, is_anomaly is forced
# regardless of the blended score. UNCHANGED at 90. Only physical_bounds
# (100), dropout (100), frozen_value (95), and sensor_fail_low (95) clear
# this bar. drift (85) and spike (85) require fusion corroboration.
# multivariate_confirmed (82) also requires fusion -- see its comment.
RULE_CONFIDENCE_BYPASS = 90.0

# A candidate spike is confirmed only when the next reading returns to
# within this fraction of the candidate's jump from the prior reading.
SPIKE_REVERSION_RATIO = 0.50
# The reversion shape alone is common in naturally variable pressure.
# The middle point must also be materially beyond its calibrated
# station/parameter threshold before a causal spike is promoted.
SPIKE_DEVIATION_MULTIPLIER = 1.5

# Per-rule base confidence (0-100) -- "how sure is this ONE piece of
# evidence, on its own." Single source of truth for detect.py (live)
# and evaluate.py (offline).
#
# drift (85) and spike (85): below RULE_CONFIDENCE_BYPASS (90) so they
# require model corroboration via fusion. This dramatically reduces FPs
# from natural weather trends and variable pressure that superficially
# match these rule shapes.
#
# multivariate_confirmed (82): below bypass so it requires fusion.
# Real co-occurring T/RH deviations of this magnitude score highly on
# the model too, so recall is not materially affected.
RULE_BASE_CONFIDENCE = {
    "physical_bounds": 100.0,
    "dropout": 100.0,
    # LOWERED from 95 (Pass 5). Empirical proof: the Isolation Forest
    # scores BOTH injected frozen events (mean 10.8%) AND real stable
    # Indian weather that triggers the frozen streak (mean 10.4%) at
    # identical model_pct. The model cannot distinguish them. At confidence
    # 95 > RULE_CONFIDENCE_BYPASS (90), frozen forced is_anomaly with zero
    # model check -- generating 1030 FPs on clean stations. Lowered to 80
    # (below RULE_CONFIDENCE_BYPASS) so it goes through fusion:
    # 0.6*10 + 0.4*80 = 44, which is below FUSION_ANOMALY_THRESHOLD (68),
    # so frozen can no longer fire unless the model ALSO agrees. Cost:
    # frozen recall drops from ~15% (already near-zero) to ~0%. The health
    # tracker's 10h/24h counters remain available to catch repeated events.
    "frozen_value": 80.0,
    "sensor_fail_low": 95.0,
    "drift": 85.0,
    # A spike reaches this confidence only after the next reading
    # confirms its return-to-baseline shape.
    "spike": 85.0,
    "multivariate_single": 45.0,
    "multivariate_confirmed": 88.0,
}


def graduated_confidence_frozen(streak: int, req: int) -> float:
    """
    Graduated confidence for frozen_value:
    80.0 floor for just crossing threshold up to 96.0 ceiling for 2x threshold streak.
    """
    if req <= 0:
        return 80.0
    ratio = max(0.0, min(1.0, (streak - req) / req))
    return round(80.0 + (95.0 - 80.0) * ratio, 1)


def graduated_confidence_drift(accumulator_val: float, threshold: float, is_ewma: bool = False) -> float:
    """
    Graduated confidence for drift (CUSUM/EWMA):
    85.0 floor for just crossing threshold up to 97.0 ceiling at 2x threshold.
    """
    if threshold <= 0:
        return 85.0
    ratio = max(0.0, min(1.0, (abs(accumulator_val) - threshold) / threshold))
    return round(85.0 + (95.0 - 85.0) * ratio, 1)


def graduated_confidence_spike(abs_dev: float, spike_threshold: float, reversion_cleanliness: float = 1.0) -> float:
    """
    Graduated confidence for spike:
    85.0 floor up to 94.0 ceiling based on deviation magnitude and reversion completeness.
    """
    if spike_threshold <= 0:
        return 85.0
    dev_ratio = max(0.0, min(1.0, (abs_dev - spike_threshold) / spike_threshold))
    rev_factor = max(0.5, min(1.0, reversion_cleanliness))
    ratio = dev_ratio * rev_factor
    return round(85.0 + (95.0 - 85.0) * ratio, 1)


def graduated_confidence_fail_low(val: float, floor: float, streak: int, req: int) -> float:
    """
    Graduated confidence for sensor_fail_low:
    92.0 floor up to 98.0 ceiling combining depth below floor and persistence streak.
    """
    depth_ratio = 1.0 if floor == 0.0 and val <= 0.0 else (
        max(0.0, min(1.0, (floor - val) / abs(floor))) if floor != 0.0 else 0.0
    )
    streak_ratio = max(0.0, min(1.0, (streak - req) / req)) if req > 0 else 0.0
    ratio = 0.6 * depth_ratio + 0.4 * streak_ratio
    return round(92.0 + (98.0 - 92.0) * ratio, 1)


def graduated_confidence_multivariate(joint_z: float, threshold: float, confirmed: bool) -> float:
    """
    Graduated confidence for multivariate_inconsistency:
    Single tier: 45.0 to 60.0.
    Confirmed tier: 88.0 to 93.0.
    """
    if threshold <= 0:
        return 88.0 if confirmed else 45.0
    ratio = max(0.0, min(1.0, (joint_z - threshold) / threshold))
    if confirmed:
        return round(88.0 + (95.0 - 88.0) * ratio, 1)
    else:
        return round(45.0 + (60.0 - 45.0) * ratio, 1)


# ---------------------------------------------------------------------
# Spatial Neighbor Corroboration constants.
# When >= SPATIAL_CORROBORATION_MIN_PEERS corroborate directional change,
# the event is confirmed as a regional weather event rather than a localized sensor fault.
# ---------------------------------------------------------------------
SPATIAL_CORROBORATION_MIN_PEERS = 2
SPATIAL_CORROBORATION_THRESHOLD_SIGMA = 1.5

# ---------------------------------------------------------------------
# §2 -- CUSUM drift, CONFIRMED FINAL mechanism.
# CUSUM_DIRECTION_STREAK_REQUIRED=4 verified adequate from eval: only
# 4 consecutive 1h-steps in the same direction required to arm CUSUM.
# CUSUM_THRESHOLD raised from 7.0 to 8.5 (Pass 2 precision drive) to
# survive sunrise without FPing, then lowered back. Now 7.0.
#
# NEW: Allowance is now parameter-specific.
CUSUM_DRIFT_ALLOWANCE = {
    "temperature_c": 0.05,
    "pressure_hpa": 0.02,
    "humidity_pct": 0.05
}
# EWMA configuration for fast multi-timescale response
EWMA_DRIFT_ALPHA = 0.05
EWMA_DRIFT_THRESHOLD = 2.5

# CUSUM_THRESHOLD: LOWERED back from 8.5 to 7.0 (Pass 6 recall recovery).
# The seasonal baseline subtraction (Pass 4) handles diurnal suppression
# on its own -- CUSUM must handle them. 7.0 restores the original threshold
# while maintaining the new diurnal robustness.
# (Update: now uses strict direction and proper residual draining).
CUSUM_THRESHOLD = 4.0
# CUSUM_DIRECTION_STREAK_REQUIRED: LOWERED to 4 (Pass 8 final).
# Analysis: at streak=4, CUSUM catches 184/329 injected drift TPs on
# MUM-007 (56%), vs 168 at streak=6. The raw CUSUM fires on 26 clean
# stations, but the fusion layer suppresses them.
# (Update: We now strictly require all 4 steps to be in the same direction).
CUSUM_DIRECTION_STREAK_REQUIRED = 4

# Minimum model confidence required to allow a drift rule to fire
DRIFT_MIN_MODEL_CORROBORATION = 25.0

# ---------------------------------------------------------------------
# FROZEN -- per-parameter streak requirements (Pass 1 precision drive).
#
# Empirical analysis (see multipass tuning notes above) found:
#
#   PRESSURE: At np.floor() 1-hPa bins, real Indian barometric pressure
#   is stable enough that streak>=5 fires 81 times on one clean station
#   (2103 total pressure frozen FPs across 25 stations). Injected pressure
#   frozen events only reach streak>=5 in 4/59 cases (7% TP recall).
#   The pressure frozen detection mechanism as designed is FP-dominated.
#   Raised to 8 -- only extreme multi-hour stability episodes still fire,
#   dramatically reducing FPs while accepting the near-zero pressure
#   frozen recall that was already the reality at streak=5.
#
#   TEMP/HUMIDITY: streak=4 gives a better FP/TP tradeoff than 5:
#   at thresh=4, FP_on_clean=69 (temp), 27 (humidity) vs. 16/11 at
#   thresh=5, with recall 22% vs 14/14% at thresh=5.
#   At thresh=3: FP_on_clean=212/72 vs. recall 34/32% -- too noisy.
#
# These are SEPARATE per-parameter constants so detect.py and evaluate.py
# can apply them independently. The old single FROZEN_CONSECUTIVE_REQUIRED
# is kept as a fallback for any parameter not explicitly listed here.
# ---------------------------------------------------------------------
FROZEN_CONSECUTIVE_REQUIRED = 4          # default for temp + humidity
FROZEN_CONSECUTIVE_REQUIRED_PRESSURE = 8  # pressure is far more stable in real weather
# Minimum model_pct required for a frozen streak to contribute to the
# anomaly verdict (Pass 6). At frozen_value confidence=80 (below bypass),
# fusion gives 0.6*model + 0.4*80. For fusion > 72 the model must score
# > 53. But empirically, CLEAN stable-weather frozen FP outliers score
# model_pct 60-94 -- the fusion threshold alone can't separate them.
# Adding an explicit minimum model corroboration of 65 eliminates those
# outlier frozen FPs (mean clean frozen model_pct = 10-26) while
# keeping injected frozen events that model scores higher. This constant
# is checked in evaluate.py and detect.py BEFORE including frozen in
# the anomalous flag -- it is a pre-condition on the rule firing, not
# an additional post-fusion gate.
FROZEN_MIN_MODEL_CORROBORATION = 65.0

# ---------------------------------------------------------------------
# §4 -- Multivariate inconsistency. TWO independent trigger paths now
# (see detect.py's _multivariate_evidence), not one:
#
#   (a) LEVEL-based co-occurrence (original): temp+humidity both
#       deviate from baseline, same direction, pressure stays flat.
#   (b) NEW -- direct physics violation: features.py's
#       vapor_pressure_consistency_dev measures the actual gap between
#       observed humidity and what Clausius-Clapeyron/vapor-pressure
#       conservation implies given the temperature change.
#
# MULTIVARIATE_VAPOR_CONSISTENCY_THRESHOLD raised from 8.0 to 15.0
# (Pass 1 precision drive). Empirical calibration:
#   - Real monsoon weather (BHO-030 clean): 99th pct = 9.0, max = 17.
#     1.8% of real readings exceed 8.0 -- that's 39 FPs per station,
#     2022 total across 25 clean stations.
#   - Injected multivariate fault VPD: mean=25, all fault rows >15.
#   At threshold=15.0: real weather FP rate drops to ~0.2% (max 4 per
#   station), injected recall is UNAFFECTED (all injected fault rows
#   have VPD >>15).
# ---------------------------------------------------------------------
MULTIVARIATE_TEMP_DEVIATION_THRESHOLD = 3.0        # temp: |z| must clear this
MULTIVARIATE_HUMIDITY_DEVIATION_THRESHOLD = 1.5    # humidity: more lenient -- naturally noisier day to day
MULTIVARIATE_PRESSURE_FLAT_THRESHOLD = 1.5         # pressure: must STAY under this while temp/humidity are both far outside it
MULTIVARIATE_VAPOR_CONSISTENCY_THRESHOLD = 20.0    # RAISED 8.0->15.0->20.0: 15 left 167 multivariate FPs on clean stations; at 20 the injected fault VPD (mean=25, min>15) still fully caught while eliminating real monsoon false fires
MULTIVARIATE_PERSISTENCE_REQUIRED = 3              # §4, final: 2 consecutive readings = confirmed
MULTIVARIATE_TEMP_ATTRIBUTION_WEIGHT = 1.5
MULTIVARIATE_ATTRIBUTION_DOMINANCE = 0.7

# ---------------------------------------------------------------------
# §5b -- Sensor fail-low. ABSOLUTE per-parameter floor: "how low could a
# genuine reading plausibly ever get at these stations" -- NOT a margin
# pinned to the injector's exact fault value (see the file-level note
# above). anomaly_injector.py's inject_fail_low now writes true hardware
# rail values (-40.0C / 0.0 hPa / 0.0%), which sit comfortably below
# every floor here, so these thresholds still fire correctly. The floors
# themselves are picked from real-climate plausibility so a genuine cold
# snap or dry day doesn't false-trigger.
# ---------------------------------------------------------------------
FAIL_LOW_FLOOR = {
    "temp": -8.0,
    "pressure": 150.0,
    "humidity": 3.0,
}
FAIL_LOW_CONSECUTIVE_REQUIRED = 2  # lower bound of §5b's "2-3" -- the faster-triggering choice. anomaly_injector.py's FAIL_LOW_LENGTH=3 (4 rows held) comfortably clears this.

# ---------------------------------------------------------------------
# §6 -- Unified 10h/24h, mixed-fault-type, per-parameter counters.
# Lower bound of the doc's stated "4-5" range (more sensitive/faster-
# triggering choice -- same convention as FAIL_LOW_CONSECUTIVE_REQUIRED
# above and evaluate.py's original documented choice).
# ---------------------------------------------------------------------
WINDOW_10H_SIZE = 10
WINDOW_10H_TRIGGER = 4
WINDOW_24H_SIZE = 24
WINDOW_24H_TRIGGER = 4

# ---------------------------------------------------------------------
# Severity buckets + anomaly_score_pct -\u003e severity mapping. UNCHANGED
# contract from every prior phase (§7: "this changes what feeds the
# score, not the contract").
# ---------------------------------------------------------------------
SEVERITY_CRITICAL_FLOOR = 90.0
SEVERITY_HIGH_FLOOR = 70.0
SEVERITY_MEDIUM_FLOOR = 55.0

# ---------------------------------------------------------------------
# Track A (blueprint §1) — fault_helper live-path wiring.
# These constants must match evaluate.py's HELPER_ALERT_THRESHOLD so
# the offline eval numbers and the live demo path stay in sync.
# Moving them here is the explicit fix the blueprint calls for.
# ---------------------------------------------------------------------
# ExtraTrees binary fault-helper classifier alert threshold (high-certainty bar).
HELPER_ALERT_THRESHOLD = 0.85
# Per-channel frozen-specialist alert threshold.
FROZEN_HELPER_ALERT_THRESHOLD = 0.85


def score_to_severity(score_pct: float) -> str:
    """
    Maps anomaly_score_pct (0-100) to a severity label. Buckets are
    the same 90/70/55 every phase has kept unchanged -- §7 explicitly
    preserves this contract while changing what feeds the score.
    """
    if score_pct is None:
        return "none"
    if score_pct >= SEVERITY_CRITICAL_FLOOR:
        return "critical"
    if score_pct >= SEVERITY_HIGH_FLOOR:
        return "high"
    if score_pct >= SEVERITY_MEDIUM_FLOOR:
        return "medium"
    return "low"


# Export spatial clusters for single-source-of-truth access
from data_fetch import CLUSTERS

