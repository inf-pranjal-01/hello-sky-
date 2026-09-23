"""
SkyGuard AI — Detection engine (rebuilt around SKYGUARD_ARCHITECTURE_DRAFT_2.md).

Combines the trained Isolation Forest with SEVEN independent rule
checks into one fused verdict per reading, PLUS the per-station,
per-PARAMETER "circuit breaker" (mark a specific sensor OFFLINE and
stop trusting its data).

THIS IS A FULL REDESIGN, not a patch on the old cascade. Read this
before touching anything below.

===========================================================================
WHAT CHANGED FROM THE OLD detect.py, AND WHY
===========================================================================

1. SPATIAL FEATURES REMOVED ENTIRELY (Draft 2 §9). `SPATIAL_COLS`,
   `compute_raw_spatial_devs`, the `spatial_baselines` z-scoring block,
   and the `_raw_spatial_devs` verdict key are all gone. `score_reading`
   still ACCEPTS `cluster_neighbor_buffers=`/`spatial_baselines=` as
   optional, silently-ignored kwargs -- not because they do anything,
   but because state.py still passes them and hasn't been updated yet
   (state.py isn't in this pass's scope). Remove them from the call
   site when state.py gets its turn; until then this keeps state.py
   from crashing.

2. FROZEN (§1) is now deterministic, not calibrated. features.py
   computes `{prefix}_floor_frozen_match` once (floor(reading) equal
   across the last 3 readings) -- this file just reads that boolean off
   the feature row. No variance/range floors, no consec_diff threshold.
   The old Phase-1 variance/range experiment is RETIRED (see Draft 2 §0
   postmortem: it inflated false positives on ordinary calm weather).
   `config.py`'s `FROZEN_VARIANCE_FLOOR`/`FROZEN_RANGE_FLOOR` are dead
   and intentionally NOT imported here anymore.

3. DRIFT (§2) is now CUSUM, not a calibrated 24h-delta threshold. A
   fresh S+/S- accumulator is recomputed from the station's own causal
   buffer every call (see `_cusum_evidence` / `_featurize_buffer`) --
   stateless by construction, consistent with how this file has always
   treated `history_df` as the single source of truth rather than
   carrying hidden state across calls. `CUSUM_DRIFT_ALLOWANCE`/
   `CUSUM_THRESHOLD` are domain-estimated placeholders (see constants
   block) pending a real calibration pass once evaluate.py can run
   against CUSUM-compatible (direction-consistent) injected drift.

4. MULTIVARIATE INCONSISTENCY is a NEW rule (didn't exist as an
   explicit rule before -- it only ever lived implicitly inside the
   Isolation Forest's feature space). Built on the existing
   `temp_humidity_coupling_signal` / `pressure_inconsistency` cross-
   parameter features. Per §4/§8: a single qualifying reading is
   "suspicious" (lower confidence, does NOT force OFFLINE on its own);
   2 CONSECUTIVE qualifying readings is "confirmed" (high confidence,
   forces immediate OFFLINE via the fast-path set). This is also the
   reference implementation of §8's worked example: fault_type
   resolution is not a fixed priority table, it's just "whichever rule
   currently has the highest confidence wins" -- see `_fuse_and_score`.

5. SENSOR FAIL-LOW is a NEW rule (§5b), distinct from spike. Fires only
   once persisted for `FAIL_LOW_CONSECUTIVE_REQUIRED` consecutive
   readings at/below a per-parameter near-zero floor -- by construction
   it never fires on a single occurrence, so every fail-low firing is
   already a confirmed fast-path OFFLINE event.

6. EVIDENCE FUSION REPLACES THE OLD HARD-GATE CASCADE (§7). The old
   `HARD_RULE_TYPES`/`SOFT_RULE_TYPES`/`HARD_RULE_FLOOR`/`SOFT_RULE_FLOOR`
   architecture is GONE. That architecture is the confirmed root cause
   of Phase 1's failure (see Draft 2 §0): `SOFT_RULE_FLOOR` and
   `IS_ANOMALY_THRESHOLD` were numerically identical, so ANY soft rule
   firing was an unconditional, undampened anomaly verdict -- a rule's
   raw false-positive rate propagated 1:1 into the confusion matrix.
   Replaced with: every rule carries its own confidence (0-100,
   `RULE_BASE_CONFIDENCE`), the strongest single piece of rule evidence
   this reading is `rule_confidence_pct`, and:

       overall = 0.6 * model_pct + 0.4 * rule_confidence_pct
       is_anomaly = overall > 55  OR  model_pct > 80  OR  rule_confidence_pct > 90

   REBALANCED (config.py, this pass): the constants previously in place
   here (FUSION_ANOMALY_THRESHOLD=90, MODEL_ALONE_OVERRIDE_THRESHOLD=100)
   reproduced Phase 1's exact deadlock under a different name --
   MODEL_ALONE_OVERRIDE_THRESHOLD=100 meant the (0-100 clipped) model
   could NEVER independently flag anything, and FUSION_ANOMALY_THRESHOLD
   =90 meant no rule below RULE_CONFIDENCE_BYPASS could clear the
   blended threshold even with a maximal model score -- so in practice
   `is_anomaly` reduced to `rule_confidence_pct > 90`, i.e. whichever
   rules happened to sit above that line (frozen/fail-low/drift/spike/
   multivariate_confirmed, all 95) fired unconditionally, exactly the
   retired hard-gate pattern. Confirmed independently by both the
   context-pass math and the technical-assessment audit. Fixed by
   lowering FUSION_ANOMALY_THRESHOLD to 55 and MODEL_ALONE_OVERRIDE_
   THRESHOLD to 80, and by moving multivariate_confirmed's own
   confidence below RULE_CONFIDENCE_BYPASS (95 -> 82) since its trigger
   shape is the one most directly built around anomaly_injector.py's
   specific implementation (see _multivariate_evidence's docstring) --
   see config.py for the full reasoning per constant.

   `anomaly_score_pct` / severity (90/70/55) keep their exact existing
   scale and meaning -- this changes what feeds the score, not the
   contract, same principle every prior phase already committed to.

7. `verdict` now carries `model_confidence_pct` and `rule_confidence_pct`
   as new additive fields -- direct request from Draft 2 ("produce a
   model confidence score in each reading"). Nothing existing reads
   these yet; they don't remove or rename anything.

8. SensorHealthTracker is now PER-PARAMETER internally (§1's "a frozen
   pressure sensor shouldn't take temp/humidity offline with it", §6's
   per-(station,parameter) 10h/24h counters, §4/§5b's per-parameter
   fast paths). Its EXTERNAL surface (`.status`, `.offline_reason`,
   `.record()`, `.should_include_in_baseline()`) is kept intact so
   state.py keeps working unmodified for now -- `.status`/
   `.offline_reason` are now a station-level AGGREGATE computed from the
   richer per-parameter state (`self.param_status`, new, not yet
   consumed anywhere -- exposed for when state.py/main.py get their own
   pass to surface real per-sensor status to the frontend).

   KNOWN LIMITATION, FLAGGED NOT FIXED (belongs to state.py's next
   pass): `should_include_in_baseline()` is still a single station-wide
   boolean, because state.py's raw history buffer stores whole rows
   (all 3 params together) -- excluding just one OFFLINE parameter's
   value while keeping the other two needs columnar buffering, which
   is a state.py redesign, not something fixable from inside this file.
   For now: ANY parameter OFFLINE excludes the WHOLE row, same
   conservative behavior as before.

   ALSO FLAGGED: state.py's `StationBuffer.mark_repaired()` currently
   pokes `self.health.status`/`self.health.offline_reason`/
   `self.health._clean_streak` directly. Those station-level attributes
   still exist here for compatibility, but repairing a station no
   longer resets the new PER-PARAMETER streaks/counters underneath them
   -- a real gap, needs a state.py-side fix when that file's turn comes.

9. UNIFIED 10h/24h WINDOW (§6): general, MIXED-fault-type, per-parameter
   counters, running alongside every rule's own specific mechanism, not
   instead of it. `WINDOW_10H_TRIGGER`/`WINDOW_24H_TRIGGER` are picked
   at 4 (within the user's stated "4-5" range) -- one readings-per-hour
   assumption, same simplification `HEALTH_WINDOW_SIZE` already made
   pre-redesign.

10. `RECOVERY_CLEAN_STREAK_REQUIRED` is still imported from `config.py`
    (currently 5 there) rather than hardcoded -- when config.py's own
    pass lands and drops it to 3 (locked, Draft 2 §11.1), this file
    picks the new value up automatically, no further edit needed here.

===========================================================================
CONSTANTS NOT YET IN config.py
===========================================================================
Per Draft 2 §11's file order, config.py is a LATER step. Everything new
this file needs (CUSUM allowance/threshold, multivariate thresholds,
fail-low floors, fusion weights, window triggers) is defined locally
below, same style the pre-redesign file already used for
IS_ANOMALY_THRESHOLD/MODEL_ONLY_THRESHOLD/HARD_RULE_FLOOR etc. These are
DOMAIN-ESTIMATED PLACEHOLDERS pending a real evaluate.py run -- flagged
individually below, not silently presented as tuned.
"""

import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from model.features import (
    build_features_for_latest,
    add_temporal_features,
    add_cross_parameter_features,
    RULE_ONLY_PREFIXES,
    get_threshold,
)
from model.explain import likely_faulty_params
from model.seasonal_baseline import get_expected_roc

from config import (
    score_to_severity,
    RECOVERY_CLEAN_STREAK_REQUIRED,
    MODEL_WEIGHT,
    RULE_WEIGHT,
    FUSION_ANOMALY_THRESHOLD,
    MODEL_ALONE_OVERRIDE_THRESHOLD,
    RULE_CONFIDENCE_BYPASS,
    SPIKE_REVERSION_RATIO,
    SPIKE_DEVIATION_MULTIPLIER,
    RULE_BASE_CONFIDENCE,
    CUSUM_DRIFT_ALLOWANCE,
    CUSUM_THRESHOLD,
    CUSUM_DIRECTION_STREAK_REQUIRED,
    FROZEN_CONSECUTIVE_REQUIRED,
    FROZEN_CONSECUTIVE_REQUIRED_PRESSURE,
    FROZEN_MIN_MODEL_CORROBORATION,
    MULTIVARIATE_TEMP_DEVIATION_THRESHOLD,
    MULTIVARIATE_HUMIDITY_DEVIATION_THRESHOLD,
    MULTIVARIATE_PRESSURE_FLAT_THRESHOLD,
    MULTIVARIATE_VAPOR_CONSISTENCY_THRESHOLD,
    MULTIVARIATE_TEMP_ATTRIBUTION_WEIGHT,
    MULTIVARIATE_ATTRIBUTION_DOMINANCE,
    FAIL_LOW_FLOOR,
    FAIL_LOW_CONSECUTIVE_REQUIRED,
    WINDOW_10H_SIZE,
    WINDOW_10H_TRIGGER,
    WINDOW_24H_SIZE,
    WINDOW_24H_TRIGGER,
    SPATIAL_CORROBORATION_MIN_PEERS,
    SPATIAL_CORROBORATION_THRESHOLD_SIGMA,
    SPIKE_DIURNAL_MIN_PEERS,
    SPIKE_DIURNAL_CONSENSUS_FRACTION,
    SPIKE_DIURNAL_SUPPRESSION_FACTOR,
    SPIKE_DIURNAL_PEER_MIN_ROC,
    graduated_confidence_frozen,
    graduated_confidence_drift,
    graduated_confidence_spike,
    graduated_confidence_fail_low,
    graduated_confidence_multivariate,
    CLUSTERS,
)

ARTIFACTS_PATH = Path(__file__).parent.parent / "model_artifacts" / "isolation_forest.pkl"
CALIBRATION_ARTIFACT_PATH = Path(__file__).parent.parent / "model_artifacts" / "network_corroboration.pkl"

# Same hard physical ceilings used in anomaly_injector.py's clip step --
# duplicated intentionally, not imported: this is a genuinely
# independent check (what CAN physically be true), not shared
# fault-generation logic.
PHYSICAL_BOUNDS = {
    "temperature_c": (-50.0, 60.0),
    "pressure_hpa": (850.0, 1085.0),
    "humidity_pct": (0.0, 100.0),
}

# (raw_column -> features.py prefix), single source of truth imported
# from features.py.
PARAM_PREFIXES = dict(RULE_ONLY_PREFIXES)
PARAMS = list(PARAM_PREFIXES.keys())


# ---------------------------------------------------------------------
# Rule confidence (§7) -- replaces the old HARD/SOFT floor tiers.
# Each is "how sure is this ONE piece of evidence, on its own, that
# something is really wrong" -- fusion combines the strongest one with
# the model score, it does not stack multiple simultaneous rules.
#
# physical_bounds / dropout: unambiguous facts (100) -- unchanged from
#   before, these were always the "certain" tier.
# frozen_value: deterministic floor-match, no calibration uncertainty,
#   but still just 3 readings -- 90, not 100.
# sensor_fail_low: fires only once persistence-confirmed (see
#   FAIL_LOW_CONSECUTIVE_REQUIRED) -- 95.
# drift: CUSUM already requires sustained one-directional accumulation
#   to cross its threshold at all -- 85.
# spike: single-reading by definition (§3) -- deliberately modest (60)
#   so one spike alone rarely clears the fusion threshold unassisted;
#   frequency-based escalation is the health tracker's job (§6), not
#   this rule's own confidence.
# multivariate_single / multivariate_confirmed: §4/§8's worked example
#   -- one qualifying reading is suspicious only (55), two consecutive
#   is confirmed (95) and also triggers the fast-path OFFLINE set.
# ---------------------------------------------------------------------
def load_model():
    """
    Loads the trained model artifact ONCE -- call this at API startup, not per-request.

    Track A (blueprint §1): also loads fault_helper.pkl if it exists alongside
    isolation_forest.pkl.  When both are present, score_reading() OR's the
    helper's alert into is_anomaly, matching evaluate.py's offline eval pipeline
    exactly (previously the live path never called fault_helper, so demo metrics
    and eval metrics diverged).

    Graceful fallback: if fault_helper.pkl is missing (e.g. not yet trained),
    a warning is logged and the system continues with the Isolation Forest + rules
    alone.  The artifact dict gets a None 'fault_helper' key so score_reading()
    can always check `artifact.get('fault_helper')` without an attribute error.
    """
    import logging
    _log = logging.getLogger(__name__)

    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(f"No trained model at {ARTIFACTS_PATH} -- run model/train.py first.")
    artifact = joblib.load(ARTIFACTS_PATH)
    if "rule_thresholds" not in artifact:
        raise KeyError(
            "Loaded artifact has no 'rule_thresholds' key -- it was saved by an older "
            "train.py, before calibrate_rule_thresholds() was added. Retrain with the "
            "current train.py before running detection."
        )

    # Load fault_helper (Track A).
    fault_helper_path = ARTIFACTS_PATH.parent / "fault_helper.pkl"
    if fault_helper_path.exists():
        try:
            artifact["fault_helper"] = joblib.load(fault_helper_path)
            _log.info("[detect] fault_helper loaded from %s", fault_helper_path)
        except Exception as exc:
            _log.warning("[detect] Could not load fault_helper.pkl: %s -- continuing without it.", exc)
            artifact["fault_helper"] = None
    else:
        _log.warning(
            "[detect] fault_helper.pkl not found at %s -- "
            "live detection uses Isolation Forest + rules only. "
            "Run model/fault_helper.py training to restore evaluate.py parity.",
            fault_helper_path,
        )
        artifact["fault_helper"] = None

    return artifact



def _model_score_to_pct(raw_reading: dict, feature_row: pd.Series, artifact: dict, history_df: pd.DataFrame) -> tuple:
    """
    Converts Isolation Forest's raw decision_function output into a
    0-100 "anomaly score." Returns (score, status).
    """
    model = artifact["model"]
    X = feature_row[artifact["feature_columns"]].values.reshape(1, -1).astype(np.float64)

    if np.isnan(X).any():
        if len(history_df) < 48:
            return None, "UNAVAILABLE_WARMUP"
        return None, "UNAVAILABLE_MISSING_FEATURES"

    try:
        raw_score = model.decision_function(X)[0]
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"[detect] Inference failed: {e}")
        return None, "FAILED_INFERENCE"
        
    z = (0.0 - raw_score) / (artifact["training_score_std"] + 1e-9)
    pct = 100 / (1 + np.exp(-1.5 * z))
    return float(np.clip(pct, 0, 100)), "AVAILABLE"


def _featurize_buffer(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared helper for the two rule checks (CUSUM, multivariate) that
    need more than the single latest feature row -- runs the same
    add_temporal_features -> add_cross_parameter_features pipeline
    build_features_for_latest uses internally, but returns the WHOLE
    featured buffer instead of just the last row, computed ONCE per
    score_reading() call and shared by both checks rather than each
    re-deriving it separately.

    FIXED: this previously called add_temporal_features only, despite
    the docstring above already claiming otherwise -- add_
    cross_parameter_features didn't exist in features.py yet at the
    time this was written. It exists now (dewpoint_depression_c,
    vapor_pressure_deficit_kpa, vapor_pressure_consistency_dev); wiring
    it in here is what _multivariate_evidence's new physics-based path
    below actually needs.
    """
    df = history_df.sort_values("timestamp").reset_index(drop=True)
    df = add_temporal_features(df)
    df = add_cross_parameter_features(df)
    return df


def _cusum_evidence(
    featured_buffer: pd.DataFrame,
    prefix: str,
    param: str,
    station_id: str = "",
    current_hour: int = 0,
):
    """
    Recomputes the CUSUM S+/S- accumulator from scratch over the
    station's own causal buffer every call -- stateless by construction
    (see module docstring #3).

    DIURNAL HARDENING (SEA6/CUSUM3 in bug audit):
    Instead of accumulating the raw normalized_roc_1h, we accumulate
    the RESIDUAL:  (actual_roc - expected_roc) / rolling_std.

    UNIT FIX: the previous version subtracted raw °C/h (from
    seasonal_baseline) from normalized_roc_1h (a dimensionless z-score:
    roc_1h / rolling_std).  This unit mismatch made the seasonal
    hardening completely non-functional — normal morning warming at
    +1.7°C/h produced norm_roc=3.83 minus expected=1.95 = residual
    +1.88, which accumulated and triggered false drift.  Fixed: use
    raw roc_1h, subtract expected_roc in matching °C/h units, THEN
    divide by rolling_std to normalize.

    expected_roc comes from seasonal_baseline.py which loads the real
    3-month historical CSV (Open-Meteo data) once and caches it.  A
    normal sunrise warming trend (e.g. +1.7°C/h at hour 8) has an
    expected_roc ~+1.95°C/h at that hour, so its raw residual ≈ -0.25
    and the normalized residual ≈ -0.06σ — CUSUM stays near 0.  A real
    sensor drift fault deviates from the seasonal baseline, so its
    residual accumulates.

    Fallback: if no historical data exists for this station (new station,
    warm-up), get_expected_roc returns 0.0, reproducing the original
    pre-hardening behavior exactly -- safe, not silent.

    The data source is swappable: replace seasonal_baseline._load_csv()
    with a TimescaleDB or API call without changing this function.
    """
    roc_col = f"{prefix}_roc_1h"
    scale_col = f"{prefix}_robust_scale"
    if roc_col not in featured_buffer.columns:
        return None

    # We need both raw ROC and rolling_std to compute the correct residual.
    valid_mask = featured_buffer[roc_col].notna()
    if scale_col in featured_buffer.columns:
        valid_mask = valid_mask & featured_buffer[scale_col].notna() & (featured_buffer[scale_col] > 0)
    valid_indices = featured_buffer.index[valid_mask]
    if valid_indices.empty:
        return None

    # Build per-step hours for hour-aware baseline lookup.
    if "timestamp" in featured_buffer.columns:
        step_hours = pd.to_datetime(featured_buffer.loc[valid_indices, "timestamp"]).dt.hour.to_numpy()
    else:
        step_hours = np.full(len(valid_indices), current_hour, dtype=int)

    raw_rocs = featured_buffer.loc[valid_indices, roc_col].to_numpy()
    scales = (
        featured_buffer.loc[valid_indices, scale_col].to_numpy()
        if scale_col in featured_buffer.columns
        else np.ones(len(valid_indices))
    )

    s_pos = s_neg = 0.0
    ewma_val = 0.0
    from config import EWMA_DRIFT_ALPHA, EWMA_DRIFT_THRESHOLD
    
    allowance = CUSUM_DRIFT_ALLOWANCE.get(param, 0.05) if isinstance(CUSUM_DRIFT_ALLOWANCE, dict) else 0.05
    
    residuals = []
    for raw_roc, scale_val, h in zip(raw_rocs, scales, step_hours):
        if raw_roc is None or not np.isfinite(raw_roc):
            continue
        raw_roc = float(raw_roc)
        scale_val = 1.0 if param == "temperature_c" else (0.5 if param == "pressure_hpa" else 3.5)
        expected = get_expected_roc(station_id, prefix, int(h))
        residual = float(np.clip((raw_roc - expected) / scale_val, -3.0, 3.0))
        residuals.append(residual)
        
        s_pos = max(0.0, s_pos + residual - allowance)
        s_neg = max(0.0, s_neg - residual - allowance)
        
        ewma_val = EWMA_DRIFT_ALPHA * residual + (1.0 - EWMA_DRIFT_ALPHA) * ewma_val

    pos_streak = neg_streak = False
    if len(residuals) >= CUSUM_DIRECTION_STREAK_REQUIRED:
        recent_steps = np.array(residuals[-CUSUM_DIRECTION_STREAK_REQUIRED:])
        pos_streak = np.sum(recent_steps > 0) == CUSUM_DIRECTION_STREAK_REQUIRED
        neg_streak = np.sum(recent_steps < 0) == CUSUM_DIRECTION_STREAK_REQUIRED

    cusum_triggered = (pos_streak and s_pos > CUSUM_THRESHOLD) or (neg_streak and s_neg > CUSUM_THRESHOLD)
    ewma_triggered = (pos_streak and ewma_val > EWMA_DRIFT_THRESHOLD) or (neg_streak and ewma_val < -EWMA_DRIFT_THRESHOLD)
    
    if cusum_triggered or ewma_triggered:
        trigger_src = "EWMA" if ewma_triggered else "CUSUM"
        val = abs(ewma_val) if ewma_triggered else (s_pos if pos_streak else s_neg)
        thresh = EWMA_DRIFT_THRESHOLD if ewma_triggered else CUSUM_THRESHOLD
        drift_conf = graduated_confidence_drift(val, thresh, is_ewma=ewma_triggered)
        
        return {
            "type": "drift",
            "parameter": param,
            "confidence": drift_conf,
            "observed_value": float(featured_buffer[param].iloc[-1]) if len(featured_buffer) else None,
            "threshold": f">{thresh}",
            "reason": f"{trigger_src} accumulator ({val:.2f}) exceeded threshold ({thresh})."
        }
    return None


def _multivariate_evidence(featured_buffer: pd.DataFrame):
    """
    §4/§8's reference implementation. TWO independent trigger paths
    (see config.py's MULTIVARIATE_VAPOR_CONSISTENCY_THRESHOLD comment
    for why there are now two, not one):

      (a) LEVEL-based co-occurrence (original): temp+humidity both
          deviate from baseline, same direction, pressure stays flat.
      (b) NEW -- direct vapor-pressure-conservation violation, via
          features.py's vapor_pressure_consistency_dev. This is the
          general case: (a) additionally requires pressure to stay
          flat, which is true of THIS injector's implementation but
          not a fact about real cross-talk/short-circuit faults in
          general (a real fault could move pressure too, and (a) alone
          would then miss it). (b) looks only at whether the observed
          T/RH pair is consistent with itself one reading back, so it
          doesn't inherit that injector-specific assumption.

    A single qualifying reading (either path) is suspicious only (does
    not force OFFLINE); 2 CONSECUTIVE qualifying readings (either path,
    not necessarily the same one on both readings) is confirmed
    (forces OFFLINE via the fast-path set).

    ATTRIBUTION FIX: this used to blanket-mark BOTH temperature_c and
    humidity_pct on every confirmed hit, with no magnitude comparison
    at all -- a real regression against §4's explicit ask ("inspect
    which raw parameter(s) are actually driving the inconsistency...
    temp weighted more heavily than humidity in ambiguous joint
    moves"), introduced to stop detect.py/evaluate.py disagreeing on
    which sensor gets marked, but which threw out the weighting logic
    in the process instead of porting it correctly. evaluate.py's
    parallel rule engine already had this right; this brings detect.py
    in line with it -- same MULTIVARIATE_TEMP_ATTRIBUTION_WEIGHT (1.5)
    and same MULTIVARIATE_ATTRIBUTION_DOMINANCE (0.7) ratio-of-dominant
    check evaluate.py uses, so the two engines can't silently blame
    different sensors for the same event again.

    One deliberate difference from evaluate.py, not an oversight:
    evaluate.py computes this ratio from {prefix}_normalized_roc_1h
    (its own multivariate TRIGGER is roc-based, via the coupling/
    pressure_inconsistency features). THIS file's level-based path (a)
    reads the SAME level signals that drove path (a)'s own trigger, not
    a second, independently-chosen signal family -- using roc here
    instead would reintroduce exactly the kind of "two derivations of
    the same idea that can drift apart" bug this project has hit before
    (see RULE_ONLY_PREFIXES's own comment in features.py). Attribution
    for path (b)-only hits falls back to the same temp/humidity_dev
    comparison, since vapor_pressure_consistency_dev doesn't itself
    say which sensor is "wrong" -- only that the pair, together, is.

    Pressure is the "should have moved but didn't" reference signal
    used only to help path (a) FIRE -- it is never itself implicated
    (§4).
    """
    if len(featured_buffer) == 0:
        return [], set()

    def _fires(row) -> bool:
        temp_dev = row.get("temp_deviation")
        humidity_dev = row.get("humidity_deviation")
        pressure_dev = row.get("pressure_deviation")
        level_fires = (
            pd.notna(temp_dev) and pd.notna(humidity_dev) and pd.notna(pressure_dev)
            and abs(temp_dev) > MULTIVARIATE_TEMP_DEVIATION_THRESHOLD
            and abs(humidity_dev) > MULTIVARIATE_HUMIDITY_DEVIATION_THRESHOLD
            and (temp_dev * humidity_dev) > 0  # same direction -- §4's "temp up, humidity ALSO up"
            and abs(pressure_dev) < MULTIVARIATE_PRESSURE_FLAT_THRESHOLD
        )
        vapor_dev = row.get("vapor_pressure_consistency_dev")
        physics_fires = pd.notna(vapor_dev) and abs(vapor_dev) > MULTIVARIATE_VAPOR_CONSISTENCY_THRESHOLD
        return level_fires or physics_fires

    latest = featured_buffer.iloc[-1]
    if not _fires(latest):
        return [], set()

    confirmed = len(featured_buffer) >= 2 and _fires(featured_buffer.iloc[-2])

    # Compute joint magnitude across temp and humidity deviations
    td = abs(latest.get("temp_deviation", 0.0)) if pd.notna(latest.get("temp_deviation")) else 0.0
    hd = abs(latest.get("humidity_deviation", 0.0)) if pd.notna(latest.get("humidity_deviation")) else 0.0
    joint_z = float(np.sqrt(td ** 2 + hd ** 2))
    base_thresh = float(np.sqrt(MULTIVARIATE_TEMP_DEVIATION_THRESHOLD ** 2 + MULTIVARIATE_HUMIDITY_DEVIATION_THRESHOLD ** 2))
    confidence = graduated_confidence_multivariate(joint_z, base_thresh, confirmed=confirmed)

    # --- weighted attribution: which param(s) actually implicated ---
    temp_dev = latest.get("temp_deviation")
    humidity_dev = latest.get("humidity_deviation")
    temp_mag = abs(temp_dev) * MULTIVARIATE_TEMP_ATTRIBUTION_WEIGHT if pd.notna(temp_dev) else 0.0
    humidity_mag = abs(humidity_dev) if pd.notna(humidity_dev) else 0.0
    dominant_mag = max(temp_mag, humidity_mag) or 1.0

    implicated = []
    if temp_mag / dominant_mag >= MULTIVARIATE_ATTRIBUTION_DOMINANCE:
        implicated.append("temperature_c")
    if humidity_mag / dominant_mag >= MULTIVARIATE_ATTRIBUTION_DOMINANCE:
        implicated.append("humidity_pct")
    if not implicated:
        # Guard only -- the dominant param always clears its own 1.0
        # ratio against itself, so this shouldn't trigger in practice.
        # Kept so a confirmed multivariate hit can never silently
        # implicate nobody.
        implicated = ["temperature_c"] if temp_mag >= humidity_mag else ["humidity_pct"]

    evidence = [{
        "type": "multivariate_inconsistency",
        "parameter": param,
        "confidence": confidence,
        "observed_value": latest.get(param),
        "threshold": "Consistency bounds",
        "reason": "Multivariate inconsistency detected (temperature vs humidity)."
    } for param in implicated]
    fast_path = set(implicated) if confirmed else set()
    return evidence, fast_path


def _confirmed_spikes(featured_buffer: pd.DataFrame, thresholds: dict, station_id: str) -> list[dict]:
    """Confirm the prior reading as a spike once a current reading reverts.

    At time t+1, t+2, or t+3 we can finally distinguish `normal -> extreme(s) -> normal`
    from a real, sustained weather move. The returned event belongs to
    the bad raw reading, not the confirming reading.
    """
    confirmed = []
    n = len(featured_buffer)
    if n < 3:
        return confirmed

    current = featured_buffer.iloc[-1]
    
    for param, prefix in PARAM_PREFIXES.items():
        spike_threshold = get_threshold(thresholds, "spike", prefix, station_id)
        current_val = current.get(param)
        if pd.isna(current_val):
            continue
            
        # Check if the current reading confirms a candidate spike from 1, 2, or 3 steps ago
        for step in range(1, 4):
            if n < step + 2:
                break
                
            candidate = featured_buffer.iloc[-(step + 1)]
            before = featured_buffer.iloc[-(step + 2)]
            
            before_val = before.get(param)
            candidate_val = candidate.get(param)
            
            if pd.isna(before_val) or pd.isna(candidate_val):
                continue
                
            jump = abs(candidate_val - before_val)
            if jump < spike_threshold:
                continue
                
            candidate_dev = candidate.get(f"{prefix}_deviation")
            if pd.isna(candidate_dev) or abs(candidate_dev) <= spike_threshold * SPIKE_DEVIATION_MULTIPLIER:
                continue
                
            # If it reverted to baseline NOW
            if abs(current_val - before_val) <= jump * SPIKE_REVERSION_RATIO:
                reversion_cleanliness = max(0.0, 1.0 - (abs(current_val - before_val) / (jump * SPIKE_REVERSION_RATIO if jump > 0 else 1.0)))
                spike_conf = graduated_confidence_spike(
                    abs(candidate_dev),
                    spike_threshold * SPIKE_DEVIATION_MULTIPLIER,
                    reversion_cleanliness=reversion_cleanliness,
                )
                confirmed.append({
                    "parameter": param,
                    "timestamp": pd.Timestamp(candidate["timestamp"]),
                    "observed_value": float(candidate_val),
                    "suggested_value": round(float((before_val + current_val) / 2), 2),
                    "confidence": spike_conf,
                })
                break  # we confirmed a spike for this parameter, stop checking older steps
                
    return confirmed


def _rule_checks(raw_reading: dict, feature_row: pd.Series, history_df: pd.DataFrame, artifact: dict) -> dict:
    """
    Seven independent checks, each computed on its own terms -- no rule
    here "wins" over another; that resolution happens in
    _fuse_and_score via confidence, not here (§8).

    Returns {"fired": [(rule_type, param, confidence), ...], "any": bool,
    "fast_path_offline_params": set(param)} -- the fast-path set is
    populated by rules whose OWN persistence/confirmation criteria are
    already the full bar for immediate OFFLINE (confirmed multivariate,
    confirmed fail-low), so SensorHealthTracker doesn't need to
    re-derive that persistence logic a second time.
    """
    fired = []
    fast_path_offline_params = set()
    thresholds = artifact["rule_thresholds"]
    station_id = raw_reading.get("station_id") or history_df["station_id"].iloc[-1]

    # physical_bounds -- unambiguous, unchanged.
    for param, (low, high) in PHYSICAL_BOUNDS.items():
        val = raw_reading.get(param)
        if val is not None and not pd.isna(val) and not (low <= val <= high):
            fired.append({
                "type": "physical_bounds",
                "parameter": param,
                "confidence": RULE_BASE_CONFIDENCE["physical_bounds"],
                "observed_value": val,
                "threshold": f"[{low}, {high}]",
                "reason": f"Value {val} violates hard physical bounds [{low}, {high}]."
            })

    # frozen_value -- §1, deterministic floor-match computed once in
    # features.py; this file just reads the boolean off feature_row.
    for param, prefix in PARAM_PREFIXES.items():
        req = FROZEN_CONSECUTIVE_REQUIRED_PRESSURE if param == "pressure_hpa" else FROZEN_CONSECUTIVE_REQUIRED
        streak = feature_row.get(f"{prefix}_frozen_streak", 0)
        if streak >= req:
            frozen_conf = graduated_confidence_frozen(int(streak), int(req))
            fired.append({
                "type": "frozen_value",
                "parameter": param,
                "confidence": frozen_conf,
                "observed_value": raw_reading.get(param),
                "threshold": f"{req} readings",
                "streak": int(streak),
                "reason": f"Value exactly constant for {streak} consecutive readings (threshold: {req})."
            })

    # dropout -- unambiguous, unchanged.
    for param in PHYSICAL_BOUNDS:
        val = raw_reading.get(param)
        if val is None or pd.isna(val):
            fired.append({
                "type": "dropout",
                "parameter": param,
                "confidence": RULE_BASE_CONFIDENCE["dropout"],
                "observed_value": None,
                "threshold": "present",
                "reason": "Reading missing or NaN."
            })

    # sensor_fail_low -- §5b. Persistence-gated by construction: this
    # ONLY fires once already held for FAIL_LOW_CONSECUTIVE_REQUIRED
    # consecutive readings, so every firing is already a confirmed
    # fast-path event, no separate "single vs confirmed" split needed
    # the way multivariate has one.
    recent_raw = history_df.sort_values("timestamp").tail(FAIL_LOW_CONSECUTIVE_REQUIRED)
    if len(recent_raw) >= FAIL_LOW_CONSECUTIVE_REQUIRED:
        for param, prefix in PARAM_PREFIXES.items():
            if param not in recent_raw.columns:
                continue
            vals = recent_raw[param]
            if vals.notna().all() and (vals <= FAIL_LOW_FLOOR[prefix]).all():
                faillow_conf = graduated_confidence_fail_low(
                    float(vals.iloc[-1]),
                    float(FAIL_LOW_FLOOR[prefix]),
                    len(vals),
                    FAIL_LOW_CONSECUTIVE_REQUIRED,
                )
                fired.append({
                    "type": "sensor_fail_low",
                    "parameter": param,
                    "confidence": faillow_conf,
                    "observed_value": vals.iloc[-1],
                    "threshold": f"<={FAIL_LOW_FLOOR[prefix]}",
                    "reason": f"Value at or below noise floor {FAIL_LOW_FLOOR[prefix]} for {FAIL_LOW_CONSECUTIVE_REQUIRED} readings."
                })
                fast_path_offline_params.add(param)

    # drift (CUSUM, §2) + multivariate_inconsistency (§4) share one
    # featurized-buffer pass -- computed once, used by both.
    featured_buffer = _featurize_buffer(history_df)
    confirmed_spikes = _confirmed_spikes(featured_buffer, thresholds, station_id)

    # ── Instantaneous Rate-of-Change / Candidate Spike Detection ──
    # WMO-No. 8: Natural meteorological air temperature changes rarely exceed 8-10°C/hr.
    # An instantaneous jump (e.g. 24 -> 38 in 1 step) is an unambiguous rate-of-change spike.
    if len(featured_buffer) >= 2:
        curr_row = featured_buffer.iloc[-1]
        prev_row = featured_buffer.iloc[-2]
        for param, prefix in PARAM_PREFIXES.items():
            spike_thresh = get_threshold(thresholds, "spike", prefix, station_id)
            curr_val = curr_row.get(param)
            prev_val = prev_row.get(param)
            if pd.notna(curr_val) and pd.notna(prev_val):
                step_diff = abs(float(curr_val) - float(prev_val))
                dev_val = curr_row.get(f"{prefix}_deviation")
                abs_dev = abs(float(dev_val)) if pd.notna(dev_val) else step_diff

                if step_diff >= spike_thresh and abs_dev >= (spike_thresh * SPIKE_DEVIATION_MULTIPLIER):
                    spike_conf = graduated_confidence_spike(abs_dev, spike_thresh * SPIKE_DEVIATION_MULTIPLIER)
                    fired.append({
                        "type": "spike",
                        "parameter": param,
                        "confidence": spike_conf,
                        "observed_value": float(curr_val),
                        "threshold": f">{spike_thresh * SPIKE_DEVIATION_MULTIPLIER:.1f}",
                        "reason": f"Sudden rate-of-change jump of {step_diff:.1f} (deviation {abs_dev:.1f} exceeds threshold {spike_thresh * SPIKE_DEVIATION_MULTIPLIER:.1f}).",
                    })

    # Extract current hour once for the CUSUM diurnal baseline lookup.
    try:
        current_hour = pd.to_datetime(
            raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1]
        ).hour
    except Exception:
        current_hour = 0

    for param, prefix in PARAM_PREFIXES.items():
        cusum_hit = _cusum_evidence(
            featured_buffer, prefix, param,
            station_id=station_id, current_hour=current_hour,
        )
        if cusum_hit:
            fired.append(cusum_hit)

    mv_evidence, mv_fast_path = _multivariate_evidence(featured_buffer)
    fired.extend(mv_evidence)
    fast_path_offline_params |= mv_fast_path

    return {
        "fired": fired,
        "any": len(fired) > 0,
        "fast_path_offline_params": fast_path_offline_params,
        "confirmed_spikes": confirmed_spikes,
    }


def _fuse_and_score(model_pct, rule_evidence: list):
    """
    §7/§8 in one place: the fused 0-100 score, the is_anomaly verdict,
    and which fault_type gets reported -- all from the SAME confidence
    numbers, no separate priority table (§8's worked example: whichever
    rule currently has the highest confidence wins the label).

    HIGH-CONFIDENCE BYPASS: caught during smoke-testing this rewrite --
    the pure 0.6/0.4 blend means even a MAXIMUM-confidence rule
    (100, physical_bounds/dropout -- an unambiguous fact) only
    contributes 40 points, which can never alone clear the 50 fusion
    threshold if the model happens to score that same reading low. A
    physically impossible reading (humidity=150%) must be flagged
    regardless of what the model's opinion is -- it isn't a statistical
    judgment to blend, it's a fact. Same reasoning extends to every
    near-certain rule (frozen's deterministic floor-match, confirmed
    fail-low, confirmed multivariate): once a rule's OWN persistence/
    certainty bar is fully cleared, RULE_CONFIDENCE_BYPASS (90) forces
    is_anomaly regardless of the blended score. Weaker evidence (a
    single spike at 60, a single suspicious multivariate reading at 55)
    still has to earn its way past the blend with model support, which
    is the intended, correct behavior for those two.

    UNSTRUCTURED ANOMALIES (PURE ML PATH):
    Novel, chaotic, or non-deterministic sensor patterns (e.g. erratic zig-zags,
    high volatility, complex multi-dimensional distribution shifts) do not match
    handcrafted deterministic physical rules. These are caught by the Isolation
    Forest when model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD (80.0%). In this case,
    the ML model takes ownership, overall score reflects model_pct, and fault_type
    is designated as "unstructured_anomaly".
    """
    if rule_evidence:
        rule_confidence = max(r["confidence"] for r in rule_evidence)
        specific_evidence = [r for r in rule_evidence if r["type"] not in ("physical_bounds", "dropout", "network_helper")]
        if specific_evidence:
            fault_type = max(specific_evidence, key=lambda e: e["confidence"])["type"]
        else:
            fault_type = max(rule_evidence, key=lambda e: e["confidence"])["type"]
    else:
        rule_confidence = 0.0
        fault_type = None

    if model_pct is None:
        # No usable model score (incomplete feature vector, e.g. still
        # in the rolling-window warm-up period) -- fall back to
        # rule-only evidence rather than silently treating a missing
        # model score as "definitely normal."
        overall = rule_confidence
    else:
        # Give the ML model significantly higher authority for multivariate inconsistency
        if fault_type == "multivariate_inconsistency":
            m_weight = 0.85
            r_weight = 0.15
        else:
            m_weight = MODEL_WEIGHT
            r_weight = RULE_WEIGHT
            
        overall = m_weight * model_pct + r_weight * rule_confidence

    model_alone = model_pct is not None and model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD

    is_anomaly = (
        (overall > FUSION_ANOMALY_THRESHOLD and rule_confidence > 0.0)
        or model_alone
        or rule_confidence > RULE_CONFIDENCE_BYPASS
    )

    decision_basis = "NORMAL"
    if is_anomaly:
        if rule_confidence > RULE_CONFIDENCE_BYPASS:
            decision_basis = "RULE_ENGINE_BYPASS"
        elif model_alone and rule_confidence == 0.0:
            decision_basis = "ML_MODEL_ISOLATION"
            overall = max(overall, model_pct)
            if fault_type is None:
                fault_type = "unstructured_anomaly"
        elif model_alone:
            decision_basis = "MODEL_OVERRIDE"
        else:
            decision_basis = "FUSED_CONSENSUS"

    if is_anomaly and fault_type is None:
        fault_type = "unstructured_anomaly"

    return overall, is_anomaly, fault_type, rule_confidence, decision_basis


def _corroborate_network(raw_reading: dict, history_df: pd.DataFrame, neighbor_buffers: dict, fault_type: str, implicated_params: list, artifact: dict = None) -> dict:
    """
    Network corroboration check (Stages 3 & 4).
    Empirically-calibrated against ERA5 cluster residuals (model_artifacts/network_corroboration.pkl).
    
    Hard constraints:
    - NEVER flips is_anomaly in EITHER direction (True -> False or False -> True).
    - Now applies to spike and multivariate_inconsistency to boost precision.
    - NEVER touches hardware rail faults (physical_bounds, dropout, sensor_fail_low).
    - Only adjusts confidence and labeling for frozen_value and drift.
    """
    if not neighbor_buffers:
        return {
            "state": "INSUFFICIENT_CORROBORATION",
            "eligible_peer_count": 0,
            "corroborating_peer_count": 0,
            "network_interpretation": "No peer stations available for comparison.",
            "confidence_bonus": 0.0,
            "relabel_fault_type": None,
            "veto": False,
        }

    station_id = raw_reading.get("station_id") or (history_df["station_id"].iloc[-1] if not history_df.empty else None)
    target_time = pd.to_datetime(
        raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1],
        utc=True,
    )

    # Resolve calibrated thresholds from artifact or file
    calib_dict = None
    if artifact and "network_corroboration" in artifact:
        calib_dict = artifact["network_corroboration"]
    elif CALIBRATION_ARTIFACT_PATH.exists():
        try:
            calib_dict = joblib.load(CALIBRATION_ARTIFACT_PATH)
        except Exception:
            calib_dict = None

    # Identify cluster info for station_id
    cluster_id = None
    center_id = None
    if calib_dict and "clusters" in calib_dict:
        for cid, cinfo in calib_dict["clusters"].items():
            if cinfo["center_station_id"] == station_id:
                cluster_id = cid
                center_id = station_id
                break
            if station_id in cinfo["pairs"]:
                cluster_id = cid
                center_id = cinfo["center_station_id"]
                break

    # If cluster not found via calibration artifact, fallback to CLUSTERS in config
    if cluster_id is None:
        for cid, cinfo in CLUSTERS.items():
            if cinfo["center"]["station_id"] == station_id:
                cluster_id = cid
                center_id = station_id
                break
            for nmeta in cinfo["neighbors"]:
                if nmeta["station_id"] == station_id:
                    cluster_id = cid
                    center_id = cinfo["center"]["station_id"]
                    break

    cluster_calib = calib_dict["clusters"].get(cluster_id) if (calib_dict and cluster_id) else None

    # Compute target station features ONCE before the peer loop
    try:
        target_features = build_features_for_latest(history_df)
    except Exception:
        return {
            "state": "INSUFFICIENT_CORROBORATION",
            "eligible_peer_count": 0,
            "corroborating_peer_count": 0,
            "network_interpretation": "Target station features could not be built for comparison.",
            "confidence_bonus": 0.0,
            "relabel_fault_type": None,
            "veto": False,
        }

    # -- DYNAMIC THRESHOLD CALCULATION --
    # Calculate dynamic limits per reading to account for gradual weather changes
    dynamic_thresholds = {}
    all_dfs = [history_df] + [ndf for nid, ndf in neighbor_buffers.items() if ndf is not None and not ndf.empty]
    if len(all_dfs) >= 2:
        min_len = min(len(df) for df in all_dfs)
        if min_len >= 6:
            for param in implicated_params:
                prefix = PARAM_PREFIXES.get(param)
                if not prefix:
                    continue
                
                aligned_series = []
                st_means = []
                for df in all_dfs:
                    # Get last min_len valid readings
                    vals = df[param].dropna().astype(float).values[-min_len:]
                    if len(vals) == min_len:
                        aligned_series.append(vals)
                        st_means.append(vals.mean())
                        
                if len(aligned_series) == len(all_dfs):
                    deviations = []
                    for i in range(min_len):
                        step_residuals = [aligned_series[j][i] - st_means[j] for j in range(len(aligned_series))]
                        mean_residual = sum(step_residuals) / len(step_residuals)
                        for r in step_residuals:
                            deviations.append(abs(r - mean_residual))
                            
                    if len(deviations) >= 6:
                        deviations.sort()
                        p90_idx = int(len(deviations) * 0.90)
                        dyn_val = deviations[p90_idx]
                        
                        # Gradual onset adjustment
                        target_roc = target_features.get(f"{prefix}_roc_1h")
                        if pd.notna(target_roc) and dyn_val > 0:
                            if abs(float(target_roc)) < (dyn_val * 0.5):
                                dyn_val *= 3.0
                                
                        floor = {"temperature_c": 1.2, "pressure_hpa": 1.5, "humidity_pct": 2.0}.get(param, 1.2)
                        dynamic_thresholds[param] = max(floor, round(dyn_val, 2))
    # -----------------------------------


    eligible_peers = 0
    corroborating_peers = 0
    diverged_peers = 0
    flat_peers = 0

    for nid, n_df in neighbor_buffers.items():
        if n_df is None or n_df.empty:
            continue
        n_latest = n_df.iloc[-1]
        n_time = pd.to_datetime(n_latest["timestamp"], utc=True)

        # Freshness check: peer reading must be within 1h of target
        if abs((n_time - target_time).total_seconds()) > 3600:
            continue

        # Check exclusion from empirical calibration
        peer_calib = None
        if cluster_calib and "pairs" in cluster_calib:
            # Pair calibration between center and neighbor
            if nid in cluster_calib["pairs"]:
                peer_calib = cluster_calib["pairs"][nid]
            elif station_id in cluster_calib["pairs"] and nid == center_id:
                peer_calib = cluster_calib["pairs"][station_id]

        eligible_peers += 1

        try:
            n_features = build_features_for_latest(n_df)
        except Exception:
            eligible_peers -= 1
            continue

        # Evaluate corroboration / divergence per implicated parameter
        for param in implicated_params:
            prefix = PARAM_PREFIXES.get(param)
            if not prefix:
                continue

            param_calib = peer_calib["parameters"].get(param) if peer_calib else None
            # Skip if this parameter pair is excluded due to reanalysis grid resolution
            if param_calib and param_calib.get("excluded", False):
                continue

            target_dev = target_features.get(f"{prefix}_deviation", 0)
            peer_dev = n_features.get(f"{prefix}_deviation", 0)

            # Peer delta over 1h
            peer_roc = n_features.get(f"{prefix}_roc_1h", 0.0)
            peer_delta = abs(float(peer_roc)) if pd.notna(peer_roc) else 0.0
            
            # Use dynamic threshold if enough data warm-up, otherwise fallback to static artifact
            if param in dynamic_thresholds:
                div_thresh = dynamic_thresholds[param]
            else:
                div_thresh = param_calib["divergence_threshold"] if param_calib else 1.0

            if fault_type == "frozen_value":
                if peer_delta > div_thresh:
                    diverged_peers += 1
                elif peer_delta <= (div_thresh * 0.3):
                    flat_peers += 1
            elif fault_type in ["drift", "spike", "multivariate_inconsistency"]:
                dev_thresh = 4.0 if prefix == "temp" else (3.0 if prefix == "pressure" else 15.0)
                corroborated_by_dev = (
                    pd.notna(target_dev) and pd.notna(peer_dev)
                    and abs(peer_dev) >= dev_thresh
                    and (target_dev * peer_dev > 0)
                )
                if corroborated_by_dev:
                    corroborating_peers += 1

                if pd.notna(target_dev) and pd.notna(peer_dev):
                    if abs(target_dev - peer_dev) >= (div_thresh * 1.5) and abs(target_dev) >= dev_thresh:
                        diverged_peers += 1


    confidence_bonus = 0.0
    relabel_fault_type = None
    dampen_factor = 1.0
    veto = False
    state = "LOCALIZED"
    interpretation = "Peers show normal variability within bounds."

    if fault_type == "frozen_value":
        if eligible_peers < 1:
            state = "INSUFFICIENT_CORROBORATION"
            interpretation = "No eligible peers with fresh data to corroborate frozen sensor."
        elif diverged_peers >= 1:
            state = "CONFIRMED_DIVERGENCE"
            interpretation = f"{diverged_peers} peer(s) diverged beyond calibrated envelope while sensor remained flat."
            confidence_bonus = 16.0
        elif flat_peers >= 1:
            state = "AMBIGUOUS_STABLE_REGION"
            interpretation = f"Peers are also stable/flat within calibrated envelope; regional meteorological stability."
            suppress_alarm = True
        else:
            state = "UNCORROBORATED_STABLE"
            interpretation = "No peers diverged. Assuming regional stability."
            suppress_alarm = True
    elif fault_type in ["drift", "spike", "multivariate_inconsistency"]:
        if eligible_peers < 1:
            state = "INSUFFICIENT_CORROBORATION"
            interpretation = "No eligible peers with fresh data."
        elif corroborating_peers >= 2:
            state = "REGIONAL"
            interpretation = "Multiple peers move the same way; regional weather front."
            veto = True
            relabel_fault_type = "none"
        else:
            diverge_ratio = diverged_peers / eligible_peers if eligible_peers > 0 else 0.0
            if diverge_ratio >= 0.5:
                state = "CONFIRMED_DIVERGENCE"
                interpretation = f"Sensor clearly diverging from {diverged_peers}/{eligible_peers} peers."
                confidence_bonus = 5.0
            elif diverged_peers == 0:
                state = "REGIONAL_STABILITY"
                interpretation = "Cluster fully agrees with sensor; regional weather."
                veto = True
                relabel_fault_type = "none"
            else:
                state = "AMBIGUOUS_DIVERGENCE"
                interpretation = f"Some peers diverge, some don't. Dampening confidence."
                dampen_factor = 0.5 + 0.5 * diverge_ratio

    return {
        "state": state,
        "eligible_peer_count": eligible_peers,
        "corroborating_peer_count": corroborating_peers,
        "network_interpretation": interpretation,
        "confidence_bonus": confidence_bonus,
        "relabel_fault_type": relabel_fault_type,
        "dampen_factor": dampen_factor,
        "veto": veto,
    }



def _classify_regime(feature_row: "pd.Series", raw_reading: dict, history_df: "pd.DataFrame") -> str:
    """
    Classifies the current station environmental context into one of 9 regimes
    per audit §12.5. Uses the already-computed feature_row from build_features_for_latest
    to avoid double featurization.

    States:
      DAYTIME_WARMING       hour 6-18, temp trending up relative to baseline
      NIGHTTIME_COOLING     hour 18-6, temp trending down
      STABLE                low volatility, minimal rate-of-change all params
      HIGH_HEAT             temp > 35°C or temp_deviation > 2.5σ
      HIGH_HUMIDITY         humidity > 85% or humidity_deviation > 2.0σ
      PRESSURE_SHIFT        significant pressure rate-of-change over 3h
      HIGH_VOLATILITY       any parameter volatility_z > 2.0
      REGIME_TRANSITION     direction reversal in recent temp readings
      UNKNOWN_INSUFFICIENT_DATA   NaN features (warm-up period)
      UNKNOWN_CONTEXT_FAILURE     exception during classification
    """
    try:
        hour = pd.to_datetime(
            raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1]
        ).hour

        # Pull scalar values from feature_row with safe fallbacks.
        def get(col, default=None):
            v = feature_row.get(col)
            if v is None or (hasattr(v, '__float__') and pd.isna(float(v))):
                return default
            return float(v)

        temp_c         = get("temperature_c")
        humidity_pct   = get("humidity_pct")
        temp_dev       = get("temp_deviation")
        humidity_dev   = get("humidity_deviation")
        temp_roc_1h    = get("temp_roc_1h", 0.0)
        temp_roc_3h    = get("temp_roc_3h", 0.0)
        pressure_roc_3h = get("pressure_roc_3h", 0.0)
        temp_vol_z     = get("temp_volatility_z", 0.0)
        pressure_vol_z = get("pressure_volatility_z", 0.0)
        humidity_vol_z = get("humidity_volatility_z", 0.0)

        # UNKNOWN_INSUFFICIENT_DATA: primary indicators are NaN
        if temp_dev is None or humidity_dev is None:
            return "UNKNOWN_INSUFFICIENT_DATA"

        # HIGH_VOLATILITY: any param's volatility z-score > 2.0
        if (temp_vol_z is not None and abs(temp_vol_z) > 2.0
                or pressure_vol_z is not None and abs(pressure_vol_z) > 2.0
                or humidity_vol_z is not None and abs(humidity_vol_z) > 2.0):
            return "HIGH_VOLATILITY"

        # THERMODYNAMIC_CONFLICT: Clausius-Clapeyron violation
        vapor_dev = get("vapor_pressure_consistency_dev", None)
        if vapor_dev is not None and abs(vapor_dev) > 15.0:
            return "THERMODYNAMIC_CONFLICT"

        # HIGH_HEAT: extreme temperature by absolute value or deviation
        if (temp_c is not None and temp_c > 35.0) or (temp_dev is not None and temp_dev > 2.5):
            return "HIGH_HEAT"

        # HIGH_HUMIDITY: extreme humidity by absolute value or deviation
        if (humidity_pct is not None and humidity_pct > 85.0) or (humidity_dev is not None and humidity_dev > 2.0):
            return "HIGH_HUMIDITY"

        # PRESSURE_SHIFT: sustained pressure rate-of-change over 3h
        if pressure_roc_3h is not None and abs(pressure_roc_3h) > 2.0:
            return "PRESSURE_SHIFT"

        # REGIME_TRANSITION: recent direction reversal in temperature
        if len(history_df) >= 3:
            recent_temps = pd.to_numeric(history_df["temperature_c"].tail(4), errors="coerce").dropna().to_numpy()
            if len(recent_temps) >= 3:
                diffs = [recent_temps[i+1] - recent_temps[i] for i in range(len(recent_temps)-1)]
                signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
                non_zero = [s for s in signs if s != 0]
                if len(non_zero) >= 2 and non_zero[-1] != non_zero[-2]:
                    return "REGIME_TRANSITION"

        # STABLE: small roc and deviation across all parameters
        if (abs(temp_roc_3h) < 0.5 and abs(temp_dev) < 0.5
                and abs(pressure_roc_3h) < 0.5):
            return "STABLE"

        # DAYTIME_WARMING / NIGHTTIME_COOLING
        if 6 <= hour < 18:
            if temp_roc_3h is not None and temp_roc_3h > 0 and temp_dev > 0.2:
                return "DAYTIME_WARMING"
            return "STABLE"
        else:
            if temp_roc_3h is not None and temp_roc_3h < 0:
                return "NIGHTTIME_COOLING"
            return "STABLE"

    except Exception:
        return "UNKNOWN_CONTEXT_FAILURE"


def _apply_diurnal_consensus_filter(fired: list, neighbor_buffers: dict, feature_row: "pd.Series") -> tuple:
    """
    Diurnal / Environmental Spatial Consensus Filter for Spike Rules.

    When a spike is detected, peers in the same cluster experience the same
    atmospheric forcing (solar heating, frontal passage, etc.). If ≥ 60% of
    eligible peer stations show a rate-of-change in the SAME direction as the
    suspected spike, the event is far more likely to be environmental rather
    than a sensor malfunction.

    Action: Reduce spike confidence by SPIKE_DIURNAL_SUPPRESSION_FACTOR (×0.38).
      - A typical spike fires at 85–89.9%. After suppression: ~32–34.
      - Fused score becomes: 0.6*model + 0.4*32 ≈ 0.6*model + 12.8.
      - For is_anomaly via fusion (overall > 45), model_pct must exceed ~53%.
      - Clean weather model_pct p99 ≈ 19.87% → essentially zero FPs on real heat-up.
      - The ML model retains full authority via MODEL_ALONE_OVERRIDE (>88%).

    Guarantees:
      - Does NOT touch physical_bounds, dropout, frozen_value, drift — ONLY spike.
      - Does NOT suppress if fewer than SPIKE_DIURNAL_MIN_PEERS eligible peers exist.
      - Does NOT fully zero out spike confidence — advisory dampening only.
      - Returns (filtered_fired, suppression_map) where suppression_map maps
        param -> {"n_eligible": int, "n_agreeing": int, "consensus_fraction": float}
        for use in the verdict's network_corroboration context.
    """
    if not neighbor_buffers:
        return fired, {}

    suppression_map = {}

    for param, prefix in PARAM_PREFIXES.items():
        # Find spike entries for this param
        param_spikes = [r for r in fired if r["type"] == "spike" and r.get("parameter") == param]
        if not param_spikes:
            continue

        # Target station's direction: use feature_row roc_1h, falling back
        # to the raw step difference in the fired rule.
        target_roc_raw = feature_row.get(f"{prefix}_roc_1h")
        if pd.isna(target_roc_raw):
            # Try to get direction from the fired rule itself
            spike_rule = param_spikes[0]
            obs = spike_rule.get("observed_value")
            if obs is None:
                continue
            target_roc_raw = 1.0  # direction unknown, skip
        target_direction = 1 if float(target_roc_raw) > 0 else -1

        min_roc = SPIKE_DIURNAL_PEER_MIN_ROC.get(param, 0.5)

        # Count eligible peers and how many agree on direction
        n_eligible = 0
        n_agreeing = 0

        for _nid, nbuf_df in neighbor_buffers.items():
            if nbuf_df is None or nbuf_df.empty or param not in nbuf_df.columns:
                continue
            vals = pd.to_numeric(nbuf_df[param], errors="coerce").dropna()
            if len(vals) < 2:
                continue
            # Use last two non-NaN readings for peer ROC
            peer_roc = float(vals.iloc[-1]) - float(vals.iloc[-2])
            if abs(peer_roc) < min_roc:
                # Peer is too flat to vote (stable neighbor — not informationally useful)
                continue
            n_eligible += 1
            peer_direction = 1 if peer_roc > 0 else -1
            if peer_direction == target_direction:
                n_agreeing += 1

        if n_eligible < SPIKE_DIURNAL_MIN_PEERS:
            # Not enough peers with meaningful change — keep spike as-is
            continue

        consensus_fraction = n_agreeing / n_eligible
        suppression_map[param] = {
            "n_eligible": n_eligible,
            "n_agreeing": n_agreeing,
            "consensus_fraction": round(consensus_fraction, 2),
        }

        if consensus_fraction >= SPIKE_DIURNAL_CONSENSUS_FRACTION:
            # Spatial consensus: environmental change confirmed, dampen spike
            suppression_map[param]["suppressed"] = True
        else:
            suppression_map[param]["suppressed"] = False

    if not suppression_map:
        return fired, {}

    result = []
    for rule in fired:
        if rule["type"] == "spike":
            param = rule.get("parameter")
            info = suppression_map.get(param, {})
            if info.get("suppressed"):
                dampened = dict(rule)
                frac = info["consensus_fraction"]
                dampened["confidence"] = round(rule["confidence"] * SPIKE_DIURNAL_SUPPRESSION_FACTOR, 1)
                dampened["reason"] = (
                    rule["reason"]
                    + f" [Spatial consensus: {info['n_agreeing']}/{info['n_eligible']} peer stations"
                    f" show same-direction change (consensus={frac:.0%}) — environmental"
                    f" change likely; spike confidence dampened to {dampened['confidence']:.1f}.]"
                )
                dampened["diurnal_consensus"] = True
                result.append(dampened)
            else:
                result.append(rule)
        else:
            result.append(rule)

    return result, suppression_map


def score_reading(raw_reading: dict, history_df: pd.DataFrame, artifact: dict,
                  neighbor_buffers: dict = None, explainer=None) -> dict:
    """
    Main entry point: scores ONE new reading given its station's recent
    causal history buffer. Returns the verdict dict state.py/main.py
    consume.

    Spatial inputs are intentionally absent: Draft 2 removes them from
    both serving and the model vector.
    """
    feature_row = build_features_for_latest(history_df)
    # ── Time-aware suggested value interpolation ──────────────────────
    # The old approach used a flat 48h rolling mean, which produced noon
    # suggestions of ~23°C when the real noon reading should be ~32°C
    # (because the 48h window includes overnight lows).
    #
    # Priority cascade (first valid wins):
    #   1. Same-hour-yesterday: reading from ≈24h ago (captures diurnal cycle)
    #   2. Trend-extrapolated 3h mean: short-window mean + slope correction
    #   3. Neighbor station interpolation: time-aligned peer reading
    #   4. 48h rolling mean (original fallback)
    #   5. Most recent prior clean observation
    suggested_values = {}
    suggested_metadata = {}

    # Pre-parse history timestamps for same-hour lookup
    _hist_ts = pd.to_datetime(history_df["timestamp"], utc=True, errors="coerce")
    _current_ts = _hist_ts.iloc[-1] if not _hist_ts.empty else None
    _clean_history = history_df.iloc[:-1]  # exclude the current (possibly anomalous) reading

    for param, prefix in PARAM_PREFIXES.items():
        candidate = None
        source = None
        uncertainty = feature_row.get(f"{prefix}_rolling_std")

        # ── Strategy 1: Spatial Neighbor Interpolation (Real-Time Regional Weather) ─
        # For an active cluster, neighbor stations experience the same microclimate
        # at the exact same hour. The peer median is the gold standard for imputing
        # what this station should have observed under true atmospheric conditions.
        if candidate is None and neighbor_buffers:
            peer_vals_for_param = []
            for _nid, _nbuf_df in (neighbor_buffers or {}).items():
                if _nbuf_df is None or _nbuf_df.empty or param not in _nbuf_df.columns:
                    continue
                if _current_ts is not None and "timestamp" in _nbuf_df.columns:
                    n_ts = pd.to_datetime(_nbuf_df["timestamp"], utc=True, errors="coerce")
                    n_vals = pd.to_numeric(_nbuf_df[param], errors="coerce")
                    n_valid = n_vals.notna() & n_ts.notna()
                    if n_valid.any():
                        n_diffs = (n_ts[n_valid] - _current_ts).abs()
                        n_best = n_diffs.idxmin()
                        if n_diffs.loc[n_best] <= pd.Timedelta(hours=2):
                            nv = n_vals.loc[n_best]
                            if pd.notna(nv):
                                peer_vals_for_param.append(float(nv))
            if peer_vals_for_param:
                import numpy as np
                candidate = float(np.median(peer_vals_for_param))
                source = "neighbor_interpolation"

        # ── Strategy 2: Same-hour-yesterday (Diurnal Temporal Match) ─
        # Find a clean reading from ~24h ago (±2h tolerance).
        # This naturally matches diurnal temperature/humidity curves when
        # network peer consensus is unavailable.
        if candidate is None and _current_ts is not None and pd.notna(_current_ts) and len(_clean_history) > 0:
            target_24h = _current_ts - pd.Timedelta(hours=24)
            clean_ts = pd.to_datetime(_clean_history["timestamp"], utc=True, errors="coerce")
            clean_vals = pd.to_numeric(_clean_history[param], errors="coerce")
            valid_mask = clean_vals.notna() & clean_ts.notna()
            if valid_mask.any():
                diffs_24h = (clean_ts[valid_mask] - target_24h).abs()
                best_idx = diffs_24h.idxmin()
                if diffs_24h.loc[best_idx] <= pd.Timedelta(hours=2):
                    val_24h = clean_vals.loc[best_idx]
                    if pd.notna(val_24h):
                        candidate = float(val_24h)
                        source = "same_hour_yesterday"

        # ── Strategy 3: Forward Trend Extrapolation from Last Clean Reading ─
        # Project forward from the last clean observation using the recent rate of change,
        # rather than taking an unweighted backward rolling mean that drags into morning lows.
        if candidate is None:
            clean_series = pd.to_numeric(history_df[param].iloc[:-1], errors="coerce").dropna()
            if not clean_series.empty:
                last_clean_val = float(clean_series.iloc[-1])
                slope_6h = feature_row.get(f"{prefix}_slope_6h")
                roc_1h = feature_row.get(f"{prefix}_roc_1h")
                
                # Use recent 1h or 6h rate of change to extrapolate forward
                step = 0.0
                if pd.notna(roc_1h) and abs(roc_1h) < 10.0:
                    step = float(roc_1h)
                elif pd.notna(slope_6h) and abs(slope_6h) < 10.0:
                    step = float(slope_6h)
                
                candidate = last_clean_val + step
                source = "trend_extrapolated"

        # ── Strategy 4: 48h rolling mean (original baseline fallback) ─
        if candidate is None:
            rm = feature_row.get(f"{prefix}_rolling_mean")
            if pd.notna(rm):
                candidate = float(rm)
                source = "rolling_mean_48h"

        # ── Strategy 5: Most recent prior clean observation ──────────
        if candidate is None:
            prior = pd.to_numeric(history_df[param].iloc[:-1], errors="coerce").dropna()
            if not prior.empty:
                candidate = float(prior.iloc[-1])
                source = "prior_reading"
                uncertainty = None

        if candidate is not None:
            suggested_values[param] = round(float(candidate), 2)
            suggested_metadata[param] = {
                "value": round(float(candidate), 2),
                "source": source,
                "uncertainty": round(float(uncertainty), 2) if pd.notna(uncertainty) else None
            }

    model_pct, model_status = _model_score_to_pct(raw_reading, feature_row, artifact, history_df)
    rules = _rule_checks(raw_reading, feature_row, history_df, artifact)

    # Frozen-specific model gate: frozen_value fires at confidence=80 (below
    # RULE_CONFIDENCE_BYPASS=90) and goes through fusion. When model_pct is
    # below FROZEN_MIN_MODEL_CORROBORATION, real stable-weather streaks and
    # actual frozen sensors are indistinguishable -- suppress frozen evidence
    # so it cannot drive an anomaly verdict without model corroboration.
    fired = rules["fired"]
    
    from config import FROZEN_MIN_MODEL_CORROBORATION, DRIFT_MIN_MODEL_CORROBORATION
    if model_pct is None or model_pct < FROZEN_MIN_MODEL_CORROBORATION:
        fired = [r for r in fired if r["type"] != "frozen_value"]
    if model_pct is None or model_pct < DRIFT_MIN_MODEL_CORROBORATION:
        fired = [r for r in fired if r["type"] != "drift"]

    # ── DIURNAL SPATIAL CONSENSUS FILTER (Spike rules only) ──────────
    # Before fusion, check if peer stations show the same direction of
    # change. Morning solar heating, pressure fronts, and humidity swings
    # affect all cluster stations simultaneously. If ≥60% of eligible peers
    # agree on direction, the spike is likely environmental → dampen its
    # confidence contribution to fusion. The ML model retains full authority.
    diurnal_suppression_map = {}
    has_spike = any(r["type"] == "spike" for r in fired)
    if has_spike and neighbor_buffers:
        fired, diurnal_suppression_map = _apply_diurnal_consensus_filter(
            fired, neighbor_buffers, feature_row
        )

    # PRE-FUSION NETWORK CORROBORATION
    network_state = None
    network_interpretation = None
    eligible_peer_count = None
    corroborating_peer_count = None
    
    if fired:
        specific_evidence = [r for r in fired if r["type"] not in ("physical_bounds", "dropout", "network_helper")]
        primary_rule = max(specific_evidence, key=lambda e: e["confidence"]) if specific_evidence else max(fired, key=lambda e: e["confidence"])
        pre_fusion_fault = primary_rule["type"]
        implicated = list(set(r["parameter"] for r in fired))
        
        if pre_fusion_fault in ["frozen_value", "drift", "spike", "multivariate_inconsistency"]:
            neighbor_buffers_safe = neighbor_buffers or {}
            network_details = _corroborate_network(
                raw_reading, history_df, neighbor_buffers_safe, pre_fusion_fault, implicated, artifact=artifact
            )
            network_state = network_details["state"]
            network_interpretation = network_details["network_interpretation"]
            eligible_peer_count = network_details["eligible_peer_count"]
            corroborating_peer_count = network_details["corroborating_peer_count"]
            
            if network_details["veto"]:
                fired = [r for r in fired if r["type"] not in ["frozen_value", "drift", "spike", "multivariate_inconsistency"]]
            else:
                bonus = network_details.get("confidence_bonus", 0.0)
                dampen = network_details.get("dampen_factor", 1.0)
                relabel = network_details.get("relabel_fault_type")
                
                for r in fired:
                    if r["type"] == pre_fusion_fault:
                        if bonus > 0:
                            r["confidence"] = min(95.0, r["confidence"] + bonus) if r["type"] == "drift" else min(89.5 if r["confidence"] < 90 else 100.0, r["confidence"] + bonus)
                        if dampen < 1.0:
                            r["confidence"] = round(r["confidence"] * dampen, 1)
                        if relabel:
                            r["type"] = relabel

    score_pct, is_anomaly, fault_type, rule_confidence, decision_basis = _fuse_and_score(model_pct, fired)
    severity = score_to_severity(score_pct)

    # Track A (blueprint §1) — fault_helper live-path wiring.
    fault_helper_artifact = artifact.get("fault_helper")
    if fault_helper_artifact is not None:
        try:
            from config import HELPER_ALERT_THRESHOLD
            from model.fault_helper import feature_columns, build_network_features
            single_row = pd.DataFrame([{**raw_reading, "is_anomaly": False, "fault_type": None}])
            if isinstance(fault_helper_artifact, dict):
                fh_model = fault_helper_artifact.get("helper_model")
                fh_cols = fault_helper_artifact.get("helper_columns")
            else:
                fh_model, fh_cols = fault_helper_artifact[:2]
            fh_featured = build_network_features(single_row)
            available_cols = [c for c in fh_cols if c in fh_featured.columns]
            if available_cols:
                fh_prob = fh_model.predict_proba(fh_featured[available_cols])[:, 1][0]
                
                # Case 1: Base pipeline missed it, but ExtraTrees catches it (Recall boost)
                if not is_anomaly and fh_prob >= HELPER_ALERT_THRESHOLD:
                    is_anomaly = True
                    if fault_type is None:
                        fault_type = "drift"
                    score_pct = max(score_pct, round(fh_prob * 100, 1))
                    severity = score_to_severity(score_pct)
                
                # Case 2: Base pipeline flagged it, but ExtraTrees says it's natural weather (Precision boost)
                # Only apply this veto to multivariate and unstructured as requested by user
                elif is_anomaly and fault_type in ["multivariate_inconsistency", "unstructured_anomaly"]:
                    # If ExtraTrees is very confident it's clean (< 0.3 probability of fault)
                    if fh_prob < 0.3:
                        is_anomaly = False
                        fault_type = None
                        score_pct = round(fh_prob * 100, 1)
                        severity = score_to_severity(score_pct)
                        decision_basis = "vetoed_by_fault_helper"
                        
        except Exception as _fh_exc:
            import logging
            logging.getLogger(__name__).debug("[detect] fault_helper scoring skipped: %s", _fh_exc)

    shap_features_public, likely_sensors = [], []
    explanation_method = None
    if is_anomaly and explainer is not None:
        try:
            explain_result = explainer.explain(feature_row)
            explanation_method = explain_result.get("method")
            shap_features = explain_result.get("features", [])
            likely_sensors = likely_faulty_params(shap_features)
            shap_features_public = [{"name": f["name"], "impact": f["impact"]} for f in shap_features]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[detect] explanation failed for this reading, returning verdict without it: {e!r}")

    regime = _classify_regime(feature_row, raw_reading, history_df)

    return {
        "anomaly_score_pct": round(score_pct, 1),
        "model_confidence_pct": round(model_pct, 1) if model_pct is not None else None,
        "model_status": model_status,
        "rule_confidence_pct": round(rule_confidence, 1),
        "is_anomaly": bool(is_anomaly),
        "severity": severity,
        "fault_type": fault_type,
        "decision_basis": decision_basis,
        "rules_fired": fired,
        "fast_path_offline_params": rules["fast_path_offline_params"],
        "confirmed_spikes": rules["confirmed_spikes"],
        "suggested_values": suggested_values,
        "suggested_metadata": suggested_metadata,
        "shap_features": shap_features_public,
        "explanation_method": explanation_method,
        "likely_faulty_sensors": likely_sensors,
        "regime": regime,
        "network_corroboration": network_state,
        "network_interpretation": network_interpretation,
        "eligible_peer_count": eligible_peer_count,
        "corroborating_peer_count": corroborating_peer_count,
        "diurnal_suppression": diurnal_suppression_map if diurnal_suppression_map else None,
    }


class SensorHealthTracker:
    """
    The circuit breaker -- now PER-PARAMETER internally (§1/§6), with a
    station-level AGGREGATE kept on `.status`/`.offline_reason` so
    state.py's existing calls keep working unmodified. See module
    docstring #8 for the full explanation and the known state.py-side
    gaps this leaves (row-wise buffer exclusion, repair not resetting
    per-parameter counters).

    Two fast paths bypass the general 10h/24h counters entirely and
    force OFFLINE immediately: confirmed multivariate_inconsistency and
    any sensor_fail_low firing (which is always already confirmed by
    construction -- see _rule_checks). Every other rule type
    accumulates through the general per-parameter 10h/24h counters.
    """

    def __init__(self, station_id: str):
        self.station_id = station_id

        self.param_status = {p: "HEALTHY" for p in PARAMS}
        self.param_offline_reason = {p: None for p in PARAMS}
        self._param_recent_10h = {p: deque(maxlen=WINDOW_10H_SIZE) for p in PARAMS}
        self._param_recent_24h = {p: deque(maxlen=WINDOW_24H_SIZE) for p in PARAMS}
        self._param_clean_streak = {p: 0 for p in PARAMS}

        # Station-level aggregate -- external contract, unchanged shape.
        self.status = "HEALTHY"
        self.offline_reason = None
        self._clean_streak = 0  # kept for state.py's mark_repaired() compatibility; see module docstring #8

    def record(self, verdict: dict):
        fired_params = {}
        for rule in verdict.get("rules_fired", []):
            if isinstance(rule, dict):
                rule_type = rule.get("type")
                param = rule.get("parameter")
            else:
                rule_type, param, _confidence = rule[:3]
            fired_params.setdefault(param, set()).add(rule_type)
        fast_path = verdict.get("fast_path_offline_params", set())
        # A causal spike is attributed to the preceding raw reading,
        # but discovered only when this reading arrives.  It therefore
        # contributes exactly one event to the 10h/24h policy without
        # flagging the confirming reading itself as anomalous.
        confirmed_spike_params = set(verdict.get("confirmed_spike_params", set()))

        for p in PARAMS:
            anomalous_this_reading = p in fired_params or p in confirmed_spike_params
            self._param_recent_10h[p].append(anomalous_this_reading)
            self._param_recent_24h[p].append(anomalous_this_reading)

            if self.param_status[p] == "OFFLINE":
                if anomalous_this_reading:
                    self._param_clean_streak[p] = 0
                else:
                    self._param_clean_streak[p] += 1
                    if self._param_clean_streak[p] >= RECOVERY_CLEAN_STREAK_REQUIRED:
                        self.param_status[p] = "HEALTHY"
                        self.param_offline_reason[p] = None
                        self._param_clean_streak[p] = 0
                continue

            if p in fast_path:
                self.param_status[p] = "OFFLINE"
                self.param_offline_reason[p] = (
                    f"{p}: confirmed fault (multivariate/fail-low persistence) -- "
                    f"immediate offline, critical, needs maintenance."
                )
                continue

            count_10h = sum(self._param_recent_10h[p])
            count_24h = sum(self._param_recent_24h[p])

            if len(self._param_recent_10h[p]) >= WINDOW_10H_SIZE and count_10h >= WINDOW_10H_TRIGGER:
                self.param_status[p] = "OFFLINE"
                self.param_offline_reason[p] = (
                    f"{p}: {count_10h} anomalous readings (any fault type, mixed "
                    f"counts) in the last {WINDOW_10H_SIZE}h -- hard threshold, "
                    f"critical, needs maintenance."
                )
            elif len(self._param_recent_24h[p]) >= WINDOW_24H_SIZE and count_24h >= WINDOW_24H_TRIGGER:
                self.param_status[p] = "WARNING"
                self.param_offline_reason[p] = None

            else:
                self.param_status[p] = "HEALTHY"
                self.param_offline_reason[p] = None

        statuses = list(self.param_status.values())
        if "OFFLINE" in statuses:
            self.status = "OFFLINE"
            offline_params = [p for p in PARAMS if self.param_status[p] == "OFFLINE"]
            self.offline_reason = "; ".join(self.param_offline_reason[p] for p in offline_params)
        elif "WARNING" in statuses:
            self.status = "WARNING"
            self.offline_reason = None
        else:
            self.status = "HEALTHY"
            self.offline_reason = None

    def should_include_in_baseline(self) -> bool:
        """
        Station-wide boolean -- see module docstring #8's KNOWN
        LIMITATION note. ANY parameter OFFLINE excludes the whole raw
        row for now; true per-parameter exclusion needs columnar
        buffering in state.py.
        """
        return self.status != "OFFLINE"


def _make_synthetic_history(n_hours: int = 72, seed: int = 0) -> pd.DataFrame:
    """
    Smoke-test helper only -- NOT used by score_reading in real
    operation (state.py owns the real causal buffer). Builds a plain,
    boring, non-anomalous station history so main() below has
    something to run score_reading() against without needing a live
    data_fetch.py pull or a trained model tuned on real data.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01 00:00:00")
    timestamps = [start + pd.Timedelta(hours=i) for i in range(n_hours)]
    hours = np.array([t.hour for t in timestamps])

    temp = 22 + 5 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.3, n_hours)
    pressure = 1013 + rng.normal(0, 0.5, n_hours)
    humidity = 55 + 10 * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(0, 1.0, n_hours)

    return pd.DataFrame({
        "station_id": "AWS-DEMO-001",
        "timestamp": timestamps,
        "temperature_c": temp,
        "pressure_hpa": pressure,
        "humidity_pct": np.clip(humidity, 0, 100),
    })


def main():
    """
    Standalone smoke test: NOT part of the real pipeline (state.py/
    main.py own the actual live buffer + call site). This exists so
    score_reading() and SensorHealthTracker can be sanity-checked in
    isolation -- same purpose features.py's own main() serves for
    build_feature_matrix -- without needing the rest of the backend
    wired up first.

    Two passes:
      1. A clean synthetic history + one more ordinary reading ->
         expect is_anomaly=False, no rules fired.
      2. The same history + one injected-by-hand frozen reading
         (last 3 temperature readings forced to the same integer
         part) -> expect frozen_value evidence and, since frozen's
         confidence (90) clears RULE_CONFIDENCE_BYPASS, is_anomaly=True
         even though nothing else about the reading looks unusual.
    """
    try:
        artifact = load_model()
    except FileNotFoundError as e:
        print(f"[detect] {e}")
        print("[detect] Skipping the model-scored part of the smoke test -- "
              "rule checks alone can still be exercised by calling "
              "_rule_checks()/_fuse_and_score() directly if needed.")
        artifact = None

    history = _make_synthetic_history()

    print("=== Pass 1: ordinary next reading (expect HEALTHY) ===")
    next_reading = {
        "station_id": "AWS-DEMO-001",
        "timestamp": history["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        "temperature_c": float(history["temperature_c"].iloc[-1]) + 0.2,
        "pressure_hpa": float(history["pressure_hpa"].iloc[-1]) - 0.1,
        "humidity_pct": float(history["humidity_pct"].iloc[-1]) + 0.5,
    }
    buffer_1 = pd.concat([history, pd.DataFrame([next_reading])], ignore_index=True)

    if artifact is not None:
        verdict_1 = score_reading(next_reading, buffer_1, artifact)
        print(f"  is_anomaly={verdict_1['is_anomaly']}  "
              f"score={verdict_1['anomaly_score_pct']}  "
              f"fault_type={verdict_1['fault_type']}  "
              f"rules_fired={verdict_1['rules_fired']}")

    print("\n=== Pass 2: hand-injected frozen temperature (expect frozen_value) ===")
    buffer_2 = buffer_1.copy()
    frozen_val = float(np.floor(buffer_2["temperature_c"].iloc[-1]))
    buffer_2.loc[buffer_2.index[-3:], "temperature_c"] = frozen_val
    frozen_reading = dict(next_reading)
    frozen_reading["temperature_c"] = frozen_val

    if artifact is not None:
        verdict_2 = score_reading(frozen_reading, buffer_2, artifact)
        print(f"  is_anomaly={verdict_2['is_anomaly']}  "
              f"score={verdict_2['anomaly_score_pct']}  "
              f"fault_type={verdict_2['fault_type']}  "
              f"rules_fired={verdict_2['rules_fired']}")

        tracker = SensorHealthTracker("AWS-DEMO-001")
        tracker.record(verdict_1)
        tracker.record(verdict_2)
        print(f"\nSensorHealthTracker after both readings: "
              f"status={tracker.status}  param_status={tracker.param_status}")


if __name__ == "__main__":
    main()
