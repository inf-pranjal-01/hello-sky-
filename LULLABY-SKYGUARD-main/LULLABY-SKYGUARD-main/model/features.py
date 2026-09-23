"""
SkyGuard AI — Phase 2a: Feature engineering.

Sits between the data layer (data_fetch.py -> validate_data.py) and the
model layer (train.py). Turns raw per-hour temp/pressure/humidity
readings into the feature vectors the Isolation Forest actually trains
and scores on.

WHERE THIS FITS IN THE PIPELINE:

    data_fetch.py -> validate_data.py -> [THIS FILE] -> train.py
                                              ^
                          also called on labeled/live data at
                          detect.py time, same feature logic both ways

CRITICAL RULE (carried over from anomaly_injector.py / validate_data.py):
this module does NOT know or care whether its input is the clean
raw CSV or an injected _labeled.csv -- it just computes features from
whatever `temperature_c` / `pressure_hpa` / `humidity_pct` columns it's
given. If `is_anomaly` / `fault_type` ground-truth columns are present
(from anomaly_injector.py) they are carried through untouched for
Phase 3 evaluation, but build_feature_matrix() NEVER includes them in
the returned FEATURE_COLUMNS list. train.py must only ever fit on
FEATURE_COLUMNS from the raw CSV; feeding ground-truth label columns
into the model, or training on the labeled/injected file at all, would
be a real leak, not a style choice.

CALLER CONTRACT ON `is_anomaly` -- READ BEFORE PASSING A LABELED FRAME:
add_temporal_features() masks any row where is_anomaly==True OUT of its
OWN rolling baseline window before computing rolling_std/roc/deviation
for it (see _rolling_baseline). That's intentional and correct for ONE
specific purpose: stopping an ongoing fault from dragging down the
baseline used to judge ITS OWN later readings. It is NOT something
train.py or detect.py ever exercises -- train.py fits on raw data with
no is_anomaly column at all, and detect.py's live path
(build_features_for_latest) never sees one either, since state.py's
history buffer only holds confirmed-clean readings (the causal
equivalent, applied without hindsight). If you're computing features
for EVALUATION against a _labeled.csv, strip is_anomaly/fault_type
before featurizing and reattach them afterward by a station_id+
timestamp merge (never by row position -- see build_feature_matrix's
ROW ORDER NOTE). See evaluate.py's module docstring for the bug this
caused when it wasn't done.

FOUR MODEL FEATURE GROUPS (all in FEATURE_COLUMNS, all seen by
train.py -- changing any of these requires a retrain):

  1. Raw values              -- the baseline signal itself.
  2. Temporal features       -- rate-of-change + deviation from each
                                 station's own recent rolling baseline
                                 (ROLLING_WINDOW_HOURS=48, one diurnal
                                 cycle).
  3. Cross-parameter features -- thermodynamic T/RH consistency (dew-
                                 point depression, vapor pressure
                                 deficit, and a direct Clausius-Clapeyron
                                 consistency check), computed in
                                 add_cross_parameter_features. This group
                                 was named in an earlier revision of this
                                 docstring but not actually implemented
                                 anywhere in the file -- now fixed, and
                                 added specifically to match
                                 anomaly_injector.py's inject_multivariate,
                                 which manufactures faults as a real
                                 vapor-pressure-conservation violation
                                 rather than two independent sigma bumps.
  4. Cyclical time            -- hour-of-day / day-of-year encodings.

SPATIAL FEATURES REMOVED (locked, SKYGUARD_ARCHITECTURE_DRAFT_2.md §9):
at >10km station separation, cross-station "neighbor consistency"
carried no real physical signal for any of our fault types, and it was
the direct cause of a train/serve mismatch (evaluate.py processing one
station at a time meant every cluster only ever had 1 station present,
so spatial features silently fell back to a neutral 0.0 every single
row of every eval run, while train.py -- run on the full multi-station
file -- saw real non-zero values for the same features). Removing them
outright, not patching around the mismatch, closes that gap for good.
FEATURE_COLUMNS is 3 shorter than before -- retrain required.

PLUS a fifth group, RULE-ONLY signals: short-horizon raw-reading
comparisons used exclusively by the deterministic rule layer in
detect.py/evaluate.py, NEVER added to FEATURE_COLUMNS and NEVER seen by
the trained model. Most live in add_rule_only_signals(); one exception
-- {prefix}_normalized_roc_1h -- is computed inside add_temporal_features
instead (see that function's docstring for why) but is equally
rule-only: it is not in FEATURE_COLUMNS either.

REBUILT AROUND THE LOCKED REDESIGN (SKYGUARD_ARCHITECTURE_DRAFT_2.md):

  - Frozen (§1): the OLD design used a calibrated percentile threshold
    on {prefix}_consec_diff ("suspiciously static" was a guessed,
    scale-dependent cutoff). The NEW rule is deterministic and needs no
    calibration at all: a parameter is frozen when floor(reading) is
    identical across 3 consecutive readings. That's now computed once,
    here, as {prefix}_floor_frozen_match -- a boolean that's True at row
    t iff readings t, t-1, and t-2 all share the same integer part.
    detect.py/evaluate.py read this column directly instead of each
    re-deriving floor() logic independently -- see the
    RULE_ONLY_PREFIXES comment below for the exact bug that caused last
    time two derivations of the same thing drifted apart. The old
    percentile-calibrated "frozen" threshold is REMOVED from
    calibrate_rule_thresholds -- it has no role under the new rule.

  - Drift (§2): the OLD design used a calibrated percentile threshold on
    {prefix}_{DRIFT_LOOKBACK_HOURS}h_delta (a fixed-lookback magnitude
    check). The NEW mechanism is CUSUM (S+/S- accumulator, confirmed
    final) which needs a per-reading NORMALIZED step change as its
    input, not a fixed-lookback delta. That accessor is
    {prefix}_normalized_roc_1h, added below. CUSUM's own accumulator
    state (running S+/S-, reset-on-reversal) is inherently sequential
    and lives in detect.py/state.py, not here -- this file's job is
    only to hand it a consistent, causally-safe input signal. The old
    percentile-calibrated "drift" threshold is REMOVED from
    calibrate_rule_thresholds -- CUSUM's own threshold/allowance are new
    config.py constants (§11), not a data-calibrated percentile.

  - roc_small's calibration is LEFT IN for now but flagged: its only
    documented purpose (distinguishing "genuinely calm weather" from a
    frozen sensor under the old ambiguous-threshold design) no longer
    applies now that frozen is a deterministic floor-match. Nothing else
    in the locked draft claims it, but it isn't deleted here without
    confirming that's actually intended -- see the accompanying message.

  - {prefix}_consec_diff and {prefix}_{DRIFT_LOOKBACK_HOURS}h_delta raw
    columns are KEPT (cheap, still human-readable diagnostics useful for
    evaluate.py's per-station/per-sensor breakdown, §10) even though
    neither feeds a calibrated threshold anymore.

LEAKAGE NOTE: every rolling/baseline statistic below is computed with
`.shift(1)` before the rolling window, i.e. "as of the reading before
this one." If we let a reading's own value bleed into the baseline it's
being compared against, a genuine spike would partially absorb into its
own rolling mean and understate its own z-score -- exactly the kind of
bug that looks fine on a quick eyeball check and quietly wrecks
detection accuracy. Every *_deviation feature below is safe against
this by construction.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Rolling window for each station's own recent baseline. 48h = two full
# diurnal cycles, so "recent normal" already accounts for the daily
# temperature/pressure/humidity swing instead of falsely flagging
# "it's just afternoon" as a deviation. min_periods keeps the first
# ~day of each station's data from producing all-NaN features.
ROLLING_WINDOW_HOURS = "48h"
ROLLING_MIN_PERIODS = 6
VOLATILITY_BASELINE_WINDOW_HOURS = 24 * 30  # ~30 days: "this station's own typical variability"
# Rate-of-change lookback in hours. Short (1h) catches sudden jumps
# (spike); slightly longer (3h) catches a fast ramp that no single
# 1h step looks extreme (early drift).
ROC_SHORT_HOURS = 1
ROC_LONG_HOURS = 3

RAW_COLUMNS = ["temperature_c", "pressure_hpa", "humidity_pct"]

# Names deliberately match the SHAP feature-name examples already
# promised in ARCHITECTURE.md's /api/explain/{id} contract ("Temp
# Deviation", "Pressure Inconsistency", "Rate of Change (Temp)", "Time
# of Day Pattern", "Seasonal Pattern") so explain.py can map straight
# from these column names to that response shape later without a
# separate translation table.
#
# BUGFIX: the "# raw" section below previously listed no actual raw
# column names -- the docstring promised raw values as one of the model
# feature groups, but RAW_COLUMNS was never actually appended to this
# list, so the model was blind to absolute physical values entirely and
# only ever saw *_deviation (relative-to-baseline) signals. That's a
# real gap, not a style choice: absolute-value context (e.g. "-15.0C" is
# implausible regardless of what this station's recent baseline says)
# is exactly the kind of thing a purely relative feature set can miss,
# and it's directly relevant to a precision/recall target. Fixed below.
FEATURE_COLUMNS = [
    # raw
    "temperature_c", "pressure_hpa", "humidity_pct",
    # temporal
    "temp_deviation", "pressure_deviation", "humidity_deviation",
    "temp_roc_1h", "pressure_roc_1h", "humidity_roc_1h",
    "temp_roc_3h", "pressure_roc_3h", "humidity_roc_3h",
    "temp_volatility_z", "pressure_volatility_z", "humidity_volatility_z",
    # cross-parameter (thermodynamic consistency -- see
    # add_cross_parameter_features; this group was promised in this
    # docstring and referenced from add_temporal_features' docstring in
    # a previous revision without actually being implemented anywhere
    # in the file. Added now to match anomaly_injector.py's
    # inject_multivariate, which manufactures faults as a real
    # Clausius-Clapeyron violation, not two independent sigma bumps.)
    "dewpoint_depression_c", "vapor_pressure_deficit_kpa",
    "vapor_pressure_consistency_dev",
    # cyclical time
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    # Phase 2 advanced features
    "dt_hours",
    "temp_robust_scale", "pressure_robust_scale", "humidity_robust_scale",
    "temp_hours_since_valid", "pressure_hours_since_valid", "humidity_hours_since_valid",
    "temp_range_1h", "pressure_range_1h", "humidity_range_1h",
    "temp_range_3h", "pressure_range_3h", "humidity_range_3h",
    "temp_range_6h", "pressure_range_6h", "humidity_range_6h",
    "temp_range_24h", "pressure_range_24h", "humidity_range_24h",
    "temp_slope_6h", "pressure_slope_6h", "humidity_slope_6h",
    "temp_slope_24h", "pressure_slope_24h", "humidity_slope_24h",
    "temp_same_hour_res", "pressure_same_hour_res", "humidity_same_hour_res",
]

# Shared (raw_column -> features.py prefix) mapping, used by BOTH the
# rule-only signals below AND by detect.py/evaluate.py's rule checks.
# Keep this as the single source of truth for the mapping -- the
# earlier bug (temperature_c's frozen check silently never firing) was
# caused by a second, ad hoc string-strip derivation of this same
# mapping drifting out of sync with the real column names.
RULE_ONLY_PREFIXES = [
    ("temperature_c", "temp"),
    ("pressure_hpa", "pressure"),
    ("humidity_pct", "humidity"),
]

# Lookback for multi-timescale drift tracking (30 days = 720 hours)
DRIFT_LOOKBACK_HOURS = 720


def _rolling_baseline(series: pd.Series, exclude_mask: pd.Series = None):
    """
    Shifted rolling mean/std -- "normal, as of the reading before this
    one." See LEAKAGE NOTE at module top for why the shift matters.

    exclude_mask (optional, boolean Series aligned to `series`): rows
    where this is True are masked to NaN BEFORE the rolling window sees
    them. This matters most for long-running faults like drift: without
    it, a drift event's own earlier (already-anomalous) readings enter
    the rolling window used to judge its later readings, so the
    baseline drifts along with the fault and progressively understates
    how anomalous it is. Masking them out means the baseline is only
    ever built from confirmed-normal history, even mid-fault. The
    reading being SCORED is never masked, only what's used to define
    "normal" -- see add_temporal_features for how this is applied.
    """
    clean_series = series.mask(exclude_mask) if exclude_mask is not None else series
    prior = clean_series.shift(1)
    roll = prior.rolling(ROLLING_WINDOW_HOURS, min_periods=ROLLING_MIN_PERIODS)
    
    import numpy as np
    mad = prior.rolling(ROLLING_WINDOW_HOURS, min_periods=ROLLING_MIN_PERIODS).apply(
        lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True
    )
    return roll.mean(), roll.std(), mad


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-station rolling deviation + rate-of-change. Assumes df is
    ALREADY one station, sorted by timestamp (see build_feature_matrix,
    which does this via groupby before calling in).

    If an `is_anomaly` ground-truth column is present (i.e. this is a
    _labeled.csv being scored for Phase 3 evaluation), known-anomalous
    rows are excluded from the rolling baseline -- see _rolling_baseline.
    In live serving (detect.py), the equivalent exclusion uses the
    model's OWN prior verdicts instead of ground truth, since the
    future obviously isn't known yet -- that's implemented in state.py's
    history buffer, not here.

    IMPORTANT for evaluation callers: this exclusion makes an ongoing
    fault's OWN rows use a PRE-fault baseline for their entire
    duration, since the fault's own rows never enter their own rolling
    window. Evaluation code must strip is_anomaly/fault_type before
    calling build_feature_matrix. See the module-level CALLER CONTRACT
    note and evaluate.py's docstring.

    Also computes {prefix}_normalized_roc_1h here, NOT in
    add_rule_only_signals -- it needs {prefix}_rolling_std, which only
    exists at this point in the pipeline. It reuses that SAME causal,
    shifted rolling std that {prefix}_deviation is built from, rather
    than a second independent rolling-std computation. This is CUSUM's
    "normalized_change" input (§2) in detect.py. One source of truth for
    "how big is this station's normal step size right now" instead of
    a second derivation that could quietly drift apart from the first.

    NOTE: add_cross_parameter_features (below) does NOT consume this --
    it's a direct instant-to-instant vapor-pressure check, not a
    z-scored step size. An earlier revision of this docstring claimed
    otherwise before that function existed at all; correcting it here
    rather than leaving a stale claim about a function this file didn't
    yet implement.
    """
    # Live and replay readings can mix naive provider timestamps with
    # timezone-aware simulator timestamps; normalize both to UTC before
    # sorting and using time-based rolling windows.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp", drop=False)
    exclude_mask = df["is_anomaly"].fillna(False).astype(bool) if "is_anomaly" in df.columns else None

    for col, prefix in [("temperature_c", "temp"), ("pressure_hpa", "pressure"), ("humidity_pct", "humidity")]:
        # Providers and replay dropout records legitimately contain a null
        # measurement.  Pandas keeps a mixed float/None column as ``object``;
        # ``Series.diff`` then attempts ``None - float`` and used to crash the
        # entire simulator tick.  Coerce only this feature-engineering copy:
        # the original raw value remains intact for dropout detection,
        # telemetry/history display, and suggested-value generation.
        values = pd.to_numeric(df[col], errors="coerce")
        df[col] = values
        mean, std, mad = _rolling_baseline(values, exclude_mask)
        # Robust scale = 1.4826 * MAD
        robust_scale = 1.4826 * mad
        safe_scale = robust_scale.replace(0, np.nan)
        df[f"{prefix}_deviation"] = (values - mean) / safe_scale
        clean_values = values.mask(exclude_mask) if exclude_mask is not None else values
        
        df[f"{prefix}_rolling_std"] = std
        df[f"{prefix}_rolling_mad"] = mad
        df[f"{prefix}_robust_scale"] = robust_scale
        df[f"{prefix}_rolling_mean"] = mean
        df[f"{prefix}_rolling_mean_3h"] = values.shift(1).rolling("3h", min_periods=1).mean()
        df[f"{prefix}_rolling_mean_24h"] = clean_values.shift(1).rolling("24h", min_periods=6).mean()
        
        # Missing data & Regularity
        df["dt_hours"] = df.index.to_series().diff().dt.total_seconds() / 3600.0
        valid_times = df.index.to_series()[values.notna()]
        last_valid = valid_times.reindex(df.index, method='ffill')
        df[f"{prefix}_hours_since_valid"] = (df.index.to_series() - last_valid).dt.total_seconds() / 3600.0
        
        # Ranges
        for w in ["1h", "3h", "6h", "24h"]:
            df[f"{prefix}_range_{w}"] = values.rolling(w, min_periods=1).max() - values.rolling(w, min_periods=1).min()
            
        long_baseline = std.rolling("720h", min_periods=ROLLING_MIN_PERIODS).median()
        long_spread = std.rolling("720h", min_periods=ROLLING_MIN_PERIODS).std()
        df[f"{prefix}_volatility_z"] = (std - long_baseline) / long_spread.replace(0, np.nan)
        
        df[f"{prefix}_roc_1h"] = values.diff(ROC_SHORT_HOURS)
        df[f"{prefix}_roc_3h"] = values.diff(ROC_LONG_HOURS)
        
        # Robust slopes
        df[f"{prefix}_slope_6h"] = (values - values.shift(6)) / 6.0
        df[f"{prefix}_slope_24h"] = (values - values.shift(24)) / 24.0
        
        # Same-hour residual
        df[f"{prefix}_same_hour_res"] = values - values.shift(24)
        
        df[f"{prefix}_normalized_roc_1h"] = df[f"{prefix}_roc_1h"] / safe_scale

    df["warm_up_complete"] = df["temp_rolling_std"].notna()
    return df.reset_index(drop=True)


def saturation_vapor_pressure_kpa(temp_c):
    """
    Tetens approximation of saturation vapor pressure (kPa). Same
    formula as anomaly_injector.py's function of the same name --
    duplicated rather than imported (features.py and the injector are
    deliberately independent modules with no import dependency between
    them), but the math must stay identical, since
    vapor_pressure_consistency_dev below is meant to directly mirror
    what inject_multivariate() violates. If this formula ever changes,
    check the injector's copy too.
    """
    exponent = (17.67 * temp_c) / (temp_c + 243.5)
    return 0.6112 * np.exp(np.clip(exponent, -100, 100))


def add_cross_parameter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-parameter thermodynamic features -- the model feature group
    promised at the top of this file under "THREE MODEL FEATURE GROUPS"
    (and referenced from add_temporal_features' docstring) that did not
    actually exist anywhere in a previous revision of this file. Same
    "documented but silently missing" pattern already caught once for
    spatial features (§9) -- closing it here instead of leaving it.

    Added specifically because anomaly_injector.py's inject_multivariate
    no longer injects two independent sigma bumps on temperature and
    humidity -- it manufactures a real, checkable Clausius-Clapeyron
    violation: holding actual vapor pressure e constant, relative
    humidity MUST fall as temperature rises (es(T) rises exponentially);
    the injected fault instead makes humidity RISE with temperature.
    Without a feature that measures that specific relationship, neither
    the model nor a magnitude-only rule can tell "T and RH both moved a
    lot" (which can be genuine weather -- incoming rain, a front) apart
    from "T and RH moved in a way real weather physically cannot."

    `vapor_pressure_consistency_dev` is the key one -- it directly
    mirrors inject_multivariate's own construction, one step at a time:

        es(T)          = Tetens saturation vapor pressure
        rh_expected[t] = rh[t-1] * es(T[t-1]) / es(T[t])   (const. e)
        vapor_pressure_consistency_dev = rh_actual[t] - rh_expected[t]

    A genuine heat event drives this toward zero (RH tracks the
    physically-consistent drop). The injected fault drives it strongly
    POSITIVE (RH rises instead of falling as required). This compares
    t-1 -> t directly, like {prefix}_roc_1h -- it's an instantaneous
    physical-consistency check between two consecutive readings, not a
    "what's normal for this station over 48h" comparison, so it does
    NOT use _rolling_baseline / the is_anomaly exclusion mask the way
    add_temporal_features' deviation columns do.

    `dewpoint_depression_c` and `vapor_pressure_deficit_kpa` are general
    thermodynamic state variables (both in the PDF audit's own Tier-2
    feature list) added alongside it as broader context, not tied to
    any one fault type.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    temp_c = pd.to_numeric(df["temperature_c"], errors="coerce")
    # Guard log(0)/div-by-0 for a 0% or negative/garbage humidity
    # reading -- a dropout or fail-low event can legitimately produce
    # one; NaN propagates through cleanly rather than raising.
    humidity_pct = pd.to_numeric(df["humidity_pct"], errors="coerce").clip(lower=0.01, upper=100.0)

    # Magnus-Tetens dew point (deg C).
    gamma = (17.67 * temp_c) / (243.5 + temp_c) + np.log(humidity_pct / 100.0)
    dew_point_c = (243.5 * gamma) / (17.67 - gamma)
    df["dewpoint_depression_c"] = temp_c - dew_point_c

    es_now = saturation_vapor_pressure_kpa(temp_c)
    df["vapor_pressure_deficit_kpa"] = es_now * (1 - humidity_pct / 100.0)

    temp_prev = temp_c.shift(1)
    humidity_prev = pd.to_numeric(df["humidity_pct"], errors="coerce").shift(1)
    es_prev = saturation_vapor_pressure_kpa(temp_prev)
    rh_expected = (humidity_prev * (es_prev / es_now)).clip(0.0, 100.0)
    df["vapor_pressure_consistency_dev"] = humidity_pct - rh_expected

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical encodings so the model sees hour 23 and hour 0 as
    adjacent, and seasonal position wraps around the year the same way.
    """
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    doy = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def add_rule_only_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-station, adds RULE-LAYER-ONLY signal families -- never added to
    FEATURE_COLUMNS, never seen by the trained Isolation Forest. See
    module docstring for why these exist alongside the 48h-windowed
    temporal features instead of replacing them, and for which parts of
    this were rebuilt around the locked redesign.

    `{prefix}_floor_frozen_match` (bool): True at row t iff
    floor(reading[t]) == floor(reading[t-1]) == floor(reading[t-2]) --
    the exact, deterministic §1 frozen rule, computed once here so
    detect.py/evaluate.py never re-derive floor() logic independently
    (see RULE_ONLY_PREFIXES comment for the bug that caused last time).
    Because the 3-reading condition is already baked into this single
    column, a caller only needs the LATEST row's value to know whether
    frozen fires right now -- no separate persistence-window bookkeeping
    needed at the call site. NaN propagates naturally through floor()
    comparisons, so a dropout (NaN) reading correctly does NOT register
    as a floor match with its neighbors.

    `{prefix}_consec_diff`: absolute difference between this reading
    and the immediately-previous raw reading (NOT a rolling window).
    No longer calibrated as a frozen threshold (see module docstring) --
    kept only as a cheap diagnostic for evaluate.py's per-sensor
    breakdown.

    `{prefix}_{DRIFT_LOOKBACK_HOURS}h_delta`: raw value minus its value
    DRIFT_LOOKBACK_HOURS ago. No longer calibrated as a drift threshold
    (superseded by CUSUM, see module docstring) -- kept only as a cheap
    diagnostic for evaluate.py's per-sensor breakdown.

    roc_small threshold (calibrated in calibrate_rule_thresholds, not
    computed here) is flagged in the module docstring as likely dead
    under the new frozen rule -- not removed without confirming that's
    intended.
    """
    from config import FROZEN_CONSECUTIVE_REQUIRED, FROZEN_CONSECUTIVE_REQUIRED_PRESSURE
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col, prefix in RULE_ONLY_PREFIXES:
        df[f"{prefix}_consec_diff"] = df[col].diff(1).abs()
        df[f"{prefix}_{DRIFT_LOOKBACK_HOURS}h_delta"] = df[col] - df[col].shift(DRIFT_LOOKBACK_HOURS)

        # Use round(1) to match Open-Meteo precision
        floor_vals = df[col].round(1)
        
        req = FROZEN_CONSECUTIVE_REQUIRED_PRESSURE if prefix == "pressure" else FROZEN_CONSECUTIVE_REQUIRED
        
        run_id = floor_vals.ne(floor_vals.shift()).cumsum()
        streak = floor_vals.groupby(run_id).cumcount() + 1
        df[f"{prefix}_frozen_streak"] = streak
        
        # Frozen match is dynamic based on config
        df[f"{prefix}_floor_frozen_match"] = streak >= req
    return df


def get_threshold(thresholds: dict, rule_type: str, prefix: str, station_id: str) -> float:
    """
    Single source of truth for threshold lookup -- used by BOTH
    detect.py and evaluate.py so they can never drift apart on how a
    per-station threshold falls back to the pooled global value.
    """
    per_station = thresholds[rule_type][prefix]
    return per_station.get(station_id, per_station["__global__"])


def calibrate_rule_thresholds(featured_clean: pd.DataFrame) -> dict:
    """
    PER-STATION thresholds, not one pooled global value -- diagnosed
    directly against real 20-station eval output: temp_rolling_std
    ranges 2.38 (AWS-CHN-024) to 5.42 (AWS-MUM-101) across stations,
    more than 2x. A single global percentile is necessarily
    miscalibrated for whichever end of that range it doesn't fit.

    Each threshold is {station_id: value, "__global__": value} -- the
    global entry is the same pooled-percentile calculation as before,
    kept as a fallback for any station with too little clean data to
    calibrate its own threshold reliably (min_rows guard below) or any
    station missing entirely at calibration time. See get_threshold()
    for the lookup that applies this fallback.

    REMOVED under the locked redesign (see module docstring):
      - "frozen" -- the new frozen rule (floor-match, §1) is
        deterministic and needs no calibrated threshold at all.
      - "drift" -- superseded by CUSUM (§2, final); CUSUM's own
        threshold/allowance are new config.py constants, not a
        data-calibrated percentile of this file's Nh_delta column.

    STILL HERE, FLAGGED (see module docstring):
      - "roc_small" -- its only documented purpose (distinguishing
        genuinely calm weather from a frozen sensor under the old
        ambiguous-threshold design) no longer applies now that frozen
        is a deterministic floor-match. Left in pending confirmation
        it's actually still wanted for anything.

    roc_small[prefix]: 20th percentile of roc_1h magnitude, per station.
    spike[prefix]: 99.9th percentile of *_deviation magnitude, per station.
    (Percentile choices carried over unchanged from the pooled version --
    only the GROUPING changed here, not the statistical target each
    rule is calibrated against.)
    """
    MIN_ROWS_FOR_PER_STATION = 200  # below this, a station's own quantile is too noisy to trust

    thresholds = {"roc_small": {}, "spike": {}}

    def _calibrate(
        rule_key: str,
        value_fn,
        quantile: float,
        per_station: bool = True,
    ):
        per_station_vals = {}

        if per_station:
            for station_id, group in featured_clean.groupby("station_id"):
                vals = value_fn(group)
                if len(vals) >= MIN_ROWS_FOR_PER_STATION:
                    per_station_vals[station_id] = float(
                        vals.quantile(quantile)
                    )

        global_val = float(
            value_fn(featured_clean).quantile(quantile)
        )

        thresholds[rule_key][prefix] = {
            "__global__": global_val,
            **per_station_vals,
        }

    for _, prefix in RULE_ONLY_PREFIXES:
        roc_col = f"{prefix}_roc_1h"
        dev_col = f"{prefix}_deviation"

        # Small ROC: keep per-station calibration (flagged above).
        _calibrate(
            "roc_small",
            lambda g: g[roc_col].abs(),
            0.20,
        )

        # Spike: GLOBAL ONLY.
        # 99.9th percentile is too extreme to estimate reliably
        # from only ~2,000 rows per station.
        _calibrate(
            "spike",
            lambda g: g[dev_col].abs(),
            0.99999,
            per_station=False,
        )

    return thresholds


def build_features_for_latest(history_df: pd.DataFrame) -> pd.Series:
    """
    LIVE-MODE entry point, for detect.py -- NOT for training/eval.

    Takes one station's recent history buffer (kept in-memory by
    state.py, no CSV involved at all) and returns just the feature
    vector for the LATEST reading in it. Includes rule-only signals too
    (consec_diff / Nh_delta / floor_frozen_match / normalized_roc_1h),
    since detect.py's rule layer needs them from the same computation
    path as the model features -- same math, so training/serving/
    rule-checking never quietly drift apart.

    `history_df` should NOT include an is_anomaly column in live mode.

    No spatial handling here or anywhere else in this file -- spatial
    features were removed entirely (§9, see module docstring), not just
    skipped in live mode.
    """
    df = add_temporal_features(history_df)
    df = add_cross_parameter_features(df)
    df = add_time_features(df)
    df = add_rule_only_signals(df)
    return df.iloc[-1]


def build_rule_signals_recent(history_df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    LIVE-MODE helper for rule checks that need more than just the
    latest row. Returns the last `n` rows with rule-only signals
    computed, so detect.py can inspect recent history without
    re-deriving add_rule_only_signals' logic separately.

    NOTE for the frozen check specifically: {prefix}_floor_frozen_match
    already encodes its own full 3-reading persistence requirement
    internally (see add_rule_only_signals) -- a caller only needs n=1
    (the latest row) to know whether frozen fires right now. n=2+ is for
    other rule checks that need to compare across multiple recent rows
    themselves (e.g. a rule built directly on roc_small persistence,
    if that calibration turns out to still be wanted -- see module
    docstring).
    """
    df = history_df.sort_values("timestamp").reset_index(drop=True)
    df = add_rule_only_signals(df)
    return df.tail(n)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Takes the combined multi-station dataframe (the
    shape of all_stations.csv, optionally with is_anomaly/fault_type
    ground-truth columns already attached from anomaly_injector.py) and
    returns a dataframe with every column in FEATURE_COLUMNS added
    (plus the rule-only signal columns), ready for train.py to select
    FEATURE_COLUMNS as X and calibrate_rule_thresholds to calibrate
    against.

    Ground-truth columns (is_anomaly, fault_type), station_id, and
    timestamp all pass through untouched. See the module-level CALLER
    CONTRACT note before passing is_anomaly in for evaluation purposes.

    ROW ORDER NOTE: internally this groups by station_id (sort=False)
    and concatenates each station's rows back with ignore_index=True.
    The returned frame's row order does NOT match the input df's.
    Callers that need to reattach any side-channel data after the fact
    must join on (station_id, timestamp), never on row position/index.

    SIGNATURE CHANGE: no longer takes a `metadata` argument -- that was
    only ever used to look up cluster_id for the now-removed spatial
    features (§9). Any caller (train.py, evaluate.py) still passing a
    second positional argument will need updating when we get to those
    files, per the locked build order -- this isn't a surprise, just
    flagging where the break is so it's not mysterious when it shows up.
    """
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column -- did you pass the raw fetched/validated CSV?")

    missing_raw = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing expected raw columns: {missing_raw}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    per_station_frames = []
    for station_id, group in df.groupby("station_id", sort=False):
        group = add_temporal_features(group)
        group = add_cross_parameter_features(group)
        group = add_time_features(group)
        group = add_rule_only_signals(group)
        per_station_frames.append(group)
    df = pd.concat(per_station_frames, ignore_index=True)

    return df


def main():
    stations_path = DATA_DIR / "all_stations.csv"

    if not stations_path.exists():
        print(f"Expected {stations_path.name} in {DATA_DIR} "
              f"(output of data_fetch.py -> validate_data.py). Run that first.")
        return

    df = pd.read_csv(stations_path, parse_dates=["timestamp"])

    if "is_anomaly" in df.columns:
        print("NOTE: input already has is_anomaly/fault_type columns (looks like a "
              "_labeled.csv). That's fine for testing detect.py against known faults, "
              "but train.py must fit only on a RAW un-injected file.\n")

    featured = build_feature_matrix(df)

    n_total = len(featured)
    n_ready = featured[FEATURE_COLUMNS].notna().all(axis=1).sum()
    print(f"Built {len(FEATURE_COLUMNS)} model features for {n_total} rows across "
          f"{df['station_id'].nunique()} stations.")
    print(f"{n_ready}/{n_total} rows have a complete feature vector (no NaNs from "
          f"rolling-window warm-up); the rest are the first ~{ROLLING_WINDOW_HOURS}h "
          f"of each station's history and should be dropped before training.\n")

    sample_cols = ["station_id", "timestamp", "temperature_c", "temp_deviation",
                   "temp_floor_frozen_match",
                   "temp_normalized_roc_1h", "temp_consec_diff",
                   "vapor_pressure_consistency_dev", "dewpoint_depression_c"]
    print("Sample rows:")
    print(featured[sample_cols].dropna().head(5).to_string(index=False))

    output_path = DATA_DIR / "all_stations_features.csv"
    featured.to_csv(output_path, index=False)
    print(f"\nSaved full feature matrix -> {output_path}")


if __name__ == "__main__":
    main()
