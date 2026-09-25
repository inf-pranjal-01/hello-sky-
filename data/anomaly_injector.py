"""
SkyGuard AI — Phase 1b: Synthetic anomaly injector.

Takes real, clean historical readings (from data_fetch.py) and
deliberately corrupts a small fraction of them in known, labeled ways.
This gives us ground truth to actually measure the model against later
(precision/recall/F1 in Phase 3) -- something that doesn't exist for
real AWS anomaly data.

Fault types implemented, each tied to a real AWS failure mode named in
the problem statement:
  - spike           : sensor malfunction -> reading jumps far outside
                       physically plausible range, either as a single
                       instant (digital bit-flip / transmission garble)
                       or with a short RC/thermal-mass decay tail back
                       toward normal (voltage transient / ESD) -- see
                       inject_spike / inject_spike_decay. Both report
                       fault_type="spike"; still six ground-truth fault
                       types total, per the locked spec.
  - frozen_value    : communication/sensor fault -> value stops tracking
                       real atmospheric variation for several consecutive
                       readings. Modeled as a bounded random walk around
                       the freeze point (ADC thermal noise / quantization
                       jitter), NOT a bit-exact repeat -- a real stuck
                       sensor still has electronic noise on top of a
                       static physical signal (Day 42/43 logic still
                       applies: near-zero *variance*, not zero variance,
                       is the anomalous signature).
  - drift           : calibration drift -> a slow, growing offset
                       SUPERIMPOSED on the station's own real recorded
                       readings (which already carry the true diurnal
                       cycle), not a monotonic ramp that overwrites and
                       erases that natural signal.
  - dropout         : communication failure -> missing/null reading
  - sensor_fail_low : hardware rail failure -> reading pinned at the
                       sensor's electrical floor (a ground short/cable
                       break pulls the ADC to 0 counts), not an arbitrary
                       intermediate low value.
  - multivariate_inconsistency : sensor cross-talk / local heating ->
                       temperature and humidity move in a way that
                       violates the Clausius-Clapeyron relation between
                       temperature and saturation vapor pressure (see
                       inject_multivariate).

FIVE FIXES applied to align injected faults with real transducer/
atmospheric physics instead of arbitrary mathematical mutations (each
noted at its function below):
  1. Column-aware overlap tracking (claimed_spans is now {column: [...]}
     instead of one global list) -- the docstring on inject_anomalies
     already claimed independent per-column faults were possible; the
     old flat list didn't actually allow it. Now it does.
  2. Frozen -> bounded random walk (ADC noise floor), not bit-exact.
  3. Spike -> dual-mode: instant (unchanged) + a new decay-tail variant.
  4. Drift -> superimposed on the real underlying signal, not overwritten.
  5. Fail-low -> true hardware rail limits, not arbitrary mid-range values.
  6. Multivariate -> grounded in the Tetens/Clausius-Clapeyron saturation
     vapor pressure curve instead of two independent, uncalibrated sigma
     bumps.

Ground truth (is_injected, fault_type) is stored alongside the data so
Phase 3 can compute real accuracy metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# This script lives in the SAME folder as your fetched CSVs
# (backend/data/AWS-*.csv), unlike data_fetch.py which saves INTO a
# ./data subfolder relative to itself. If your CSVs are somewhere else,
# change this to point at that folder directly.
DATA_DIR = Path(__file__).parent

# How much of the data to corrupt. Keep this modest and realistic --
# real sensor faults are rare events, not half your dataset.
# Base prevalence for the normal test dataset. Do not tune this per
# evaluation result; it represents the project's default scenario.
INJECTION_RATE = 0.05

# BENCHMARK REGIME SWITCH ---------------------------------------------------
# Options:
#   "benchmark_b" -> PCL-compatible operational benchmark (<= 1 fault per cluster per timestamp)
#   "benchmark_a" -> Unrestricted multi-fault stress test (allows simultaneous same-cluster faults)
DEFAULT_REGIME = "benchmark_b"

# BACKEND-ONLY CONTROL KNOB -------------------------------------------------
# Relative amount of injected fault data. Change ONLY this value when you
# want a lighter or heavier replay dataset:
#   0.0 = no injected faults, 0.5 = roughly half normal density,
#   1.0 = normal density, 2.0 = roughly double normal density.
# It scales both the row target and the per-fault minimum, preserving the
# realistic fault-type mix instead of turning one type up in isolation.
ANOMALY_DENSITY_MULTIPLIER = 2.5

# Held-out replay seed: distinct placements and fault directions from
# the initial calibration replay. Change deliberately and record it in
# evaluation output; train.py never consumes these labelled files.
RANDOM_SEED = 20260924

# Fixed (not randomized) fail-low window length. §5b's detector rule
# triggers reclassification at 2-3 consecutive hours -- 3 sits right at
# that bar, guaranteeing every injected fail-low event is long enough
# to be caught, with no per-event variance to account for in eval.
FAIL_LOW_LENGTH = 3


def compute_bounds(series: pd.Series, z_thresh: float = 3.0):
    """
    Day 42 (z-score method): mean +/- z_thresh * std defines the
    'normal' envelope. We use this to make sure injected spikes are
    genuinely, unambiguously outside normal behavior -- not borderline.
    """
    mean = series.mean()
    std = series.std()
    return mean, std, mean + z_thresh * std, mean - z_thresh * std


# Hard physical ceilings that NO fault should cross, because they're
# not just statistically unusual -- they're physically impossible.
# Humidity is the critical one: it's a percentage, so a sensor CANNOT
# genuinely report 173% no matter how broken it is (a real malfunctioning
# sensor saturates/clips at its measurement limits, it doesn't exceed
# them). Pressure gets a generous real-world floor/ceiling too. This is
# NOT applied to inject_fail_low (which intentionally uses an even lower
# fixed sentinel to represent total sensor failure -- a different, valid
# fault archetype) or inject_dropout (NaN has no numeric bound to violate).
HARD_PHYSICAL_LIMITS = {
    "temperature_c": (-50.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "pressure_hpa": (850.0, 1085.0),
}

# FIX 2/3/5: real transducers never sit at a bit-exact repeated value or
# an arbitrary intermediate failure value -- they have a noise floor
# (frozen) or fail all the way to the electrical rail (fail-low). One
# entry per parameter, since each lives on its own physical scale.
ADC_NOISE_FLOOR_STD = {
    # Small enough that it never crosses a floor()-boundary the way the
    # old bit-exact approach was calibrated to avoid, but large enough
    # to be real quantization/thermal jitter, not zero.
    "temperature_c": 0.05,
    "pressure_hpa": 0.03,
    "humidity_pct": 0.05,
}

# How far a frozen random walk is allowed to wander from its starting
# point before being clamped back -- a stuck sensor's noise floor does
# NOT accumulate into a real drift; it's a stationary process.
FROZEN_MAX_DEVIATION = {
    "temperature_c": 0.3,
    "pressure_hpa": 0.2,
    "humidity_pct": 0.4,
}

# FIX 5: true hardware rail limits (0 ADC counts / max ADC counts pulled
# to ground or supply), not an arbitrary "somewhat low" sentinel. These
# are deliberately OUTSIDE HARD_PHYSICAL_LIMITS -- a real ground short
# reads a value no real atmosphere could ever produce, and clip_to_
# physical_limits is intentionally never applied to this fault (see its
# docstring below).
FAIL_LOW_RAIL_VALUE = {
    "temperature_c": -40.0,
    "pressure_hpa": 0.0,
    "humidity_pct": 0.0,
}
# Small, fixed (not floor-proportional -- a proportional formula breaks
# at a 0.0 rail) noise representing residual ADC jitter even at the rail.
FAIL_LOW_NOISE_STD = 0.05


def saturation_vapor_pressure_kpa(temp_c):
    """
    Tetens approximation of saturation vapor pressure (kPa), used to
    ground inject_multivariate() in real Clausius-Clapeyron thermodynamics
    instead of two independently-chosen sigma magnitudes. Vectorized --
    accepts a scalar or a numpy array/Series.
    """
    return 0.6112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def clip_to_physical_limits(df: pd.DataFrame, column: str, start_idx: int, end_idx: int):
    """Clamps an injected window back within hard physical limits, if the column has any."""
    if column in HARD_PHYSICAL_LIMITS:
        low, high = HARD_PHYSICAL_LIMITS[column]
        df.loc[start_idx:end_idx, column] = df.loc[start_idx:end_idx, column].clip(low, high)


def spans_overlap(a_start, a_end, b_start, b_end):
    """True if interval [a_start, a_end] intersects [b_start, b_end]."""
    return not (a_end < b_start or a_start > b_end)


def has_overlap(claimed_spans, cols, start, end):
    """True if a proposed event intersects an existing fault event ON
    ANY OF THE SAME COLUMNS.

    FIX 1: `claimed_spans` is now {column: [(start, end), ...]}, checked
    per column, instead of one global list checked regardless of which
    parameter was touched. Ground truth has one row-level `fault_type`
    field PER COLUMN'S worth of conflict potential -- two single-column
    faults on DIFFERENT parameters at the same timestamp (e.g. a frozen
    pressure sensor while temperature spikes normally) are independent,
    real, and exactly the kind of event §1 of the module docstring
    already claimed was supported. A multivariate fault (which touches
    all three columns) still reserves all three, so it can't silently
    collide with a single-column fault on any of them.
    """
    return any(
        spans_overlap(start, end, claimed_start, claimed_end)
        for col in cols
        for claimed_start, claimed_end in claimed_spans.get(col, [])
    )


def inject_spike(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Push one reading far from the clean distribution without clipping it.

    A clipped candidate (for example humidity ``100 -> 100``) is not a
    spike at all. Returning ``None`` lets the placement loop choose a
    different time/parameter instead of writing an invalid ground-truth
    label.
    """
    mean, std, upper, lower = compute_bounds(df[column])
    # Push well beyond the clean-data tail (3.5-5.0 sigma) while staying within physical bounds
    magnitude = rng.uniform(3.5, 5.0)
    low, high = HARD_PHYSICAL_LIMITS.get(column, (-np.inf, np.inf))
    candidates = [
        mean + magnitude * std,
        mean - magnitude * std,
    ]
    viable = [value for value in candidates if low <= value <= high]
    if not viable:
        return None
    value = float(rng.choice(viable))
    # An injected spike must be visibly distinct from both of its
    # neighbours. The future neighbour is checked by the detector; this
    # guard prevents an already-flat source point from being mislabeled.
    if abs(value - float(df.loc[idx, column])) < max(std * 2.5, 0.5):
        return None
    df.loc[idx, column] = value
    return "spike"


def inject_spike_decay(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    FIX 3: second spike SUBTYPE (same fault_type="spike", still six
    ground-truth types total) representing a real voltage transient /
    ESD event rather than a discrete bit-flip: an instantaneous jump
    followed by an exponential decay back toward the natural signal,
    reflecting RC input-filter and thermal-mass dissipation kinetics --
    T(t) = T_natural(t) + delta_T0 * exp(-(t - t0) / tau).

    inject_spike (the bit-flip/transmission-garble variant, single
    reading, instant full reversion) is kept as-is and NOT replaced --
    real AWS networks see both signatures, so both stay in the pool
    (see FAULT_WEIGHTS) rather than one replacing the other.
    """
    mean, std, upper, lower = compute_bounds(df[column])
    low, high = HARD_PHYSICAL_LIMITS.get(column, (-np.inf, np.inf))

    tail_len = int(rng.integers(2, 5))
    end_idx = min(idx + tail_len, len(df) - 1)
    n_steps = end_idx - idx + 1

    magnitude = rng.uniform(3.0, 5.0)
    tau = rng.uniform(0.5, 1.5)  # decay time constant, in readings
    sign = float(rng.choice([-1.0, 1.0]))
    delta0 = sign * magnitude * std

    baseline = df.loc[idx:end_idx, column].to_numpy(dtype=float)
    t = np.arange(n_steps)
    decayed = baseline + delta0 * np.exp(-t / tau)

    peak_value = decayed[0]
    if not (low <= peak_value <= high):
        return None  # would clip to nothing distinctive -- try elsewhere
    if abs(peak_value - baseline[0]) < max(std * 4.0, 0.5):
        return None  # not distinct enough from the natural reading

    df.loc[idx:end_idx, column] = decayed
    clip_to_physical_limits(df, column, idx, end_idx)
    return "spike", idx, end_idx


def inject_frozen(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Hold the value at idx roughly constant for the next few rows -- a
    comms/sensor fault where the station stops tracking real atmospheric
    variation. Day 42/43: near-zero variance in a rolling window is
    itself a strong outlier signal.

    FIX 2: bit-exact repeat replaced with a bounded stationary random
    walk (x_t = x_{t-1} + N(0, ADC_NOISE_FLOOR_STD^2), clamped to stay
    within FROZEN_MAX_DEVIATION of the freeze point). A genuinely stuck
    sensor still has real ADC thermal noise on top of its static
    physical signal -- a plain equality check on real field data would
    never fire. The clamp keeps this a stationary process (variance
    near zero, per Day 42/43) rather than letting the walk accidentally
    accumulate into something that looks like drift instead.
    """
    freeze_length = rng.integers(6, 9)
    end_idx = min(idx + freeze_length, len(df) - 1)
    n_steps = end_idx - idx + 1

    anchor = float(df.loc[idx, column])
    noise_std = ADC_NOISE_FLOOR_STD[column]
    max_dev = FROZEN_MAX_DEVIATION[column]

    walk = np.cumsum(rng.normal(0, noise_std, n_steps))
    walk = np.clip(walk, -max_dev, max_dev)  # stays stationary, doesn't drift away
    walk[0] = 0.0  # first frozen row anchors exactly at the transition value

    df.loc[idx:end_idx, column] = anchor + walk
    clip_to_physical_limits(df, column, idx, end_idx)
    return "frozen_value", idx, end_idx


def inject_drift(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Calibration drift -- a slow, growing offset starting at idx and
    continuing to the end of the window. Unlike a spike, no single
    point looks extreme; only the trend over time reveals it.

    FIX 4: the offset is now SUPERIMPOSED on the station's own real,
    already-recorded readings for this span (T_natural(t) + delta(t)),
    not written over them starting from a single anchor value. The old
    `start_value + direction * ramp` approach discarded the real
    diurnal cycle (and any genuine weather movement) for the entire
    fault window and replaced it with a synthetic curve anchored at one
    point -- exactly the "suppresses natural nighttime cooling / daytime
    heating" failure mode. Adding the ramp on top keeps the real
    underlying signal intact; the fault is genuinely just the
    accumulating calibration offset, which is what a real drifting
    sensor actually looks like superimposed on true weather.

    The generated offset is direction-consistent for its whole labeled
    span. This is essential: Draft 2's CUSUM detector is expressly
    designed for accumulating one-directional bias, so an offset that
    reversed direction mid-span would make its ground truth invalid.
    Curvature is allowed, but the OFFSET itself (not the resulting raw
    value, which still moves with real weather on top of it) moves
    monotonically in one direction.
    """
    drift_length = rng.integers(20, 50)
    end_idx = min(idx + drift_length, len(df) - 1)
    direction = rng.choice([-1.0, 1.0])
    # Realistic calibration drift: 2.5 to 4.5 sigma offset
    param_std = min(float(df[column].std()), 3.0 if column == "temperature_c" else 5.0)
    max_offset = param_std * rng.uniform(2.5, 4.5)
    steps = end_idx - idx + 1

    # Linear or curved, but strictly monotonic offset.
    if rng.choice([True, False]):
        ramp = np.linspace(0, max_offset, steps)
    else:
        ramp = max_offset * (np.linspace(0, 1, steps) ** rng.uniform(1.3, 2.0))
    # A tiny positive increment prevents equal adjacent offsets in a
    # shallow curved ramp, while keeping the fault physically smooth.
    min_step = max(df[column].std() * 1e-4, 1e-6)
    ramp = np.maximum.accumulate(ramp + np.arange(steps) * min_step)

    natural_values = df.loc[idx:end_idx, column].to_numpy(dtype=float)
    df.loc[idx:end_idx, column] = natural_values + direction * ramp
    clip_to_physical_limits(df, column, idx, end_idx)
    return "drift", idx, end_idx


def inject_dropout(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Communication failure -- reading goes missing entirely."""
    df.loc[idx, column] = np.nan
    return "dropout"


def inject_fail_low(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Hardware fail-low -- distinct from a spike. Real sensors fail this
    way when a ground short, severed cable, or fully detached element
    pulls the analog input pin straight to 0 ADC counts, mapping to the
    scale's electrical floor -- not to some plausible-sounding low
    weather value. This is a flatline at the hardware RAIL, held for a
    short window -- different from `frozen` (which freezes at whatever
    the last real reading happened to be) and different from `spike`
    (a brief statistical extreme in either direction).

    FIX 5: floors changed from arbitrary intermediate sentinels
    (e.g. -15.0C, 50.0 hPa -- physically plausible-ish, cold-snap-ish
    values that undersell how a real ground short actually reads) to
    FAIL_LOW_RAIL_VALUE's true 0-ADC-count rail floors. Noise is now a
    small FIXED std (FAIL_LOW_NOISE_STD), not proportional to the floor
    magnitude -- a proportional formula divides to ~zero jitter at a
    0.0 hPa / 0.0% rail, which would silently reintroduce a bit-exact
    repeat exactly like the frozen-value bug this project already fixed
    once. clip_to_physical_limits is deliberately NOT called here (see
    HARD_PHYSICAL_LIMITS' module-level note) -- these values are
    supposed to sit outside any plausible atmospheric range.

    Window length is FIXED (FAIL_LOW_LENGTH), not randomized -- §5b's
    detector rule reclassifies fail-low at 2-3 consecutive hours, so a
    fixed 3-row window guarantees every injected event actually clears
    that bar, with no per-event variance to account for at eval time.
    """
    end_idx = min(idx + FAIL_LOW_LENGTH, len(df) - 1)
    n_steps = end_idx - idx + 1

    rail_value = FAIL_LOW_RAIL_VALUE[column]
    noise = rng.normal(0, FAIL_LOW_NOISE_STD, n_steps)
    df.loc[idx:end_idx, column] = rail_value + noise
    return "sensor_fail_low", idx, end_idx


def inject_multivariate(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Multivariate inconsistency -- the PS's own example scenario: a
    station reports a temperature spike while pressure/humidity move
    in directions that don't physically make sense together. `column`
    is ignored here since this fault always touches all three
    parameters at once -- it's the multivariate case the single-column
    fault functions can't represent.

    FIX 6: grounded in the Tetens/Clausius-Clapeyron saturation vapor
    pressure curve (see saturation_vapor_pressure_kpa) instead of two
    independently-chosen sigma magnitudes for temp and humidity. If
    ambient temperature rises with no real moisture transport, actual
    vapor pressure e = (RH/100)*es(T) stays constant, so relative
    humidity MUST fall as es(T) rises -- that's the real physically-
    consistent behavior, computed below as `rh_physically_consistent`.
    The FAULT is that the sensor instead reports humidity moving IN
    THE SAME DIRECTION as temperature (rising, not falling) -- the
    genuinely unphysical signature a real weather event (which follows
    the curve) would never produce. This makes the injected magnitude
    a real, checkable violation of the physics (a large, calibrated gap
    from rh_physically_consistent) rather than an arbitrary "+2 to
    3.5 sigma" guess, while pressure barely moving is still the third
    leg of the signature (a real event this size usually shows up in
    pressure too).
    """
    window = rng.integers(2, 5)
    end_idx = min(idx + window, len(df) - 1)
    n_steps = end_idx - idx + 1

    temp_std = min(float(df["temperature_c"].std()), 3.0)
    pressure_std = min(float(df["pressure_hpa"].std()), 4.0)

    temp_before = df.loc[idx:end_idx, "temperature_c"].to_numpy(dtype=float)
    rh_before = df.loc[idx:end_idx, "humidity_pct"].to_numpy(dtype=float)

    temp_delta = rng.uniform(2.2, 3.5) * temp_std
    temp_after = np.clip(temp_before + temp_delta, -5.0, 48.0)

    # What a REAL heat event would do to humidity, holding actual vapor
    # pressure constant (the physically-consistent case this fault must
    # violate, not reproduce).
    es_before = saturation_vapor_pressure_kpa(temp_before)
    es_after = saturation_vapor_pressure_kpa(temp_after)
    rh_physically_consistent = np.clip(rh_before * (es_before / es_after), 0.0, 100.0)

    # The FAULT: humidity instead rises, moving away from (not toward)
    # the physically-consistent value -- same-direction-as-temperature,
    # which real vapor-pressure physics forbids without real moisture
    # advection. Magnitude is anchored to how far the physically-honest
    # value would have dropped, so the violation is calibrated to this
    # specific temperature delta rather than a flat sigma guess.
    physically_expected_drop = rh_before - rh_physically_consistent
    fault_rh_rise = np.maximum(physically_expected_drop, 0.5) * rng.uniform(1.5, 2.5)
    rh_after = rh_before + fault_rh_rise

    df.loc[idx:end_idx, "temperature_c"] = temp_after
    df.loc[idx:end_idx, "humidity_pct"] = rh_after
    # Pressure barely moves, when a real weather event of this
    # magnitude would typically show a pressure change too.
    df.loc[idx:end_idx, "pressure_hpa"] += rng.normal(0, pressure_std * 0.1, n_steps)

    clip_to_physical_limits(df, "temperature_c", idx, end_idx)
    clip_to_physical_limits(df, "humidity_pct", idx, end_idx)
    clip_to_physical_limits(df, "pressure_hpa", idx, end_idx)

    return "multivariate_inconsistency", idx, end_idx


def inject_unstructured_anomaly(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Unstructured / miscellaneous anomaly -- specifically designed to test
    UNSUPERVISED model-only detection performance.

    Real-world field sensors occasionally suffer complex chaotic faults (e.g.
    erratic analog preamplifier oscillation, power supply ripple, or partial
    bridge degradation) that DO NOT fit any simple 1D pattern:
      - NOT a spike: remains strictly within normal 2.0-sigma bounds and physical limits.
      - NOT frozen: continually fluctuates with natural variance.
      - NOT fail-low: nowhere near the electrical 0 rail.
      - NOT a drift: fluctuates chaotically with 0 cumulative directional ramp.
      - NOT a simple 1D bound violation.

    Instead, it induces high-dimensional covariance breakdown:
    perturbs temperature, pressure, and humidity simultaneously with
    decorrelated high-frequency fluctuations. In feature space, this
    fractures the joint probability density P(T, P, RH, ROC), testing
    whether our unsupervised Isolation Forest model can independently
    detect the fault without any heuristic rule triggering.
    """
    window = rng.integers(3, 8)
    end_idx = min(idx + window, len(df) - 1)
    n_steps = end_idx - idx + 1

    t_std = min(float(df["temperature_c"].std()), 3.0)
    p_std = min(float(df["pressure_hpa"].std()), 4.0)
    h_std = min(float(df["humidity_pct"].std()), 8.0)

    # Rapid alternating perturbations:
    sign_t = rng.choice([-1.0, 1.0], size=n_steps)
    sign_p = -sign_t  # counter-correlated to break barometric pressure-temperature relation
    sign_h = rng.choice([-1.0, 1.0], size=n_steps)

    # Moderate magnitude: 1.8 to 2.4 sigma (well below 3-sigma spike threshold, within physical limits)
    t_pert = sign_t * rng.uniform(1.8, 2.4, size=n_steps) * t_std
    p_pert = sign_p * rng.uniform(1.6, 2.2, size=n_steps) * p_std
    h_pert = sign_h * rng.uniform(1.8, 2.4, size=n_steps) * h_std

    df.loc[idx:end_idx, "temperature_c"] += t_pert
    df.loc[idx:end_idx, "pressure_hpa"] += p_pert
    df.loc[idx:end_idx, "humidity_pct"] += h_pert

    # Unstructured anomalies must strictly remain within realistic meteorological bounds
    # so deterministic physical_bounds rules (temp in [-10, 55], pres in [850, 1080], hum in [0, 100])
    # do NOT fire on them -- testing purely unsupervised ML covariance breakdown:
    df.loc[idx:end_idx, "temperature_c"] = df.loc[idx:end_idx, "temperature_c"].clip(5.0, 45.0)
    df.loc[idx:end_idx, "pressure_hpa"] = df.loc[idx:end_idx, "pressure_hpa"].clip(920.0, 1040.0)
    df.loc[idx:end_idx, "humidity_pct"] = df.loc[idx:end_idx, "humidity_pct"].clip(15.0, 95.0)

    return "unstructured_anomaly", idx, end_idx


# =============================================================================
# OPERATIONAL BENCHMARK V2 FAULT INJECTORS (Benchmark_O_OPERATIONAL_v1)
# =============================================================================

def inject_drift_v2(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Operational Drift V2: Physical Sensor Bias Process.
    x_fault(t) = x_clean(t) + sign * b_mature * ((t-t0)/L)^gamma + sensor_noise
    """
    drift_length = int(rng.integers(18, 49))  # 18 to 48 hours
    end_idx = min(idx + drift_length, len(df) - 1)
    steps = end_idx - idx + 1
    direction = float(rng.choice([-1.0, 1.0]))

    clean_series = df[column].dropna()
    param_mad = float(np.median(np.abs(clean_series - clean_series.median())))
    if param_mad <= 0:
        param_mad = float(clean_series.std()) or 1.0

    b_mature = param_mad * float(rng.uniform(3.0, 4.8))
    if column == "temperature_c":
        b_mature = max(3.5, min(8.0, b_mature))
    elif column == "pressure_hpa":
        b_mature = max(3.5, min(12.0, b_mature))
    elif column == "humidity_pct":
        b_mature = max(18.0, min(35.0, b_mature))

    gamma = float(rng.uniform(1.0, 1.2))
    t_norm = np.linspace(0.0, 1.0, steps)
    ramp = b_mature * (t_norm ** gamma)

    noise_std = 0.02
    noise = rng.normal(0, noise_std, steps)

    natural_values = df.loc[idx:end_idx, column].to_numpy(dtype=float)
    df.loc[idx:end_idx, column] = natural_values + direction * ramp + noise
    clip_to_physical_limits(df, column, idx, end_idx)
    return "drift", idx, end_idx


def inject_frozen_v2(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Operational Frozen V2: Transducer Stagnation / Atmospheric Divergence.
    Sensor output remains anchored around x(t0) with small ADC noise,
    while underlying atmospheric weather continues normally.
    """
    freeze_length = int(rng.integers(8, 25))  # 8 to 24 hours
    end_idx = min(idx + freeze_length, len(df) - 1)
    n_steps = end_idx - idx + 1

    anchor = float(df.loc[idx, column])
    noise_std = 0.01
    max_dev = 0.05

    walk = np.cumsum(rng.normal(0, noise_std, n_steps))
    walk = np.clip(walk, -max_dev, max_dev)
    walk[0] = 0.0

    df.loc[idx:end_idx, column] = anchor + walk
    clip_to_physical_limits(df, column, idx, end_idx)
    return "frozen_value", idx, end_idx


def inject_spike_v2(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Operational Spike V2: Discrete impulse jump / transient.
    Single-reading impulse jump or discrete 1-reading spike.
    """
    clean_series = df[column].dropna()
    diff_1h = clean_series.diff().dropna()
    diff_mad = float(np.median(np.abs(diff_1h - diff_1h.median())))
    if diff_mad <= 0:
        diff_mad = float(clean_series.std() * 0.5) or 1.0

    low, high = HARD_PHYSICAL_LIMITS.get(column, (-np.inf, np.inf))
    sign = float(rng.choice([-1.0, 1.0]))
    magnitude = float(rng.uniform(4.5, 7.0)) * diff_mad
    if column == "temperature_c":
        magnitude = max(5.0, magnitude)
    elif column == "pressure_hpa":
        magnitude = max(6.5, magnitude)
    elif column == "humidity_pct":
        magnitude = max(20.0, magnitude)

    val_orig = float(df.loc[idx, column])
    val_cand = val_orig + sign * magnitude
    if not (low <= val_cand <= high):
        val_cand = val_orig - sign * magnitude
        if not (low <= val_cand <= high):
            return None
    df.loc[idx, column] = val_cand
    return "spike", idx, idx


def inject_dropout_v2(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Operational Dropout V2: Communication / Power Loss.
    40% isolated single NaN, 40% short burst loss [2, 6] h, 20% intermittent loss.
    """
    pattern = rng.choice(["single", "burst", "intermittent"], p=[0.4, 0.4, 0.2])
    if pattern == "single":
        df.loc[idx, column] = np.nan
        return "dropout", idx, idx
    elif pattern == "burst":
        burst_len = int(rng.integers(2, 7))  # 2 to 6 hours
        end_idx = min(idx + burst_len - 1, len(df) - 1)
        df.loc[idx:end_idx, column] = np.nan
        return "dropout", idx, end_idx
    else:  # intermittent
        window_len = 6
        end_idx = min(idx + window_len - 1, len(df) - 1)
        mask = rng.choice([True, False], size=(end_idx - idx + 1), p=[0.6, 0.4])
        mask[0] = True  # anchor start
        df_slice = df.loc[idx:end_idx, column].copy()
        df_slice.iloc[mask] = np.nan
        df.loc[idx:end_idx, column] = df_slice
        return "dropout", idx, end_idx


def inject_fail_low_v2(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Operational Fail-Low V2: Ground Short / Rail Disconnect.
    Duration L in [3, 12] consecutive hours.
    """
    fail_len = int(rng.integers(3, 13))  # 3 to 12 hours
    end_idx = min(idx + fail_len - 1, len(df) - 1)
    n_steps = end_idx - idx + 1

    rail_val = FAIL_LOW_RAIL_VALUE.get(column, -40.0)
    noise = rng.normal(0, FAIL_LOW_NOISE_STD, n_steps)
    df.loc[idx:end_idx, column] = rail_val + noise
    return "sensor_fail_low", idx, end_idx


def inject_multivariate_v2(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Operational Multivariate V2: Psychrometric / Thermodynamic Dislocation.
    Simultaneous positive T jump and positive RH rise violating Clausius-Clapeyron.
    Duration L in [3, 8] hours.
    """
    window = int(rng.integers(3, 9))  # 3 to 8 hours
    end_idx = min(idx + window - 1, len(df) - 1)
    n_steps = end_idx - idx + 1

    temp_before = df.loc[idx:end_idx, "temperature_c"].to_numpy(dtype=float)
    rh_before = df.loc[idx:end_idx, "humidity_pct"].to_numpy(dtype=float)

    delta_t = float(rng.uniform(3.5, 5.5))
    delta_rh = float(rng.uniform(15.0, 30.0))

    df.loc[idx:end_idx, "temperature_c"] = temp_before + delta_t
    df.loc[idx:end_idx, "humidity_pct"] = rh_before + delta_rh
    df.loc[idx:end_idx, "pressure_hpa"] += rng.normal(0, 0.2, n_steps)

    clip_to_physical_limits(df, "temperature_c", idx, end_idx)
    clip_to_physical_limits(df, "humidity_pct", idx, end_idx)
    clip_to_physical_limits(df, "pressure_hpa", idx, end_idx)
    return "multivariate_inconsistency", idx, end_idx


def inject_unstructured_v2(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Operational Unstructured V2: Heteroscedastic Sensor Noise Corruption.
    Duration L in [4, 12] hours.
    """
    window = int(rng.integers(4, 13))  # 4 to 12 hours
    end_idx = min(idx + window - 1, len(df) - 1)
    n_steps = end_idx - idx + 1

    t_noise = rng.normal(0, float(rng.uniform(3.0, 5.0)) * 0.8, size=n_steps)
    p_noise = rng.normal(0, float(rng.uniform(3.0, 5.0)) * 0.6, size=n_steps)
    h_noise = rng.normal(0, float(rng.uniform(3.0, 5.0)) * 4.0, size=n_steps)

    df.loc[idx:end_idx, "temperature_c"] += t_noise
    df.loc[idx:end_idx, "pressure_hpa"] += p_noise
    df.loc[idx:end_idx, "humidity_pct"] += h_noise

    df.loc[idx:end_idx, "temperature_c"] = df.loc[idx:end_idx, "temperature_c"].clip(5.0, 45.0)
    df.loc[idx:end_idx, "pressure_hpa"] = df.loc[idx:end_idx, "pressure_hpa"].clip(920.0, 1040.0)
    df.loc[idx:end_idx, "humidity_pct"] = df.loc[idx:end_idx, "humidity_pct"].clip(15.0, 95.0)
    return "unstructured_anomaly", idx, end_idx
    return "unstructured_anomaly", idx, end_idx


def inject_drift_v3(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Observable Operational Drift V3: Physical Sensor Bias Process with Observable Onset.
    x_fault(t) = x_clean(t) + sign * [b0 + (b_mature - b0) * ((t-t0)/L)^gamma] + sensor_noise
    Guarantees initial SNR >= 1.8x over ambient clean MAD from onset t0.
    """
    drift_length = int(rng.integers(18, 49))  # 18 to 48 hours
    end_idx = min(idx + drift_length, len(df) - 1)
    steps = end_idx - idx + 1
    direction = float(rng.choice([-1.0, 1.0]))

    clean_series = df[column].dropna()
    param_mad = float(np.median(np.abs(clean_series - clean_series.median())))
    if param_mad <= 0:
        param_mad = float(clean_series.std()) or 1.0

    if column == "temperature_c":
        b0 = float(rng.uniform(1.6, 2.4))
        b_mature = max(b0 + 1.5, float(rng.uniform(4.0, 7.5)))
    elif column == "pressure_hpa":
        b0 = float(rng.uniform(2.5, 4.0))
        b_mature = max(b0 + 2.0, float(rng.uniform(5.5, 10.0)))
    elif column == "humidity_pct":
        b0 = float(rng.uniform(12.0, 18.0))
        b_mature = max(b0 + 8.0, float(rng.uniform(22.0, 35.0)))
    else:
        b0 = param_mad * 1.8
        b_mature = param_mad * 4.0

    gamma = float(rng.uniform(1.0, 1.2))
    t_norm = np.linspace(0.0, 1.0, steps)
    ramp = b0 + (b_mature - b0) * (t_norm ** gamma)

    noise_std = 0.02
    noise = rng.normal(0, noise_std, steps)

    natural_values = df.loc[idx:end_idx, column].to_numpy(dtype=float)
    df.loc[idx:end_idx, column] = natural_values + direction * ramp + noise
    clip_to_physical_limits(df, column, idx, end_idx)
    return "drift", idx, end_idx


def inject_frozen_v3(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Observable Operational Frozen V3: Transducer Stagnation / Atmospheric Divergence.
    """
    freeze_length = int(rng.integers(8, 25))  # 8 to 24 hours
    end_idx = min(idx + freeze_length, len(df) - 1)
    n_steps = end_idx - idx + 1

    anchor = float(df.loc[idx, column])
    noise_std = 0.01
    max_dev = 0.04

    walk = np.cumsum(rng.normal(0, noise_std, n_steps))
    walk = np.clip(walk, -max_dev, max_dev)
    walk[0] = 0.0

    df.loc[idx:end_idx, column] = anchor + walk
    clip_to_physical_limits(df, column, idx, end_idx)
    return "frozen_value", idx, end_idx


def inject_spike_v3(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Observable Operational Spike V3: High-SNR Impulse jump.
    """
    clean_series = df[column].dropna()
    diff_1h = clean_series.diff().dropna()
    diff_mad = float(np.median(np.abs(diff_1h - diff_1h.median())))
    if diff_mad <= 0:
        diff_mad = float(clean_series.std() * 0.5) or 1.0

    low, high = HARD_PHYSICAL_LIMITS.get(column, (-np.inf, np.inf))
    sign = float(rng.choice([-1.0, 1.0]))
    magnitude = float(rng.uniform(5.0, 7.5)) * diff_mad
    if column == "temperature_c":
        magnitude = max(5.5, magnitude)
    elif column == "pressure_hpa":
        magnitude = max(7.0, magnitude)
    elif column == "humidity_pct":
        magnitude = max(22.0, magnitude)

    val_orig = float(df.loc[idx, column])
    val_cand = val_orig + sign * magnitude
    if not (low <= val_cand <= high):
        val_cand = val_orig - sign * magnitude
        if not (low <= val_cand <= high):
            return None
    df.loc[idx, column] = val_cand
    return "spike", idx, idx


def inject_dropout_v3(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Observable Operational Dropout V3.
    """
    return inject_dropout_v2(df, idx, column, rng)


def inject_fail_low_v3(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Observable Operational Fail-Low V3.
    """
    return inject_fail_low_v2(df, idx, column, rng)


def inject_multivariate_v3(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Observable Operational Multivariate V3: Psychrometric / Thermodynamic Dislocation.
    """
    window = int(rng.integers(3, 9))  # 3 to 8 hours
    end_idx = min(idx + window - 1, len(df) - 1)
    n_steps = end_idx - idx + 1

    temp_before = df.loc[idx:end_idx, "temperature_c"].to_numpy(dtype=float)
    rh_before = df.loc[idx:end_idx, "humidity_pct"].to_numpy(dtype=float)

    delta_t = float(rng.uniform(4.0, 6.0))
    delta_rh = float(rng.uniform(18.0, 32.0))

    df.loc[idx:end_idx, "temperature_c"] = temp_before + delta_t
    df.loc[idx:end_idx, "humidity_pct"] = rh_before + delta_rh
    df.loc[idx:end_idx, "pressure_hpa"] += rng.normal(0, 0.2, n_steps)

    clip_to_physical_limits(df, "temperature_c", idx, end_idx)
    clip_to_physical_limits(df, "humidity_pct", idx, end_idx)
    clip_to_physical_limits(df, "pressure_hpa", idx, end_idx)
    return "multivariate_inconsistency", idx, end_idx


def inject_unstructured_v3(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Observable Operational Unstructured V3."""
    return inject_unstructured_v2(df, idx, column, rng)


def inject_drift_v4(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Tiered Observable Operational Drift V4: Physical Sensor Bias Process with Strong Observable Onset.
    x_fault(t) = x_clean(t) + sign * [b0 + (b_mature - b0) * ((t-t0)/L)^gamma] + sensor_noise
    Guarantees initial SNR >= 2.2x over ambient clean MAD from onset t0 across all channels.
    """
    drift_length = int(rng.integers(24, 49))  # 24 to 48 hours
    end_idx = min(idx + drift_length, len(df) - 1)
    steps = end_idx - idx + 1
    direction = float(rng.choice([-1.0, 1.0]))

    clean_series = df[column].dropna()
    param_mad = float(np.median(np.abs(clean_series - clean_series.median())))
    if param_mad <= 0:
        param_mad = float(clean_series.std()) or 1.0

    if column == "temperature_c":
        b0 = float(rng.uniform(2.0, 3.0))
        b_mature = max(b0 + 2.0, float(rng.uniform(5.0, 8.5)))
    elif column == "pressure_hpa":
        b0 = float(rng.uniform(3.0, 5.0))
        b_mature = max(b0 + 3.0, float(rng.uniform(6.5, 12.0)))
    elif column == "humidity_pct":
        b0 = float(rng.uniform(15.0, 22.0))
        b_mature = max(b0 + 10.0, float(rng.uniform(25.0, 40.0)))
    else:
        b0 = param_mad * 2.2
        b_mature = param_mad * 4.5

    gamma = float(rng.uniform(1.0, 1.15))
    t_norm = np.linspace(0.0, 1.0, steps)
    ramp = b0 + (b_mature - b0) * (t_norm ** gamma)

    noise_std = 0.02
    noise = rng.normal(0, noise_std, steps)

    natural_values = df.loc[idx:end_idx, column].to_numpy(dtype=float)
    df.loc[idx:end_idx, column] = natural_values + direction * ramp + noise
    clip_to_physical_limits(df, column, idx, end_idx)
    return "drift", idx, end_idx


def inject_frozen_v4(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Tiered Observable Frozen V4."""
    return inject_frozen_v3(df, idx, column, rng)


def inject_spike_v4(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Tiered Observable Spike V4."""
    return inject_spike_v3(df, idx, column, rng)


def inject_dropout_v4(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Tiered Observable Dropout V4."""
    return inject_dropout_v2(df, idx, column, rng)


def inject_fail_low_v4(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Tiered Observable Fail-Low V4."""
    return inject_fail_low_v2(df, idx, column, rng)


def inject_multivariate_v4(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Tiered Observable Multivariate V4: Clear Psychrometric / Thermodynamic Dislocation.
    """
    window = int(rng.integers(4, 10))  # 4 to 9 hours
    end_idx = min(idx + window - 1, len(df) - 1)
    n_steps = end_idx - idx + 1

    temp_before = df.loc[idx:end_idx, "temperature_c"].to_numpy(dtype=float)
    rh_before = df.loc[idx:end_idx, "humidity_pct"].to_numpy(dtype=float)

    delta_t = float(rng.uniform(4.5, 6.5))
    delta_rh = float(rng.uniform(22.0, 38.0))

    df.loc[idx:end_idx, "temperature_c"] = temp_before + delta_t
    df.loc[idx:end_idx, "humidity_pct"] = rh_before + delta_rh
    df.loc[idx:end_idx, "pressure_hpa"] += rng.normal(0, 0.2, n_steps)

    clip_to_physical_limits(df, "temperature_c", idx, end_idx)
    clip_to_physical_limits(df, "humidity_pct", idx, end_idx)
    clip_to_physical_limits(df, "pressure_hpa", idx, end_idx)
    return "multivariate_inconsistency", idx, end_idx


def inject_unstructured_v4(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Tiered Observable Unstructured V4."""
    return inject_unstructured_v2(df, idx, column, rng)


# Upper bound on window length per fault type, used to pre-check
# overlap BEFORE mutating df -- must stay in sync with each function's
# own rng.integers(...) upper bound (exclusive), or its fixed length.
FAULT_MAX_LEN = {
    inject_spike: 1,
    inject_frozen: 8,
    inject_drift: 49,
    inject_dropout: 1,
    inject_multivariate: 4,
    inject_fail_low: FAIL_LOW_LENGTH,
    inject_unstructured_anomaly: 8,
    # V2 mappings
    inject_spike_v2: 1,
    inject_frozen_v2: 24,
    inject_drift_v2: 48,
    inject_dropout_v2: 6,
    inject_multivariate_v2: 8,
    inject_fail_low_v2: 12,
    inject_unstructured_v2: 12,
    # V3 mappings
    inject_spike_v3: 1,
    inject_frozen_v3: 24,
    inject_drift_v3: 48,
    inject_dropout_v3: 6,
    inject_multivariate_v3: 8,
    inject_fail_low_v3: 12,
    inject_unstructured_v3: 12,
    # V4 mappings
    inject_spike_v4: 1,
    inject_frozen_v4: 24,
    inject_drift_v4: 48,
    inject_dropout_v4: 6,
    inject_multivariate_v4: 9,
    inject_fail_low_v4: 12,
    inject_unstructured_v4: 12,
}

# Faults that touch all three parameters at once -- they must claim
# all three columns' spans, not just the sampled one.
MULTI_COLUMN_FAULTS = {
    inject_multivariate, inject_unstructured_anomaly,
    inject_multivariate_v2, inject_unstructured_v2,
    inject_multivariate_v3, inject_unstructured_v3,
    inject_multivariate_v4, inject_unstructured_v4
}

# Relative frequency weights for how often each fault type actually
# occurs on a real AWS network.
FAULT_WEIGHTS = {
    inject_spike: 1.8,
    inject_dropout: 3.0,
    inject_frozen: 2.0,
    inject_fail_low: 1.5,
    inject_drift: 1.0,
    inject_multivariate: 1.0,
    inject_unstructured_anomaly: 1.2,
}

FAULT_WEIGHTS_V2 = {
    inject_spike_v2: 1.8,
    inject_dropout_v2: 3.0,
    inject_frozen_v2: 2.0,
    inject_fail_low_v2: 1.5,
    inject_drift_v2: 1.0,
    inject_multivariate_v2: 1.0,
    inject_unstructured_v2: 1.2,
}

FAULT_WEIGHTS_V3 = {
    inject_spike_v3: 1.8,
    inject_dropout_v3: 3.0,
    inject_frozen_v3: 2.0,
    inject_fail_low_v3: 1.5,
    inject_drift_v3: 1.0,
    inject_multivariate_v3: 1.0,
    inject_unstructured_v3: 1.2,
}

FAULT_WEIGHTS_V4 = {
    inject_spike_v4: 1.8,
    inject_dropout_v4: 3.0,
    inject_frozen_v4: 2.0,
    inject_fail_low_v4: 1.5,
    inject_drift_v4: 1.0,
    inject_multivariate_v4: 1.0,
    inject_unstructured_v4: 1.2,
}

# Every fault type gets AT LEAST this many injected EVENTS, regardless
# of its weight above.
MIN_EVENTS_PER_TYPE = 4


def inject_anomalies(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    cluster_claimed_spans: list = None,
    return_spans: bool = False,
    regime: str = DEFAULT_REGIME,
) -> pd.DataFrame:
    """
    Walks through one station's dataframe and injects labeled faults
    at random locations across temperature/pressure/humidity columns.
    Returns a new dataframe with two extra columns: is_anomaly (bool)
    and fault_type (str or None) -- this is the ground truth label set.

    Supported regimes:
    - 'benchmark_b' / 'stress_v1' / 'pcl_compatible': Existing Benchmark B stress benchmark.
    - 'operational_v1' / 'benchmark_o': Observable operational benchmark V1.
    - 'observable_v1' / 'benchmark_o_observable': Fully observable operational benchmark V2.
    - 'observable_v2' / 'benchmark_o_tiered_observable_v2': Tiered observable operational benchmark V3.
    """
    rng = np.random.default_rng(seed)
    if ANOMALY_DENSITY_MULTIPLIER < 0:
        raise ValueError("ANOMALY_DENSITY_MULTIPLIER must be zero or positive")
    df = df.copy().reset_index(drop=True)
    df["is_anomaly"] = False
    df["fault_type"] = None

    columns = ["temperature_c", "pressure_hpa", "humidity_pct"]
    df[columns] = df[columns].astype(float)
    n_rows = len(df)

    target_anomalous_rows = int(n_rows * INJECTION_RATE * ANOMALY_DENSITY_MULTIPLIER)

    if regime in ("observable_v2", "benchmark_o_tiered_observable_v2"):
        fault_functions = [
            inject_spike_v4, inject_frozen_v4, inject_drift_v4,
            inject_dropout_v4, inject_multivariate_v4, inject_fail_low_v4,
            inject_unstructured_v4,
        ]
        weights_dict = FAULT_WEIGHTS_V4
    elif regime in ("observable_v1", "benchmark_o_observable"):
        fault_functions = [
            inject_spike_v3, inject_frozen_v3, inject_drift_v3,
            inject_dropout_v3, inject_multivariate_v3, inject_fail_low_v3,
            inject_unstructured_v3,
        ]
        weights_dict = FAULT_WEIGHTS_V3
    elif regime in ("operational_v1", "benchmark_o"):
        fault_functions = [
            inject_spike_v2, inject_frozen_v2, inject_drift_v2,
            inject_dropout_v2, inject_multivariate_v2, inject_fail_low_v2,
            inject_unstructured_v2,
        ]
        weights_dict = FAULT_WEIGHTS_V2
    else:
        fault_functions = [
            inject_spike, inject_frozen, inject_drift,
            inject_dropout, inject_multivariate, inject_fail_low,
            inject_unstructured_anomaly,
        ]
        weights_dict = FAULT_WEIGHTS

    # FIX 1: per-column timelines instead of one global list -- a
    # reading belongs to at most one injected fault event PER
    # PARAMETER, so independent faults on different parameters can
    # legitimately share a timestamp (see has_overlap's docstring).
    # inject_multivariate claims all three columns since it touches
    # all three at once.
    claimed_spans = {col: [] for col in columns}
    all_placed_spans = []
    fault_counts = {fn: 0 for fn in fault_functions}
    rows_injected = 0

    def try_inject(fault_fn, attempts_budget):
        nonlocal rows_injected
        attempts = 0
        while attempts < attempts_budget:
            attempts += 1
            idx = int(rng.integers(10, n_rows - 60))
            column = rng.choice(columns)
            cols_needed = list(columns) if fault_fn in MULTI_COLUMN_FAULTS else [column]

            # Conservative pre-check: reserve the fault type's MAX
            # possible window before running it, so we never have to
            # revert a mutation after the fact.
            candidate_end = min(idx + FAULT_MAX_LEN[fault_fn], n_rows - 1)
            if has_overlap(claimed_spans, cols_needed, idx, candidate_end):
                continue

            # Cluster-level concurrency check (Benchmark B PCL-Compatible regime):
            # Prohibit scheduling if another station in the same cluster is already active
            if cluster_claimed_spans is not None:
                if any(spans_overlap(idx, candidate_end, cs, ce) for cs, ce in cluster_claimed_spans):
                    continue

            result = fault_fn(df, idx, column, rng)

            # An injector may decline an invalid candidate (notably a
            # would-be spike that physical clipping would erase). It
            # has made no mutation, so safely keep searching.
            if result is None:
                continue

            if isinstance(result, tuple) and len(result) == 3:
                fault_type, start, end = result
            else:
                fault_type = result
                start = end = idx

            df.loc[start:end, "is_anomaly"] = True
            df.loc[start:end, "fault_type"] = fault_type
            for col in cols_needed:
                claimed_spans[col].append((start, end))
            all_placed_spans.append((start, end))
            if cluster_claimed_spans is not None:
                cluster_claimed_spans.append((start, end))
            rows_injected += (end - start + 1)
            fault_counts[fault_fn] += 1
            return True
        return False

    # Pass 1 -- guarantee the floor. Every fault type gets at least
    # MIN_EVENTS_PER_TYPE events before anything else happens, so a
    # rare-weighted or overlap-unlucky type can never end up at zero.
    scaled_min_events = (
        0 if ANOMALY_DENSITY_MULTIPLIER == 0
        else max(1, round(MIN_EVENTS_PER_TYPE * ANOMALY_DENSITY_MULTIPLIER))
    )
    for fault_fn in fault_functions:
        placed = 0
        while placed < scaled_min_events:
            if not try_inject(fault_fn, attempts_budget=n_rows):
                break  # genuinely no room left for this type -- move on
            placed += 1

    # Pass 2 -- fill the remaining row budget with weighted-random
    # draws across fault types.
    weight_fns = list(weights_dict.keys())
    weight_probs = np.array([weights_dict[fn] for fn in weight_fns])
    weight_probs = weight_probs / weight_probs.sum()

    max_total_attempts = n_rows * 3  # generous safety valve
    total_attempts = 0
    while rows_injected < target_anomalous_rows and total_attempts < max_total_attempts:
        total_attempts += 1
        fault_fn = weight_fns[rng.choice(len(weight_fns), p=weight_probs)]
        try_inject(fault_fn, attempts_budget=1)

    if return_spans:
        return df, all_placed_spans
    return df


def generate_network_benchmark(
    regime: str = DEFAULT_REGIME,
    seed: int = RANDOM_SEED,
    faulty_station_ids: set = None,
    data_dir: Path = DATA_DIR,
    output_dir: Path = DATA_DIR,
    save_to_disk: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Generates full multi-station benchmark dataset under specified regime:
    - 'benchmark_b' / 'stress_v1' / 'pcl_compatible': PCL-compatible operational stress benchmark (max 1 fault per cluster per timestamp).
    - 'operational_v1' / 'benchmark_o': PCL-compatible observable operational benchmark.
    - 'benchmark_a' / 'unrestricted': Unrestricted multi-fault stress test.
    """
    from collections import defaultdict
    station_files = sorted(
        p for p in data_dir.glob("AWS-*.csv") if "_labeled" not in p.name
    )
    if len(station_files) < 7:
        raise FileNotFoundError(f"Only {len(station_files)} station CSVs found in {data_dir}")

    # Map stations to clusters
    station_to_cluster = {}
    try:
        from config import CLUSTERS
        for cid, cinfo in CLUSTERS.items():
            all_s = [cinfo["center"]["station_id"]] + [n["station_id"] for n in cinfo["neighbors"]]
            for sid in all_s:
                station_to_cluster[sid] = cid
    except Exception:
        pass

    if faulty_station_ids is None:
        # All stations in the network can receive fault episodes across the year,
        # subject to the strict constraint: <= 1 active faulty station per cluster at any timestamp.
        faulty_station_ids = {p.stem for p in station_files}

    cluster_spans = defaultdict(list)
    results = {}

    is_cluster_constrained = regime in (
        "benchmark_b", "pcl_compatible", "operational_v1", "benchmark_o", "stress_v1",
        "observable_v1", "benchmark_o_observable",
        "observable_v2", "benchmark_o_tiered_observable_v2"
    )

    for csv_path in station_files:
        sid = csv_path.stem
        cid = station_to_cluster.get(sid, "UNKNOWN")
        df_clean = pd.read_csv(csv_path, parse_dates=["timestamp"])

        if sid in faulty_station_ids:
            station_seed = seed + station_files.index(csv_path) * 1009
            c_spans = cluster_spans[cid] if is_cluster_constrained else None
            injected, new_spans = inject_anomalies(
                df_clean,
                seed=station_seed,
                cluster_claimed_spans=c_spans,
                return_spans=True,
                regime=regime,
            )
            if is_cluster_constrained and new_spans:
                cluster_spans[cid].extend(new_spans)
            results[sid] = injected
        else:
            injected = df_clean.copy()
            injected["is_anomaly"] = False
            injected["fault_type"] = None
            results[sid] = injected

        if save_to_disk:
            out_file = output_dir / f"{sid}_labeled.csv"
            injected.to_csv(out_file, index=False)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SkyGuard AI Synthetic Anomaly Injector")
    parser.add_argument(
        "--regime",
        choices=["benchmark_a", "benchmark_b", "unrestricted", "pcl_compatible"],
        default=DEFAULT_REGIME,
        help="Benchmark regime: benchmark_a (unrestricted stress test) or benchmark_b (PCL-compatible max 1 fault/cluster)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    print(
        f"Generating network benchmark in regime '{args.regime}' with seed {args.seed}...\n"
        f"Injection density multiplier: {ANOMALY_DENSITY_MULTIPLIER:g} (base rate {INJECTION_RATE:.1%})\n"
    )
    generate_network_benchmark(regime=args.regime, seed=args.seed, save_to_disk=True)
    print(f"\nBenchmark generation complete -> Saved to {DATA_DIR}/*_labeled.csv")


if __name__ == "__main__":
    main()
