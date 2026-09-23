"""
SkyGuard AI — Phase 2b: Model training.

Trains the Isolation Forest ONCE, offline, on RAW real historical data
only -- NEVER on a *_labeled.csv. Feeding injected/corrupted data into
training would teach the model a wrong definition of "normal" (see
BACKEND_BLUEPRINT.md section 4, and the leakage warnings throughout
anomaly_injector.py / features.py). This script actively checks for
and refuses that mistake below, rather than just warning about it.

Unlocked by: Day 80 (trees: entropy/gini/information gain), 84
(ensemble learning), 88 (bagging), 91-93 (random forest, bias-variance,
bagging vs RF). Isolation Forest reuses all of this directly: it's a
forest of randomized trees, each isolating points via random
feature/threshold splits -- but instead of voting on a prediction
(what Random Forest's trees do), a point's anomaly score comes from
HOW FEW splits it took to isolate it. Genuinely anomalous points sit
far from the bulk of the data, so random splits separate them out
quickly (few splits = anomalous); normal points are surrounded by
similar points and take many splits to isolate alone. n_estimators=100
stabilizes this score across many random trees, the same bias-variance
reasoning as Day 92 for Random Forest.

PATCH LOG:
  - Added calibrate_rule_thresholds() call (features.py). detect.py and
    evaluate.py both now require artifact["rule_thresholds"] to exist --
    an artifact saved by the previous version of this file lacks that
    key entirely, and both will raise KeyError on load rather than
    silently running with no rule-layer thresholds. Retrain after
    pulling this patch.
  - SKYGUARD_ARCHITECTURE_DRAFT_2.md §9 SYNC (this patch): this file was
    still written against the PRE-§9 features.py contract in two places,
    both would have failed hard the first time this ran against the
    current features.py/config.py:
      1. build_feature_matrix(df, metadata) -- features.py's signature
         is single-arg now (spatial features removed, metadata was only
         ever used for cluster_id lookup for those). TypeError otherwise.
         Since nothing else in this file needed metadata, the
         stations_metadata.csv load is dropped entirely rather than kept
         around unused.
      2. The per-station threshold diagnostic read
         rule_thresholds["frozen"][prefix] -- calibrate_rule_thresholds()
         no longer produces a "frozen" key at all (§1: frozen is a
         deterministic floor-match now, calibrated thresholds only exist
         for "roc_small" and "spike"). KeyError otherwise. Diagnostic
         rewritten to print both real keys.
      3. contamination reverted 0.02 -> 'auto'. The comment directly
         above the IsolationForest(...) call argues, at length, for
         'auto' specifically BECAUSE we're fitting on data we've
         validated as clean -- asserting a fixed anomaly rate on data we
         just asserted is normal bakes in a contradiction. The code had
         drifted to a hardcoded 0.02 that contradicts its own comment.
         Restoring 'auto' to match the documented reasoning. If 0.02 was
         actually an intentional, separately-justified choice, say so and
         I'll put it back -- flagging this as a real behavior change, not
         a formatting fix.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.append(str(Path(__file__).parent.parent))
from model.features import build_feature_matrix, FEATURE_COLUMNS, calibrate_rule_thresholds

DATA_DIR = Path(__file__).parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).parent.parent / "model_artifacts"

N_ESTIMATORS = 100
RANDOM_STATE = 42


def load_clean_training_data():
    """
    Loads the RAW combined dataset. Hard-fails if the input looks like a
    labeled/injected file -- this is the training/serving separation
    rule enforced in code, not just a comment.

    NOTE (this patch): no longer loads stations_metadata.csv. It was
    only ever used to pass cluster_id through to build_feature_matrix
    for the now-removed spatial features (§9) -- nothing else in this
    file needs it, and build_feature_matrix no longer accepts it.
    """
    stations_path = DATA_DIR / "all_stations.csv"

    if not stations_path.exists():
        raise FileNotFoundError(
            f"Expected {stations_path.name} in {DATA_DIR} "
            f"(output of data_fetch.py -> validate_data.py). Run that first."
        )

    df = pd.read_csv(stations_path, parse_dates=["timestamp"])

    if "is_anomaly" in df.columns or "fault_type" in df.columns:
        raise ValueError(
            "all_stations.csv contains is_anomaly/fault_type columns -- this looks "
            "like a labeled/injected file, not the raw combined dataset. train.py "
            "must only ever see raw data. Re-run data_fetch.py's combined output, "
            "or check you haven't accidentally pointed this at a *_labeled.csv."
        )

    return df


def train():
    df = load_clean_training_data()

    # True Temporal Split (P0/Phase 3): Ensure we don't mix future and past data randomly.
    # Train only on the first 70% of the dataset chronologically.
    df = df.sort_values("timestamp")
    cutoff_idx = int(len(df) * 0.7)
    cutoff_date = df.iloc[cutoff_idx]["timestamp"]
    print(f"Temporal Split: Training on data before {cutoff_date}")
    df = df[df["timestamp"] < cutoff_date]

    print(f"Loaded {len(df)} raw rows across {df['station_id'].nunique()} stations.\n")

    # SIGNATURE CHANGE (§9, features.py): single-arg now -- metadata was
    # only ever used for spatial-feature cluster lookups, removed with
    # the rest of that machinery.
    featured = build_feature_matrix(df)

    # Drop warm-up rows (first ~48h per station) that don't have a
    # complete rolling baseline yet -- an incomplete feature vector
    # isn't a real training example, it's a startup artifact.
    before = len(featured)
    featured = featured.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    after = len(featured)
    print(f"Dropped {before - after} warm-up rows with incomplete features "
          f"(expected: roughly ROLLING_MIN_PERIODS=6h x {df['station_id'].nunique()} "
          f"stations -- NOT the full 48h window, since min_periods lets rolling "
          f"features start producing values after just 6h of history).")
    print(f"{after} rows remain for training.\n")

    X = featured[FEATURE_COLUMNS].values

    # contamination=0.01 (NOT 'auto'). 'auto' uses the original paper's
    # offset_=-0.5 heuristic which flags ~11% of training data as anomalous
    # even on a fully clean training set. Empirically: 4.4% of training rows
    # score model_pct > 72, meaning ~93 clean readings per station will
    # cross the fusion threshold when ANY rule (drift/spike) also fires.
    # That directly drives the 85-226 clean-station FPs we measured.
    # Setting contamination=0.01 makes the model's internal boundary
    # conservative: only the most extreme 1% of training rows are flagged,
    # substantially lowering model_pct on normal readings. Injected fault
    # events (extreme rail values, sustained drifts, T/RH violations) will
    # still score very high relative to the rescaled distribution.
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=0.01,
        random_state=RANDOM_STATE,
        # Single-process fitting keeps training reliable in restricted
        # Windows environments where joblib cannot create worker IPC
        # handles. It changes throughput only, not the model contract.
        n_jobs=1,
    )
    model.fit(X)

    # decision_function: higher = more normal, lower/negative = more
    # anomalous. Saving this training distribution alongside the model
    # gives detect.py/evaluate.py a reference point for "what did
    # normal actually look like," instead of calibrating blind.
    scores = model.decision_function(X)
    print("Training score distribution (decision_function; LOWER = more anomalous):")
    print(pd.Series(scores).describe())

    # Rule-layer thresholds (roc_small, spike -- frozen/drift are no
    # longer calibrated, see features.py's calibrate_rule_thresholds
    # docstring), calibrated empirically from this SAME real clean
    # `featured` data. Using the already-dropna'd `featured` (same rows
    # as X) is fine even though a handful of rows within the first
    # DRIFT_LOOKBACK_HOURS=24h per station may still be NaN in the
    # diagnostic-only Nh_delta column specifically (FEATURE_COLUMNS only
    # requires ROLLING_MIN_PERIODS=6h, shorter than the 24h delta needs)
    # -- pandas' .quantile() skips NaN by default, and it's a small
    # fraction of ~2000+ rows/station either way.
    rule_thresholds = calibrate_rule_thresholds(featured)

    # DIAGNOSTIC FIX (this patch): the old block read
    # rule_thresholds["frozen"][prefix] -- that key doesn't exist
    # anymore (§1: frozen is a deterministic floor-match, not a
    # calibrated threshold). The two real calibrated rule types are
    # "roc_small" and "spike" -- print both instead.
    print("\n--- PER-STATION THRESHOLD DIAGNOSTIC ---")
    for rule_key in ("roc_small", "spike"):
        for prefix in ["temp", "pressure", "humidity"]:
            entries = rule_thresholds[rule_key][prefix]
            n_stations = len([k for k in entries if k != "__global__"])
            print(f"{rule_key}[{prefix}]: {n_stations} stations calibrated individually, "
                  f"__global__={entries['__global__']:.4f}")
            print(f"  sample values: {dict(list(entries.items())[:5])}")
    print("--- END DIAGNOSTIC ---")

    print("\nCalibrated rule thresholds (from real clean data, see features.py's "
          "calibrate_rule_thresholds docstring for the percentiles used):")
    print(rule_thresholds)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "training_score_mean": float(scores.mean()),
        "training_score_std": float(scores.std()),
        "training_score_min": float(scores.min()),
        "training_score_max": float(scores.max()),
        "rule_thresholds": rule_thresholds,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "n_training_rows": after,
    }
    output_path = ARTIFACTS_DIR / "isolation_forest.pkl"
    joblib.dump(artifact, output_path)
    print(f"\nSaved trained model + metadata -> {output_path}")
    print("\nThis .pkl is what detect.py loads ONCE at API startup -- see "
          "BACKEND_BLUEPRINT.md section 4 for why it must never retrain per-request.")


if __name__ == "__main__":
    train()
