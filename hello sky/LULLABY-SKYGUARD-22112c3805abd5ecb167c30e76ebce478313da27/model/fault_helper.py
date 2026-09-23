"""Fault-oriented helper model trained on independent injector replays.

This module deliberately does not alter ``data/anomaly_injector.py``.  It
learns from new, sparse network replays and is evaluated on a separate replay.
All temporal signals are trailing/causal; frozen episodes may be confirmed
late, then their buffered onset is backfilled for incident reporting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from data.anomaly_injector import inject_anomalies
from model.features import build_feature_matrix

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_COLUMNS = ("temperature_c", "pressure_hpa", "humidity_pct")
PREFIXES = ("temp", "pressure", "humidity")
BASE_FEATURES = [
    "temperature_c", "pressure_hpa", "humidity_pct",
    "temp_deviation", "pressure_deviation", "humidity_deviation",
    "temp_roc_1h", "pressure_roc_1h", "humidity_roc_1h",
    "temp_roc_3h", "pressure_roc_3h", "humidity_roc_3h",
    "vapor_pressure_consistency_dev",
]


def feature_columns() -> list[str]:
    cols = list(BASE_FEATURES)
    for prefix in PREFIXES:
        cols.append(f"{prefix}_was_missing")
        cols.extend([
            f"{prefix}_range_6h", f"{prefix}_peer_range_6h",
            f"{prefix}_peer_residual", f"{prefix}_peer_residual_6h_change",
            f"{prefix}_activity_gap",
        ])
        for lag in range(1, 7):
            cols.extend([
                f"{prefix}_peer_residual_lag_{lag}",
                f"{prefix}_own_change_lag_{lag}",
                f"{prefix}_peer_change_lag_{lag}",
            ])
    return cols


def build_network_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Produce causal station and leave-one-out cluster features."""
    required = ["station_id", "timestamp", "is_anomaly", "fault_type"]
    if "__network" in frame:
        required.append("__network")
    required.extend([c for c in frame.columns if c.startswith("frozen_param_")])
    labels = frame[required].copy()
    raw = frame.drop(
        columns=["is_anomaly", "fault_type", *[c for c in frame.columns if c.startswith("frozen_param_")]],
        errors="ignore",
    )
    featured = build_feature_matrix(raw)
    
    featured["timestamp"] = pd.to_datetime(featured["timestamp"]).dt.tz_localize(None)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"]).dt.tz_localize(None)
    
    featured = featured.merge(labels, on=["station_id", "timestamp"], how="left")
    featured["is_anomaly"] = featured["is_anomaly"].fillna(False).astype(bool)
    featured["fault_type"] = featured["fault_type"].fillna("none")
    if "__network" not in featured:
        featured["__network"] = 0
    featured["__base_station"] = featured["station_id"].str.replace(r"__n\d+$", "", regex=True)
    meta = pd.read_csv(DATA_DIR / "stations_metadata.csv")[["station_id", "cluster_id"]]
    featured = featured.merge(meta, left_on="__base_station", right_on="station_id", how="left", suffixes=("", "_meta"))
    featured = featured.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    featured = featured.copy()

    for raw_col, prefix in zip(RAW_COLUMNS, PREFIXES):
        featured[f"{prefix}_was_missing"] = featured[raw_col].isna().astype(float)
        
        by_station = featured.groupby("station_id", group_keys=False)[raw_col]
        featured[f"{prefix}_range_6h"] = by_station.transform(
            lambda s: s.rolling(6, min_periods=4).max() - s.rolling(6, min_periods=4).min()
        )
        
        lookup = featured[["__network", "cluster_id", "timestamp", "station_id", raw_col]]
        merged = featured[["__network", "cluster_id", "timestamp", "station_id"]].merge(
            lookup, on=["__network", "cluster_id", "timestamp"], suffixes=("", "_peer")
        )
        merged = merged[merged["station_id"] != merged["station_id_peer"]]
        res = merged.groupby(["__network", "cluster_id", "timestamp", "station_id"])[raw_col].median()
        df_idx = featured.set_index(["__network", "cluster_id", "timestamp", "station_id"])
        featured[f"{prefix}_peer_value"] = res.reindex(df_idx.index).values
        
        residual = featured[raw_col] - featured[f"{prefix}_peer_value"]
        featured[f"{prefix}_peer_residual"] = residual
        featured[f"{prefix}_peer_residual_6h_change"] = residual.groupby(featured["station_id"]).diff(6)
        peer_by_station = featured.groupby("station_id", group_keys=False)[f"{prefix}_peer_value"]
        featured[f"{prefix}_peer_range_6h"] = peer_by_station.transform(
            lambda s: s.rolling(6, min_periods=4).max() - s.rolling(6, min_periods=4).min()
        )
        featured[f"{prefix}_activity_gap"] = featured[f"{prefix}_peer_range_6h"] - featured[f"{prefix}_range_6h"]
        own_change = featured.groupby("station_id")[raw_col].diff()
        peer_change = featured.groupby("station_id")[f"{prefix}_peer_value"].diff()
        for lag in range(1, 7):
            featured[f"{prefix}_peer_residual_lag_{lag}"] = residual.groupby(featured["station_id"]).shift(lag)
            featured[f"{prefix}_own_change_lag_{lag}"] = own_change.groupby(featured["station_id"]).shift(lag - 1)
            featured[f"{prefix}_peer_change_lag_{lag}"] = peer_change.groupby(featured["station_id"]).shift(lag - 1)

    cols = feature_columns()
    featured[cols] = featured[cols].replace([np.inf, -np.inf], np.nan).clip(-1e10, 1e10)
    return featured


def make_sparse_training_replays(excluded_stations: set[str], seeds: list[int]) -> pd.DataFrame:
    """Generate fresh independent replays while leaving most peers clean."""
    ids = sorted(p.stem for p in DATA_DIR.glob("AWS-*.csv") if "_labeled" not in p.name and p.stem not in excluded_stations)
    networks = []
    for network_id, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        faulty = set(rng.choice(ids, size=min(3, len(ids)), replace=False))
        for station_id in ids:
            clean = pd.read_csv(DATA_DIR / f"{station_id}.csv", parse_dates=["timestamp"])
            raw = clean.copy()
            if station_id in faulty:
                raw = inject_anomalies(raw, seed=seed + 1009 * ids.index(station_id))
            else:
                raw["is_anomaly"] = False
                raw["fault_type"] = None
            # The injector's row label deliberately does not record which
            # channel was changed.  Recover that training-only attribution by
            # comparing its output to the untouched input; this does not alter
            # injection behavior or expose a label to serving.
            frozen_rows = raw["fault_type"].eq("frozen_value")
            for raw_col, prefix in zip(RAW_COLUMNS, PREFIXES):
                changed = ~np.isclose(
                    raw[raw_col].to_numpy(dtype=float), clean[raw_col].to_numpy(dtype=float),
                    equal_nan=True,
                )
                raw[f"frozen_param_{prefix}"] = frozen_rows & changed
            raw["station_id"] = f"{station_id}__n{network_id}"
            raw["__network"] = network_id
            networks.append(raw)
    return pd.concat(networks, ignore_index=True)


def add_frozen_channel_labels_from_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach evaluation-only frozen channel labels from untouched source CSVs.

    This is ground-truth attribution for reporting and validation; the helper
    itself never reads these labels when it scores a network.
    """
    result = frame.copy()
    for prefix in PREFIXES:
        result[f"frozen_param_{prefix}"] = False
    for station_id, positions in result.groupby("station_id").groups.items():
        source = DATA_DIR / f"{station_id}.csv"
        if not source.exists():
            continue
        clean = pd.read_csv(source)
        subset = result.loc[positions]
        frozen = subset["fault_type"].fillna("none").eq("frozen_value").to_numpy()
        for raw_col, prefix in zip(RAW_COLUMNS, PREFIXES):
            changed = ~np.isclose(
                subset[raw_col].to_numpy(dtype=float), clean[raw_col].to_numpy(dtype=float), equal_nan=True,
            )
            result.loc[positions, f"frozen_param_{prefix}"] = frozen & changed
    return result


def frozen_channel_columns(prefix: str) -> list[str]:
    """Features visible to one sensor channel's frozen specialist."""
    return [
        f"{prefix}_deviation", f"{prefix}_roc_1h", f"{prefix}_roc_3h",
        f"{prefix}_range_6h", f"{prefix}_peer_range_6h",
        f"{prefix}_peer_residual", f"{prefix}_peer_residual_6h_change",
        f"{prefix}_activity_gap",
        f"{prefix}_floor_frozen_match", f"{prefix}_frozen_streak", 
        f"{prefix}_normalized_roc_1h",
        *[f"{prefix}_peer_residual_lag_{lag}" for lag in range(1, 7)],
        *[f"{prefix}_own_change_lag_{lag}" for lag in range(1, 7)],
        *[f"{prefix}_peer_change_lag_{lag}" for lag in range(1, 7)],
    ]


def fit_frozen_channel_helpers(training_networks: pd.DataFrame):
    """Fit three channel-specific frozen confirmation classifiers."""
    featured = build_network_features(training_networks)
    helpers = {}
    for prefix in PREFIXES:
        cols = frozen_channel_columns(prefix)
        target = featured.get(f"frozen_param_{prefix}", pd.Series(False, index=featured.index)).astype(bool)
        positive = featured[target]
        if positive.empty:
            continue
        normal = featured[~featured["is_anomaly"]].sample(n=len(positive), random_state=314)
        sample = pd.concat([positive, normal]).sample(frac=1, random_state=314)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("forest", ExtraTreesClassifier(
                n_estimators=40, min_samples_leaf=3, class_weight="balanced",
                random_state=314, n_jobs=1,
            )),
        ])
        model.fit(sample[cols], target.loc[sample.index])
        helpers[prefix] = (model, cols)
    return helpers


def score_frozen_channels(scored: pd.DataFrame, helpers: dict, threshold: float) -> pd.DataFrame:
    """Return a row alert when any channel has high-confidence frozen evidence."""
    result = scored.copy()
    alerts = np.zeros(len(result), dtype=bool)
    for prefix, (model, cols) in helpers.items():
        probability = model.predict_proba(result[cols])[:, 1]
        result[f"frozen_{prefix}_probability"] = probability
        alerts |= probability >= threshold
    result["frozen_helper_alert"] = alerts
    return result


def fit_fault_helper(training_networks: pd.DataFrame):
    """Fit a conservative binary detector on balanced fresh-replay samples."""
    featured = build_network_features(training_networks)
    cols = feature_columns()
    positive = featured[featured["is_anomaly"]]
    normal = featured[~featured["is_anomaly"]].sample(n=len(positive), random_state=42)
    sample = pd.concat([positive, normal]).sample(frac=1, random_state=42)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", ExtraTreesClassifier(
            n_estimators=100, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=1,
        )),
    ])
    model.fit(sample[cols], sample["is_anomaly"])
    return model, cols


def predict_faults(model, cols: list[str], network: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = build_network_features(network)
    probability = model.predict_proba(result[cols])[:, 1]
    raw_nan_alert = network[["temperature_c", "pressure_hpa", "humidity_pct"]].isna().any(axis=1).values
    helper_columns = pd.DataFrame({
        "helper_probability": probability,
        "helper_alert": (probability >= threshold) | raw_nan_alert,
        "raw_nan_alert": raw_nan_alert,
    }, index=result.index)
    return pd.concat([result, helper_columns], axis=1)
