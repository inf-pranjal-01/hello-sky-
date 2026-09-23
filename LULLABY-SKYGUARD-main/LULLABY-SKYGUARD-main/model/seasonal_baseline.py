"""
model/seasonal_baseline.py

SkyGuard AI — Seasonal / Diurnal Baseline

Loads the real 3-month historical CSV fetched from Open-Meteo (the unlabelled
files in data/) and computes per-hour expected rates-of-change for each
parameter.  Used by detect.py's CUSUM to accumulate residuals instead of raw
ROC, preventing normal diurnal warming/cooling from triggering drift alerts.

DESIGN GOAL — easily swappable later:
  The single `_load_csv(station_id)` function is the only place that touches
  the filesystem.  Replace it with a TimescaleDB / WebSocket / API call and
  nothing else in this module changes.  The public interface (get_expected_roc)
  is stable.

USAGE:
  from model.seasonal_baseline import get_expected_roc
  residual = actual_roc - get_expected_roc(station_id, "temp", hour)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration — swap this path to change the
# data source without touching any other module.
# ─────────────────────────────────────────────
BASELINE_DATA_DIR = Path(__file__).parent.parent / "data"

# Fallback ROC when baseline is unavailable (e.g. new station).
# 0.0 means CUSUM behaves identically to the pre-hardening version for
# stations without historical data — safe, not silent.
FALLBACK_EXPECTED_ROC = 0.0

# ─────────────────────────────────────────────
# Internal cache: station_id → {param → array[24]}
# ─────────────────────────────────────────────
_cache: Dict[str, Dict[str, np.ndarray]] = {}

# Parameters we model — keys must match features.py PARAM_PREFIXES
_PARAMS = {
    "temp":     "temperature_c",
    "pressure": "pressure_hpa",
    "humidity": "humidity_pct",
}


def _load_csv(station_id: str) -> Optional[pd.DataFrame]:
    """
    Load the unlabelled historical CSV for a station.
    Returns None if the file does not exist (new station, live-only data).
    Swappable: replace with a DB query or API call without changing callers.
    """
    path = BASELINE_DATA_DIR / f"{station_id}.csv"
    if not path.exists():
        logger.debug("[seasonal_baseline] No historical CSV for %s at %s", station_id, path)
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception as exc:
        logger.warning("[seasonal_baseline] Could not load %s: %s", path, exc)
        return None


def _build_expected_roc(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Given the raw historical DataFrame, compute mean hourly ROC for each
    parameter, grouped by hour-of-day.

    Returns dict: param_prefix → np.ndarray of shape (24,).
    Array index = hour (0–23).  Values are mean diffs in the original units
    per hour (°C/h, hPa/h, %/h).
    """
    df = df.sort_values("timestamp").copy()
    df["_hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    result: Dict[str, np.ndarray] = {}
    for prefix, col in _PARAMS.items():
        if col not in df.columns:
            result[prefix] = np.zeros(24)
            continue
        # Causal diff: roc at row i = value[i] - value[i-1]
        # We compute this globally first, then group by hour of the *current*
        # row (the reading whose ROC we're assessing).
        df["_roc"] = df[col].diff()
        by_hour = (
            df.dropna(subset=["_roc"])
            .groupby("_hour")["_roc"]
            .mean()
            .reindex(range(24), fill_value=0.0)
        )
        result[prefix] = by_hour.to_numpy(dtype=float)

    return result


def _ensure_loaded(station_id: str) -> None:
    """Populate cache for station_id if not already present."""
    if station_id in _cache:
        return
    df = _load_csv(station_id)
    if df is None:
        # Mark as attempted so we don't retry on every reading.
        _cache[station_id] = {p: np.zeros(24) for p in _PARAMS}
    else:
        _cache[station_id] = _build_expected_roc(df)
        logger.info(
            "[seasonal_baseline] Loaded baseline for %s (%d rows, %d months)",
            station_id, len(df),
            max(1, round((df["timestamp"].max() - df["timestamp"].min()).days / 30))
        )


def get_expected_roc(station_id: str, param: str, hour: int) -> float:
    """
    Public API.  Returns the expected rate-of-change (same units as raw
    sensor reading, per hour) for `param` at `hour` for `station_id`.

    Args:
        station_id:  e.g. "AWS-MUM-007"
        param:       one of "temp", "pressure", "humidity"
        hour:        0–23 (local time)

    Returns:
        float — expected ROC.  0.0 if data unavailable.

    This is the single function detect.py calls.  Swap the data source by
    replacing _load_csv() without touching this function's signature.
    """
    _ensure_loaded(station_id)
    try:
        return float(_cache[station_id][param][hour % 24])
    except (KeyError, IndexError):
        return FALLBACK_EXPECTED_ROC


def preload_all(data_dir: Optional[Path] = None) -> None:
    """
    Optional warm-up: preload baselines for all stations whose CSVs exist in
    data_dir.  Call once at startup (e.g. from lifespan in main.py) to avoid
    the tiny per-station lazy-load latency on the first reading.
    """
    target = data_dir or BASELINE_DATA_DIR
    for csv_path in sorted(target.glob("AWS-*.csv")):
        if "_labeled" in csv_path.name:
            continue
        sid = csv_path.stem
        _ensure_loaded(sid)
    logger.info("[seasonal_baseline] Preloaded baselines for %d stations", len(_cache))
