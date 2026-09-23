"""
SkyGuard AI — Phase 2e: Explainability.

Answers "WHICH sensor is most likely at fault," not just "the score is
94%" -- a rule-fired verdict already carries this (rules_fired is
(rule_type, param) tuples), but a model-only "statistical_anomaly"
verdict is one scalar across all 22 features with no indication of
which raw sensor -- temperature/pressure/humidity -- is actually
implicated. Any one of the three can be the genuinely faulty reading
even when the other two are fine; this file localizes the station-level
verdict down to specific sensor(s). Powers GET /api/explain/{anomaly_id}
per ARCHITECTURE.md's contract.

SHAP + sklearn's IsolationForest is genuinely fragile across shap
versions (IsolationForest's decision_function isn't the same shape
TreeExplainer expects for a standard regressor/classifier -- support
varies by shap/sklearn version pairing and is known to raise on some
combinations). Rather than let a shap incompatibility take the whole
endpoint down, this degrades to a deterministic magnitude ranking (same
feature-magnitude-based ordering, no shap dependency) if TreeExplainer
construction or scoring fails -- logged loudly, not silently, so a
degraded explanation is visible if it's happening, not mistaken for a
real SHAP attribution.

FEATURE NAME MAP matches ARCHITECTURE.md's /api/explain/{id} example
shape ("Temp Deviation", "Pressure Inconsistency", "Rate of Change
(Temp)", "Time of Day Pattern", "Seasonal Pattern") so main.py returns
this file's output with zero translation.
"""

import sys
from pathlib import Path
import logging

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# feature_column -> display name. Every features.py FEATURE_COLUMNS
# entry MUST have one here -- a silently-missing feature in an
# explanation is worse than a loud startup error, see ExplainerCache.explain.
FEATURE_DISPLAY_NAMES = {
    "temperature_c": "Temperature",
    "pressure_hpa": "Pressure",
    "humidity_pct": "Humidity",
    "temp_deviation": "Temp Deviation",
    "pressure_deviation": "Pressure Deviation",
    "humidity_deviation": "Humidity Deviation",
    "temp_roc_1h": "Rate of Change (Temp, 1h)",
    "pressure_roc_1h": "Rate of Change (Pressure, 1h)",
    "humidity_roc_1h": "Rate of Change (Humidity, 1h)",
    "temp_roc_3h": "Rate of Change (Temp, 3h)",
    "pressure_roc_3h": "Rate of Change (Pressure, 3h)",
    "humidity_roc_3h": "Rate of Change (Humidity, 3h)",
    "temp_volatility_z": "Temp Volatility",
    "pressure_volatility_z": "Pressure Volatility",
    "humidity_volatility_z": "Humidity Volatility",
    "hour_sin": "Time of Day Pattern",
    "hour_cos": "Time of Day Pattern",
    "doy_sin": "Seasonal Pattern",
    "doy_cos": "Seasonal Pattern",
    "dewpoint_depression_c": "Dewpoint Depression",
    "vapor_pressure_deficit_kpa": "Vapor Pressure Deficit",
    "vapor_pressure_consistency_dev": "Vapor Pressure Consistency",
    "dt_hours": "Elapsed Time",
    "temp_robust_scale": "Robust Temperature Scale",
    "pressure_robust_scale": "Robust Pressure Scale",
    "humidity_robust_scale": "Robust Humidity Scale",
    "temp_hours_since_valid": "Hours Since Valid Temperature",
    "pressure_hours_since_valid": "Hours Since Valid Pressure",
    "humidity_hours_since_valid": "Hours Since Valid Humidity",
    "temp_range_1h": "Temperature Range (1h)",
    "pressure_range_1h": "Pressure Range (1h)",
    "humidity_range_1h": "Humidity Range (1h)",
    "temp_range_3h": "Temperature Range (3h)",
    "pressure_range_3h": "Pressure Range (3h)",
    "humidity_range_3h": "Humidity Range (3h)",
    "temp_range_6h": "Temperature Range (6h)",
    "pressure_range_6h": "Pressure Range (6h)",
    "humidity_range_6h": "Humidity Range (6h)",
    "temp_range_24h": "Temperature Range (24h)",
    "pressure_range_24h": "Pressure Range (24h)",
    "humidity_range_24h": "Humidity Range (24h)",
    "temp_slope_6h": "Temperature Slope (6h)",
    "pressure_slope_6h": "Pressure Slope (6h)",
    "humidity_slope_6h": "Humidity Slope (6h)",
    "temp_slope_24h": "Temperature Slope (24h)",
    "pressure_slope_24h": "Pressure Slope (24h)",
    "humidity_slope_24h": "Humidity Slope (24h)",
    "temp_same_hour_res": "Temperature Same-Hour Residual",
    "pressure_same_hour_res": "Pressure Same-Hour Residual",
    "humidity_same_hour_res": "Humidity Same-Hour Residual",
}

# Which raw sensor each feature implicates -- None means the feature
# couples multiple sensors or is a time signal, not attributable to
# one sensor alone.
FEATURE_TO_PARAM = {
    "temperature_c": "temperature_c",
    "pressure_hpa": "pressure_hpa",
    "humidity_pct": "humidity_pct",
    "temp_deviation": "temperature_c", "temp_roc_1h": "temperature_c", "temp_roc_3h": "temperature_c",
    "temp_volatility_z": "temperature_c",
    "pressure_deviation": "pressure_hpa", "pressure_roc_1h": "pressure_hpa", "pressure_roc_3h": "pressure_hpa",
    "pressure_volatility_z": "pressure_hpa",
    "humidity_deviation": "humidity_pct", "humidity_roc_1h": "humidity_pct", "humidity_roc_3h": "humidity_pct",
    "humidity_volatility_z": "humidity_pct",
    "hour_sin": None, "hour_cos": None, "doy_sin": None, "doy_cos": None,
    "dewpoint_depression_c": None,
    "vapor_pressure_deficit_kpa": None,
    "vapor_pressure_consistency_dev": None,
    "dt_hours": None,
    "temp_robust_scale": "temperature_c", "pressure_robust_scale": "pressure_hpa", "humidity_robust_scale": "humidity_pct",
    "temp_hours_since_valid": "temperature_c", "pressure_hours_since_valid": "pressure_hpa", "humidity_hours_since_valid": "humidity_pct",
    "temp_range_1h": "temperature_c", "pressure_range_1h": "pressure_hpa", "humidity_range_1h": "humidity_pct",
    "temp_range_3h": "temperature_c", "pressure_range_3h": "pressure_hpa", "humidity_range_3h": "humidity_pct",
    "temp_range_6h": "temperature_c", "pressure_range_6h": "pressure_hpa", "humidity_range_6h": "humidity_pct",
    "temp_range_24h": "temperature_c", "pressure_range_24h": "pressure_hpa", "humidity_range_24h": "humidity_pct",
    "temp_slope_6h": "temperature_c", "pressure_slope_6h": "pressure_hpa", "humidity_slope_6h": "humidity_pct",
    "temp_slope_24h": "temperature_c", "pressure_slope_24h": "pressure_hpa", "humidity_slope_24h": "humidity_pct",
    "temp_same_hour_res": "temperature_c", "pressure_same_hour_res": "pressure_hpa", "humidity_same_hour_res": "humidity_pct",
}


class ExplainerCache:
    """
    shap.TreeExplainer(model) walks every tree to build -- expensive.
    Build ONCE per loaded artifact, same "load once at startup"
    principle as detect.py's load_model(). state.py's StateManager
    owns one instance.
    """

    def __init__(self, artifact: dict):
        self.artifact = artifact
        self._explainer = None
        self._shap_broken = not _SHAP_AVAILABLE
        if _SHAP_AVAILABLE:
            try:
                self._explainer = shap.TreeExplainer(artifact["model"])
            except Exception as e:
                logging.getLogger(__name__).warning(f"[explain] shap.TreeExplainer construction failed, falling back to magnitude ranking for all explanations: {e!r}")
                self._shap_broken = True

    def explain(self, feature_row: pd.Series) -> dict:
        """Returns {"method": str, "features": [{"name", "impact", "column"}, ...]}"""
        feature_columns = self.artifact["feature_columns"]
        missing = [c for c in feature_columns if c not in FEATURE_DISPLAY_NAMES]
        if missing:
            raise KeyError(f"FEATURE_DISPLAY_NAMES missing entries for: {missing}")

        X = feature_row[feature_columns].values.reshape(1, -1).astype(np.float64)
        if np.isnan(X).any():
            return {"method": "none", "features": []}

        if not self._shap_broken:
            try:
                shap_values = self._explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                return {"method": "shap", "features": self._format(feature_columns, shap_values[0])}
            except Exception as e:
                logging.getLogger(__name__).warning(f"[explain] shap_values() failed on this reading, falling back to magnitude ranking: {e!r}")

        # Standardize features for fallback ranking so physical units (e.g. pressure 1013 hPa)
        # do not artificially dominate over genuine sensor deviations and rate of changes.
        scaled_impacts = []
        for col, val in zip(feature_columns, X[0]):
            if col == "pressure_hpa":
                scaled_impacts.append((val - 1013.25) / 10.0)
            elif col == "temperature_c":
                scaled_impacts.append((val - 25.0) / 10.0)
            elif col == "humidity_pct":
                scaled_impacts.append((val - 50.0) / 25.0)
            elif "sin" in col or "cos" in col or "dt_hours" in col:
                scaled_impacts.append(0.0)
            else:
                scaled_impacts.append(float(val))

        return {"method": "feature_magnitude_fallback", "features": self._format(feature_columns, scaled_impacts)}

    def _format(self, feature_columns, impacts) -> list[dict]:
        rows = [
            {"name": FEATURE_DISPLAY_NAMES[col], "impact": round(float(val), 4), "column": col}
            for col, val in zip(feature_columns, impacts)
        ]
        rows.sort(key=lambda r: abs(r["impact"]), reverse=True)
        return rows


def likely_faulty_params(features: list[dict], top_n: int = 3) -> list[str]:
    """Collapses the ranked feature list to WHICH raw sensor(s) are implicated, using the top_n highest-magnitude features."""
    seen = []
    for row in features[:top_n]:
        param = FEATURE_TO_PARAM.get(row["column"])
        if param and param not in seen:
            seen.append(param)
    return seen
