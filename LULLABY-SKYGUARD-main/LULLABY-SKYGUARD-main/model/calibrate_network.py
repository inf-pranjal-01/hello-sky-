"""
model/calibrate_network.py -- Stage 1 Calibration Script.

Empirically calibrates hour-to-hour change residuals between AWS cluster centers
and their respective neighbors across all 7 clusters defined in config.py.

Residual definition:
    residual_t = [center(t) - center(t-1)] - [neighbor(t) - neighbor(t-1)]

Historical data is sourced from Open-Meteo reanalysis (ERA5/ERA5-Land/ECMWF IFS,
9-25km native grid). Real physical AWS hardware in production will naturally exhibit
higher variance than reanalysis products.

This script identifies and explicitly flags pairs whose residual standard deviation
is near or below the sensor/grid resolution floor (e.g. Kolkata <-> Bidhannagar ~5.5km,
Varanasi <-> Ramnagar ~7.5km). Flagged pairs are marked as excluded from corroboration
for that parameter.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLUSTERS

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
OUTPUT_ARTIFACT_PATH = ARTIFACTS_DIR / "network_corroboration.pkl"
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "isolation_forest.pkl"

PARAMETERS = ["temperature_c", "pressure_hpa", "humidity_pct"]

# Multiplier for divergence threshold: mean +/- SIGMA_MULTIPLIER * std
SIGMA_MULTIPLIER = 3.5

# Empirical minimum standard deviation required to separate physical weather divergence
# from reanalysis grid interpolation / ADC quantization noise floors.
RESOLUTION_FLOORS = {
    "temperature_c": 0.35,   # °C
    "pressure_hpa": 0.09,     # hPa (below single ADC count quantization)
    "humidity_pct": 2.5,     # % RH
}


def compute_pair_residuals(
    center_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    parameter: str
) -> pd.Series:
    """Computes hour-to-hour rate-of-change residual between center and neighbor."""
    # Ensure aligned timestamps
    merged = pd.merge(
        center_df[["timestamp", parameter]],
        neighbor_df[["timestamp", parameter]],
        on="timestamp",
        suffixes=("_center", "_neighbor"),
    ).sort_values("timestamp").reset_index(drop=True)

    c_diff = merged[f"{parameter}_center"].diff()
    n_diff = merged[f"{parameter}_neighbor"].diff()
    residual = (c_diff - n_diff).dropna()
    return residual


def calibrate_network_corroboration() -> dict:
    """
    Main calibration function.
    Iterates through all clusters and (center, neighbor) pairs, computing
    distribution statistics and divergence thresholds for each parameter.
    """
    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_data": "raw clean historical CSVs (Open-Meteo ERA5 reanalysis)",
            "sigma_multiplier": SIGMA_MULTIPLIER,
            "resolution_floors": RESOLUTION_FLOORS,
            "documentation": (
                "Thresholds are empirically calibrated against ERA5/ERA5-Land reanalysis historical data. "
                "Pairs near or below the reanalysis grid resolution have been flagged and excluded. "
                "Physical AWS hardware in production will show higher natural disagreement than this calibration assumes."
            ),
        },
        "clusters": {},
    }

    excluded_summary = []

    for cluster_id, cluster_info in CLUSTERS.items():
        center_meta = cluster_info["center"]
        center_id = center_meta["station_id"]
        center_csv = DATA_DIR / f"{center_id}.csv"

        if not center_csv.exists():
            print(f"[WARN] Center CSV not found: {center_csv}")
            continue

        center_df = pd.read_csv(center_csv, parse_dates=["timestamp"])

        cluster_entry = {
            "center_station_id": center_id,
            "center_name": center_meta["name"],
            "pairs": {},
        }

        for neighbor_meta in cluster_info["neighbors"]:
            neighbor_id = neighbor_meta["station_id"]
            neighbor_csv = DATA_DIR / f"{neighbor_id}.csv"

            if not neighbor_csv.exists():
                print(f"[WARN] Neighbor CSV not found: {neighbor_csv}")
                continue

            neighbor_df = pd.read_csv(neighbor_csv, parse_dates=["timestamp"])

            pair_entry = {
                "neighbor_station_id": neighbor_id,
                "neighbor_name": neighbor_meta["name"],
                "parameters": {},
            }

            for param in PARAMETERS:
                res = compute_pair_residuals(center_df, neighbor_df, param)
                mean_val = float(res.mean())
                std_val = float(res.std())
                floor = RESOLUTION_FLOORS[param]

                # Check if std is too small to provide meaningful corroboration
                is_excluded = std_val < floor
                reason = "near_or_below_grid_resolution" if is_excluded else None

                divergence_threshold = SIGMA_MULTIPLIER * std_val

                if is_excluded:
                    excluded_summary.append({
                        "cluster": cluster_id,
                        "pair": f"{center_meta['name']} ({center_id}) <-> {neighbor_meta['name']} ({neighbor_id})",
                        "parameter": param,
                        "std": std_val,
                        "floor": floor,
                        "reason": reason,
                    })

                pair_entry["parameters"][param] = {
                    "mean": round(mean_val, 6),
                    "std": round(std_val, 6),
                    "divergence_threshold": round(divergence_threshold, 6),
                    "lower_threshold": round(mean_val - divergence_threshold, 6),
                    "upper_threshold": round(mean_val + divergence_threshold, 6),
                    "n_samples": len(res),
                    "excluded": is_excluded,
                    "exclusion_reason": reason,
                }

            cluster_entry["pairs"][neighbor_id] = pair_entry

        results["clusters"][cluster_id] = cluster_entry

    # Persist standalone artifact
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(results, OUTPUT_ARTIFACT_PATH)
    print(f"[INFO] Saved calibrated network thresholds -> {OUTPUT_ARTIFACT_PATH}")

    # Also update model_artifacts/isolation_forest.pkl if present
    if MODEL_ARTIFACT_PATH.exists():
        try:
            model_artifact = joblib.load(MODEL_ARTIFACT_PATH)
            model_artifact["network_corroboration"] = results
            joblib.dump(model_artifact, MODEL_ARTIFACT_PATH)
            print(f"[INFO] Bundled network_corroboration into {MODEL_ARTIFACT_PATH}")
        except Exception as e:
            print(f"[WARN] Could not update {MODEL_ARTIFACT_PATH}: {e}")

    # Print summary report
    print("\n" + "=" * 90)
    print(f"EMPIRICALLY CALIBRATED NETWORK CORROBORATION REPORT (Threshold = mean +/- {SIGMA_MULTIPLIER} * std)")
    print("=" * 90)

    for cid, cdata in results["clusters"].items():
        print(f"\n--- Cluster {cid} (Center: {cdata['center_name']} - {cdata['center_station_id']}) ---")
        for nid, pdata in cdata["pairs"].items():
            print(f"  Neighbor: {pdata['neighbor_name']} ({nid}):")
            for param, stats in pdata["parameters"].items():
                status = "[EXCLUDED - Sub-Grid Noise]" if stats["excluded"] else "[ACTIVE]"
                print(f"    {param:<15}: mean={stats['mean']:+.4f}, std={stats['std']:.4f}, "
                      f"thresh=+/-{stats['divergence_threshold']:.4f}  {status}")

    print("\n" + "=" * 90)
    print("STAGE 1 SUMMARY: EXCLUDED PAIRS DUE TO REANALYSIS GRID RESOLUTION LIMITATION")
    print("=" * 90)
    if excluded_summary:
        print(f"{'Cluster':<9} {'Pair':<48} {'Parameter':<16} {'Std':<8} {'Floor':<8} {'Reason'}")
        print("-" * 105)
        for item in excluded_summary:
            print(f"{item['cluster']:<9} {item['pair']:<48} {item['parameter']:<16} "
                  f"{item['std']:<8.4f} {item['floor']:<8.4f} {item['reason']}")
    else:
        print("No pairs were excluded.")
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    calibrate_network_corroboration()
