import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RULE_CONFIDENCE_BYPASS
from model.features import build_feature_matrix, FEATURE_COLUMNS
from model.evaluate import run_rule_engine_and_health, _score_and_report

DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "uscrn_validation_slice.csv"

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run fetch_uscrn_validation_slice.py first.")
        return

    print(f"Loading {INPUT_FILE.name}...")
    df_full = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    
    labels = df_full[['station_id', 'timestamp', 'is_anomaly', 'fault_type']].copy()
    labels['is_anomaly'] = labels['is_anomaly'].fillna(False).astype(bool)
    labels['fault_type'] = labels['fault_type'].fillna('none')
    
    df = df_full.drop(columns=['is_anomaly', 'fault_type'], errors='ignore')

    print("Building features (using mocked pressure to satisfy pipeline)...")
    # Mock pressure with safe random noise to prevent dropouts, frozen, or physical bounds from firing
    np.random.seed(42)
    df['pressure_hpa'] = np.random.normal(1013.25, 1.0, size=len(df))
    
    # Build features
    featured = build_feature_matrix(df)
    
    # We cannot use evaluate.py's strict complete-row filter because pressure is always NaN.
    # Instead, we just drop the first 48 hours for each station to simulate the warm-up period,
    # or drop rows where temperature_rolling_mean is NaN.
    complete_temp_humidity = featured[['temp_rolling_mean', 'humidity_rolling_mean']].notna().all(axis=1)
    n_dropped = (~complete_temp_humidity).sum()
    featured = featured[complete_temp_humidity].reset_index(drop=True)
    
    # Dummy artifact with global thresholds for everything, since we aren't retraining on USCRN
    # We will just pull the global fallback threshold from config or features? 
    # Actually, evaluate.py requires artifact["rule_thresholds"].
    # Let's mock the artifact to provide dummy thresholds.
    # We can load the real artifact to reuse its calibrated thresholds!
    import joblib
    ARTIFACTS_PATH = PROJECT_ROOT / "model_artifacts" / "isolation_forest.pkl"
    if ARTIFACTS_PATH.exists():
        artifact = joblib.load(ARTIFACTS_PATH)
        print("Loaded real artifact for rule thresholds.")
    else:
        print("No artifact found. Cannot run rule engine because calibrated thresholds are missing.")
        return

    print(f"Running rule engine on {len(featured)} rows...")
    featured, row_hard, row_rule_conf, per_sensor_log, recovery_log = run_rule_engine_and_health(featured, artifact)
    
    # In a rules-only eval, there is no model. So anomaly is triggered if rule confidence > bypass (or any hard rule).
    # We will use RULE_CONFIDENCE_BYPASS.
    predicted = row_hard | (row_rule_conf > RULE_CONFIDENCE_BYPASS)
    
    featured = featured.merge(labels, on=["station_id", "timestamp"], how="left")
    featured["is_anomaly"] = featured["is_anomaly"].fillna(False).astype(bool)
    featured["fault_type"] = featured["fault_type"].fillna("none")
    featured["__predicted"] = predicted
    
    _score_and_report(featured, "RULES ONLY EVALUATION (USCRN SLICE)", n_dropped)
    
    print("\nCaveat: This validates the rule layer only. The fused model+rules pipeline is not tested here due to permanently missing pressure data.")

if __name__ == '__main__':
    main()
