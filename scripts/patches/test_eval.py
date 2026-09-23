import sys
import json
import time
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.state import StateManager
ARTIFACTS_PATH = PROJECT_ROOT / "model_artifacts" / "isolation_forest.pkl"

DATA_DIR = PROJECT_ROOT / "data"

def evaluate_chronological(labeled_files, artifact):
    print("Loading data...")
    frames = []
    for path in labeled_files:
        d = pd.read_csv(path, parse_dates=["timestamp"])
        d["__source_file"] = path.name
        frames.append(d)
        
    df_full = pd.concat(frames, ignore_index=True)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], utc=True)
    df_full = df_full.sort_values("timestamp")
    
    print(f"Data loaded. Total rows: {len(df_full)}")
    print("Running canonical DecisionEngine over all rows chronologically...")
    
    state = StateManager(artifact, mock_history_store=True)
    
    results = []
    
    start = time.time()
    count = 0
    total = len(df_full)
    for _, row in df_full.iterrows():
        raw = row.to_dict()
        try:
            res = state.ingest_reading(raw)
            results.append({
                "station_id": raw["station_id"],
                "timestamp": raw["timestamp"],
                "is_anomaly_gt": raw.get("is_anomaly", False),
                "fault_type_gt": raw.get("fault_type", "none"),
                "is_anomaly_pred": res["is_anomaly"],
                "fault_type_pred": res["fault_type"]
            })
        except Exception as e:
            pass
            
        count += 1
        if count % 5000 == 0:
            print(f"Processed {count}/{total} rows... ({(time.time() - start):.1f}s)")
            # Let's break early just for the test
            break
            
    end = time.time()
    print(f"Evaluation finished in {end - start:.1f} seconds.")

if __name__ == "__main__":
    import joblib
    artifact = joblib.load(ARTIFACTS_PATH)
    files = list((DATA_DIR).glob("*_labeled.csv"))
    evaluate_chronological(files, artifact)
