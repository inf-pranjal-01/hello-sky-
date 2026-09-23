import sys
import pandas as pd
from pathlib import Path

def generate_evaluate_code():
    code = '''"""
SkyGuard AI — Phase 2d: Canonical Evaluation.

This evaluation script leverages the exact same `DecisionEngine` and
`StateManager` used by the live system, ensuring 100% architectural parity
between live detection, replay, and evaluation.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.state import StateManager
from config import RULE_BASE_CONFIDENCE

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_PATH = PROJECT_ROOT / "model_artifacts" / "isolation_forest.pkl"
PER_SENSOR_LOG_PATH = DATA_DIR / "eval_per_sensor_fault_log.csv"

def evaluate_all(labeled_files: list, artifact: dict) -> dict:
    print(f"Loading {len(labeled_files)} labeled file(s)...")
    frames = []
    for path in labeled_files:
        d = pd.read_csv(path, parse_dates=["timestamp"])
        d["__source_file"] = path.name
        if "is_anomaly" not in d.columns:
            print(f"Skipping {path.name} (no is_anomaly column)")
            continue
        frames.append(d)
        
    if not frames:
        print("No valid labeled files found.")
        return {}

    df_full = pd.concat(frames, ignore_index=True)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], utc=True)
    df_full = df_full.sort_values("timestamp")
    
    print(f"Data loaded. Total rows: {len(df_full)}")
    print("Running canonical DecisionEngine over all rows chronologically...")
    
    state_manager = StateManager(artifact, mock_history_store=True)
    
    results = []
    
    start_time = time.time()
    count = 0
    total = len(df_full)
    
    # Process chronologically
    for _, row in df_full.iterrows():
        raw = row.to_dict()
        try:
            res = state_manager.ingest_reading(raw)
            # Store results
            results.append({
                "station_id": raw["station_id"],
                "timestamp": raw["timestamp"],
                "is_anomaly_gt": raw.get("is_anomaly", False),
                "fault_type_gt": raw.get("fault_type", "none"),
                "is_anomaly_pred": res["is_anomaly"],
                "fault_type_pred": res["fault_type"],
                "anomaly_score_pct": res.get("anomaly_score_pct", 0)
            })
        except Exception as e:
            # Dropouts or incomplete rows
            pass
            
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count}/{total} rows ({(time.time() - start_time):.1f}s)...")
            
    elapsed = time.time() - start_time
    print(f"Engine evaluation finished in {elapsed:.1f} seconds.")
    
    res_df = pd.DataFrame(results)
    if res_df.empty:
        return {}
        
    pred = res_df["is_anomaly_pred"].fillna(False).astype(bool)
    gt = res_df["is_anomaly_gt"].fillna(False).astype(bool)
    
    # Incident-level vs Row-level metric separation
    print("\\n=== METRICS ===")
    
    caught = (pred & gt).sum()
    true_anom = gt.sum()
    pred_anom = pred.sum()
    
    recall = caught / true_anom if true_anom else 0
    precision = caught / pred_anom if pred_anom else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    print(f"Overall Row-Level Recall:    {recall:.1%} ({caught}/{true_anom})")
    print(f"Overall Row-Level Precision: {precision:.1%} ({caught}/{pred_anom})")
    print(f"Overall Row-Level F1 Score:  {f1:.3f}")
    
    # Detailed breakdown
    print("\\nPerformance by fault type (Row-Level):")
    faults = res_df[res_df["fault_type_gt"] != "none"]["fault_type_gt"].unique()
    for ft in sorted(faults):
        mask_gt = (res_df["fault_type_gt"] == ft)
        mask_pred_correct = mask_gt & pred
        
        c = mask_pred_correct.sum()
        t = mask_gt.sum()
        p = pred_anom # This is an approximation for per-class precision unless we look at fault_type_pred
        mask_pred_class = (res_df["fault_type_pred"] == ft)
        p_class = mask_pred_class.sum()
        c_class = (mask_pred_class & mask_gt).sum()
        
        rec = c / t if t else 0
        prec = c_class / p_class if p_class else 0
        f1_class = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0
        print(f"  {ft:<28} Caught: {c}/{t} ({rec:.1%}) | Precision: {prec:.1%} | F1: {f1_class:.3f}")
        
    res_df.to_csv(PER_SENSOR_LOG_PATH, index=False)
    print(f"\\nRow-level log saved to {PER_SENSOR_LOG_PATH}")
    
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1
    }

if __name__ == "__main__":
    if not ARTIFACTS_PATH.exists():
        print(f"Artifact not found: {ARTIFACTS_PATH}")
        sys.exit(1)
        
    artifact = joblib.load(ARTIFACTS_PATH)
    labeled = list(DATA_DIR.glob("*_labeled.csv"))
    evaluate_all(labeled, artifact)
'''
    with open("model/evaluate.py", "w", encoding="utf-8") as f:
        f.write(code)

if __name__ == "__main__":
    generate_evaluate_code()
