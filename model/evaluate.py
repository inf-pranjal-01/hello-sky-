"""
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


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _row_metrics(frame: pd.DataFrame) -> dict:
    truth = frame["is_anomaly_gt"].fillna(False).astype(bool)
    predicted = frame["is_anomaly_pred"].fillna(False).astype(bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall)}


def _episodes(frame: pd.DataFrame, positive_col: str, type_col: str, label: str) -> list[dict]:
    """Build episodes from contiguous positive rows, scoped by station and fault type."""
    episodes = []
    for station, group in frame.groupby("station_id", dropna=False):
        group = group.sort_values("timestamp")
        active = group[positive_col].fillna(False).astype(bool)
        starts = active & ~active.shift(1, fill_value=False)
        ends = active & ~active.shift(-1, fill_value=False)
        for start_idx, end_idx in zip(group.index[starts], group.index[ends]):
            episodes.append({"station_id": station, "fault_type": frame.at[start_idx, type_col],
                             "start": frame.at[start_idx, "timestamp"],
                             "end": frame.at[end_idx, "timestamp"], "label": label})
    return episodes


def episodic_metrics(frame: pd.DataFrame, fault_type: str) -> dict:
    """Overlap-match predicted and labeled frozen/drift episodes per station.

    Current replay labels do not retain the injected parameter when two
    independent faults overlap. Matching is therefore station+fault-type
    scoped until the labeled format carries parameter-level event IDs.
    """
    subset = frame.copy()
    subset["gt_episode"] = subset["fault_type_gt"].eq(fault_type)
    subset["pred_episode"] = subset["is_anomaly_pred"].fillna(False).astype(bool) & subset["fault_type_pred"].eq(fault_type)
    gt = _episodes(subset, "gt_episode", "fault_type_gt", "gt")
    pred = _episodes(subset, "pred_episode", "fault_type_pred", "pred")
    matched_gt, matched_pred = set(), set()
    for pi, pe in enumerate(pred):
        for gi, ge in enumerate(gt):
            if (pe["station_id"] == ge["station_id"] and
                    max(pe["start"], ge["start"]) <= min(pe["end"], ge["end"])):
                matched_pred.add(pi)
                matched_gt.add(gi)
    tp, fp, fn = len(matched_gt), len(pred) - len(matched_pred), len(gt) - len(matched_gt)
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "gt_episodes": len(gt), "pred_episodes": len(pred),
            "precision": precision, "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall)}

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
    
    class MockHistoryStore:
        def append(self, *args, **kwargs): pass
        def mark_spike(self, *args, **kwargs): pass
        def get_all(self, *args, **kwargs): return pd.DataFrame()
        
    metadata = pd.read_csv(DATA_DIR / "stations_metadata.csv")
    state_manager = StateManager(metadata, artifact, history_store=MockHistoryStore())
    # USER REQUEST: Skip heavy SHAP generation during pure evaluation benchmark to save time
    state_manager.explainer = None
    
    results = []
    
    start_time = time.time()
    count = 0
    total = len(df_full)
    
    # Process chronologically
    for _, row in df_full.iterrows():
        raw = row.to_dict()
        try:
            res = state_manager.ingest_reading(raw["station_id"], raw, raw["timestamp"])
            # Store results
            results.append({
                "station_id": raw["station_id"],
                "timestamp": raw["timestamp"],
                "is_anomaly_gt": raw.get("is_anomaly", False),
                "fault_type_gt": raw.get("fault_type", "none"),
                "is_anomaly_pred": res["is_anomaly"],
                "fault_type_pred": res["fault_type"],
                "anomaly_score_pct": res.get("anomaly_score_pct", 0),
                "fault_parameter_pred": ",".join(res.get("likely_faulty_sensors") or [])
            })
        except Exception as e:
            # Dropouts or incomplete rows
            if count == 0:
                print(f"ERROR: {e}")
            pass
            
        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count}/{total} rows ({(time.time() - start_time):.1f}s)...")
            
    elapsed = time.time() - start_time
    print(f"Engine evaluation finished in {elapsed:.1f} seconds.")
    
    res_df = pd.DataFrame(results)

    if res_df.empty:
        return {}
        
    pred = res_df["is_anomaly_pred"].fillna(False).astype(bool)
    gt = res_df["is_anomaly_gt"].fillna(False).astype(bool)
    
    # Incident-level vs Row-level metric separation
    print("\n=== METRICS ===")
    
    overall = _row_metrics(res_df)
    recall, precision, f1 = overall["recall"], overall["precision"], overall["f1"]
    print(f"Overall Row-Level Precision: {precision:.1%} (TP={overall['tp']}, FP={overall['fp']})")
    print(f"Overall Row-Level Recall:    {recall:.1%} (TP={overall['tp']}, FN={overall['fn']})")
    print(f"Overall Row-Level F1 Score:  {f1:.3f}")
    print("\nEpisodic metrics (same station + fault type; overlap match):")
    episodic = {}
    for ft in ("frozen_value", "drift"):
        episodic[ft] = episodic_metrics(res_df, ft)
        m = episodic[ft]
        print(f"  {ft}: P={m['precision']:.1%}, R={m['recall']:.1%}, "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']} "
              f"(GT episodes={m['gt_episodes']}, predicted={m['pred_episodes']})")
    print("  Note: labeled files lack parameter-level event IDs; overlapping faults cannot yet be parameter-matched.")
    
    # Detailed breakdown
    print("\nPerformance by fault type (Row-Level):")
    faults = res_df[res_df["fault_type_gt"] != "none"]["fault_type_gt"].unique()
    for ft in sorted(faults):
        mask_gt = (res_df["fault_type_gt"] == ft)
        mask_pred_correct = mask_gt & pred
        
        c = mask_pred_correct.sum()
        t = mask_gt.sum()
        mask_pred_class = (res_df["fault_type_pred"] == ft)
        p_class = mask_pred_class.sum()
        c_class = (mask_pred_class & mask_gt).sum()
        
        rec = c / t if t else 0
        prec = c_class / p_class if p_class else 0
        f1_class = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0
        print(f"  {ft:<28} Caught: {c}/{t} ({rec:.1%}) | Precision: {prec:.1%} | F1: {f1_class:.3f}")
        
    res_df.to_csv(PER_SENSOR_LOG_PATH, index=False)
    print(f"\nRow-level log saved to {PER_SENSOR_LOG_PATH}")
    
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": overall["tp"], "fp": overall["fp"], "fn": overall["fn"],
        "episodic": episodic
    }

if __name__ == "__main__":
    if not ARTIFACTS_PATH.exists():
        print(f"Artifact not found: {ARTIFACTS_PATH}")
        sys.exit(1)
        
    artifact = joblib.load(ARTIFACTS_PATH)
    labeled = list(DATA_DIR.glob("*_labeled.csv"))
    evaluate_all(labeled, artifact)
