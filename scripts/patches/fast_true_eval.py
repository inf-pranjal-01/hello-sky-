import sys
import pandas as pd
import joblib
from pathlib import Path
from model.engine import DecisionEngine

def run_fast_true_eval():
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    data_dir = project_root / "data"
    artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")
    
    # Load target
    df = pd.read_csv(data_dir / "AWS-CHN-024_labeled.csv", parse_dates=["timestamp"])
    df["__source_file"] = "AWS-CHN-024_labeled.csv"
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Load neighbors
    neighbors = {}
    for nid in ["AWS-CHN-101", "AWS-CHN-102", "AWS-CHN-103"]:
        ndf = pd.read_csv(data_dir / f"{nid}.csv", parse_dates=["timestamp"])
        ndf["__source_file"] = f"{nid}.csv"
        neighbors[nid] = ndf.sort_values("timestamp").reset_index(drop=True)
        
    print(f"Loaded. Evaluating {len(df)} rows...")
    
    results = []
    
    for i in range(72, len(df)):
        raw = df.iloc[i].to_dict()
        ts = raw["timestamp"]
        
        start_idx = max(0, i - 1440)
        history_df = df.iloc[start_idx : i].copy()
        history_df["station_id"] = "AWS-CHN-024"
        
        nbufs = {}
        for nid, ndf in neighbors.items():
            nbufs[nid] = ndf.iloc[start_idx : i].copy()
            nbufs[nid]["station_id"] = nid
                
        verdict = DecisionEngine.decide(raw, history_df, nbufs, artifact, state=None)
        
        pred_item = {
            "timestamp": ts,
            "is_anomaly_gt": raw.get("is_anomaly", False),
            "fault_type_gt": raw.get("fault_type", "none"),
            "is_anomaly_pred": verdict["is_anomaly"],
            "fault_type_pred": verdict["fault_type"] or "none",
        }
        results.append(pred_item)
            
    res_df = pd.DataFrame(results)
    
    print("\\n=== TRUE SCORECARD METRICS (AWS-CHN-024) ===")
    pred = res_df["is_anomaly_pred"]
    gt = res_df["is_anomaly_gt"].fillna(False).astype(bool)
    
    pred_anom = pred.sum()
    caught = (pred & gt).sum()
    true_anom = gt.sum()
    precision = caught / pred_anom if pred_anom else 0
    recall = caught / true_anom if true_anom else 0
    
    print(f"Overall Recall:    {recall:.1%} ({caught}/{true_anom})")
    print(f"Overall Precision: {precision:.1%} ({caught}/{pred_anom})")
    
    print("\\nBy Fault Type:")
    faults = [f for f in res_df["fault_type_gt"].unique() if isinstance(f, str) and f != "none"]
    for ft in sorted(faults):
        mask_gt = (res_df["fault_type_gt"] == ft)
        t = mask_gt.sum()
        c = (mask_gt & pred).sum()
        
        mask_pred_ft = (res_df["fault_type_pred"] == ft)
        p_class = mask_pred_ft.sum()
        c_class = (mask_pred_ft & mask_gt).sum()
        
        rec = c / t if t else 0
        prec = c_class / p_class if p_class else 0
        print(f"  {ft:<28} Caught: {c}/{t} ({rec:.1%}) | Precision: {prec:.1%}")

if __name__ == "__main__":
    run_fast_true_eval()
