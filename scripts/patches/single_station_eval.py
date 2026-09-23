import sys
import pandas as pd
import joblib
from pathlib import Path
from model.engine import DecisionEngine
import time

def run_single():
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    data_dir = project_root / "data"
    artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")
    
    # Load target
    df = pd.read_csv(data_dir / "AWS-CHN-024_labeled.csv", parse_dates=["timestamp"])
    df["__source_file"] = "AWS-CHN-024_labeled.csv"
    df = df.sort_values("timestamp")
    
    # Load neighbors
    neighbors = {}
    for nid in ["AWS-CHN-101", "AWS-CHN-102", "AWS-CHN-103"]:
        ndf = pd.read_csv(data_dir / f"{nid}.csv", parse_dates=["timestamp"])
        ndf["__source_file"] = f"{nid}.csv"
        neighbors[nid] = ndf.sort_values("timestamp")
        
    print(f"Loaded. Evaluating {len(df)} rows...")
    
    history_df = pd.DataFrame()
    nbufs = {nid: pd.DataFrame() for nid in neighbors}
    
    results = []
    
    start = time.time()
    for i, row in df.iterrows():
        raw = row.to_dict()
        ts = raw["timestamp"]
        
        # update target history
        current_row = dict(raw, station_id="AWS-CHN-024")
        if history_df.empty:
            history_df = pd.DataFrame([current_row])
        else:
            history_df = pd.concat([history_df, pd.DataFrame([current_row])], ignore_index=True)
            
        # keep only last 72 hours
        history_df = history_df.tail(1440)
        
        # update neighbor history
        for nid, ndf in neighbors.items():
            n_row = ndf[ndf["timestamp"] == ts]
            if not n_row.empty:
                n_dict = n_row.iloc[0].to_dict()
                n_dict["station_id"] = nid
                if nbufs[nid].empty:
                    nbufs[nid] = pd.DataFrame([n_dict])
                else:
                    nbufs[nid] = pd.concat([nbufs[nid], pd.DataFrame([n_dict])], ignore_index=True)
                nbufs[nid] = nbufs[nid].tail(1440)
                
        verdict = DecisionEngine.decide(raw, history_df, nbufs, artifact, state=None)
        
        pred_item = {
            "timestamp": ts,
            "station_id": "AWS-CHN-024",
            "is_anomaly_gt": raw.get("is_anomaly", False),
            "fault_type_gt": raw.get("fault_type", "none"),
            "is_anomaly_pred": verdict["is_anomaly"],
            "fault_type_pred": verdict["fault_type"] or "none",
        }
        results.append(pred_item)
        
        if i % 100 == 0:
            print(f"Row {i}/{len(df)}")
            
    res_df = pd.DataFrame(results)
    
    # Calculate metrics
    print("\\n=== SINGLE STATION METRICS ===")
    pred = res_df["is_anomaly_pred"]
    gt = res_df["is_anomaly_gt"].fillna(False).astype(bool)
    
    pred_anom = pred.sum()
    caught = (pred & gt).sum()
    true_anom = gt.sum()
    precision = caught / pred_anom if pred_anom else 0
    recall = caught / true_anom if true_anom else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    print(f"Overall Recall:    {recall:.1%} ({caught}/{true_anom})")
    print(f"Overall Precision: {precision:.1%} ({caught}/{pred_anom})")
    
    print("\\nBy Fault Type:")
    faults = res_df[res_df["fault_type_gt"] != "none"]["fault_type_gt"].unique()
    for ft in sorted(faults):
        if pd.isna(ft): continue
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
    run_single()
