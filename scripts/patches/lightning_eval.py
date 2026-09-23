import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from model.engine import DecisionEngine
from model.features import build_feature_matrix

def run_fast_true_eval():
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    data_dir = project_root / "data"
    artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")
    
    # 1. Load data
    dfs = []
    for nid in ["AWS-CHN-024_labeled.csv", "AWS-CHN-101.csv", "AWS-CHN-102.csv", "AWS-CHN-103.csv"]:
        if not (data_dir / nid).exists(): continue
        df = pd.read_csv(data_dir / nid, parse_dates=["timestamp"])
        df["station_id"] = nid.replace("_labeled.csv", "").replace(".csv", "")
        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = False
        dfs.append(df)
        
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)
    
    df_feat = build_feature_matrix(df_all)
    df_feat = df_feat.sort_values("timestamp").reset_index(drop=True)
    
    target_df = df_feat[df_feat["station_id"] == "AWS-CHN-024"].copy().reset_index(drop=True)
    peers_df = df_feat[df_feat["station_id"] != "AWS-CHN-024"].copy()
    
    print(f"Loaded. Evaluating {len(target_df)} rows at lightning speed...")
    
    results = []
    RAW_PARAMS = ["temperature_c", "pressure_hpa", "humidity_pct"]
    
    for i in range(72, len(target_df)):
        raw = target_df.iloc[i].to_dict()
        ts = raw["timestamp"]
        feat_row = target_df.iloc[i]
        
        start_idx = max(0, i - 1440)
        
        history_df = target_df.iloc[start_idx : i][RAW_PARAMS + ["timestamp", "station_id", "is_anomaly", "fault_type"]].copy()
        precomp_history_feat = target_df.iloc[start_idx : i].copy()
        
        precomp_n = {}
        nbufs = {}
        for nid in ["AWS-CHN-101", "AWS-CHN-102", "AWS-CHN-103"]:
            ndf = peers_df[(peers_df["station_id"] == nid) & (peers_df["timestamp"] <= ts)]
            if not ndf.empty:
                precomp_n[nid] = ndf.iloc[-1]
                n_start = max(0, len(ndf) - 1440)
                nbufs[nid] = ndf.iloc[n_start:].copy()[RAW_PARAMS + ["timestamp", "station_id"]]
                
        verdict = DecisionEngine.decide(
            raw, 
            history_df, 
            nbufs, 
            artifact, 
            state=None,
            precomputed_features=feat_row,
            precomputed_neighbors=precomp_n,
            precomputed_history_featured=precomp_history_feat
        )
        
        results.append({
            "is_anomaly_gt": raw.get("is_anomaly", False),
            "fault_type_gt": raw.get("fault_type", "none"),
            "is_anomaly_pred": verdict["is_anomaly"],
            "fault_type_pred": verdict["fault_type"] or "none",
        })
        
    res_df = pd.DataFrame(results)
    
    print("\\n=== LIGHTNING SCORECARD METRICS (AWS-CHN-024) ===")
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
