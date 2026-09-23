import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.features import add_temporal_features, add_cross_parameter_features, add_time_features, add_rule_only_signals
from model.engine import DecisionEngine

RAW_PARAMS = ["temperature_c", "pressure_hpa", "humidity_pct"]

def run_ultimate_eval():
    data_dir = project_root / "data"
    artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")
    
    dfs = []
    for nid in ["AWS-CHN-024_labeled.csv", "AWS-CHN-101.csv", "AWS-CHN-102.csv", "AWS-CHN-103.csv"]:
        if not (data_dir / nid).exists(): continue
        df = pd.read_csv(data_dir / nid, parse_dates=["timestamp"])
        df["station_id"] = nid.replace("_labeled.csv", "").replace(".csv", "")
        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = False
        dfs.append(df)
        
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("timestamp")
    
    df_feats = []
    for nid, df_nid in df_all.groupby("station_id"):
        df_nid = df_nid.copy().sort_values("timestamp")
        
        df_clean = df_nid.copy()
        anomaly_mask = df_clean["is_anomaly"] == True
        for param in RAW_PARAMS:
            df_clean.loc[anomaly_mask, param] = np.nan
            
        df_clean = add_temporal_features(df_clean)
        
        for param in RAW_PARAMS:
            df_clean[param] = df_nid[param]
            
        df_feat = add_temporal_features(df_nid.copy())
        for col in df_clean.columns:
            if col.endswith("_baseline") or col.endswith("_std"):
                df_feat[col] = df_clean[col]
                
        df_feat = add_cross_parameter_features(df_feat)
        df_feat = add_time_features(df_feat)
        df_feat = add_rule_only_signals(df_feat)
        df_feats.append(df_feat)
        
    df_all = pd.concat(df_feats, ignore_index=True).sort_values("timestamp")
    
    target_df = df_all[df_all["station_id"] == "AWS-CHN-024"].copy().reset_index(drop=True)
    peers_df = df_all[df_all["station_id"] != "AWS-CHN-024"].copy()
    
    results = []
    for i in range(72, len(target_df)):
        raw = target_df.iloc[i].to_dict()
        ts = raw["timestamp"]
        feat_row = target_df.iloc[i]
        
        precomp_n = {}
        for nid in ["AWS-CHN-101", "AWS-CHN-102", "AWS-CHN-103"]:
            ndf = peers_df[(peers_df["station_id"] == nid) & (peers_df["timestamp"] <= ts)]
            if not ndf.empty:
                precomp_n[nid] = ndf.iloc[-1]
                
        history_df = pd.DataFrame([raw])
        nbufs = {nid: pd.DataFrame([raw]) for nid in precomp_n.keys()}
        
        verdict = DecisionEngine.decide(
            raw, 
            history_df, 
            nbufs, 
            artifact, 
            state=None,
            precomputed_features=feat_row,
            precomputed_neighbors=precomp_n
        )
        
        results.append({
            "is_anomaly_gt": raw.get("is_anomaly", False),
            "fault_type_gt": raw.get("fault_type", "none"),
            "is_anomaly_pred": verdict["is_anomaly"],
            "fault_type_pred": verdict["fault_type"] or "none",
        })
        
    res_df = pd.DataFrame(results)
    
    print("\\n=== ULTIMATE SCORECARD METRICS (AWS-CHN-024) ===")
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
        rec = c / t if t else 0
        print(f"  {ft:<28} Caught: {c}/{t} ({rec:.1%})")

if __name__ == "__main__":
    run_ultimate_eval()
