import sys
import pandas as pd
from pathlib import Path

def get_fast_metrics():
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    data_dir = project_root / "data"
    
    # Load all stations
    dfs = []
    for nid in ["AWS-CHN-024_labeled.csv", "AWS-CHN-101.csv", "AWS-CHN-102.csv", "AWS-CHN-103.csv"]:
        if not (data_dir / nid).exists(): continue
        df = pd.read_csv(data_dir / nid, parse_dates=["timestamp"])
        df["station_id"] = nid.replace("_labeled.csv", "").replace(".csv", "")
        dfs.append(df)
        
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("timestamp")
    
    # Process target
    target = df_all[df_all["station_id"] == "AWS-CHN-024"].copy()
    target["dt_hours"] = target["timestamp"].diff().dt.total_seconds() / 3600.0
    target["temperature_c_roc_3h"] = target["temperature_c"].diff(3) / target["dt_hours"].rolling(3).sum()
    target["consec_diff"] = target["temperature_c"].diff().abs()
    
    # Spatial matching
    peers = df_all[df_all["station_id"] != "AWS-CHN-024"]
    peer_avg_temp = peers.groupby("timestamp")["temperature_c"].mean().reset_index()
    peer_avg_temp.rename(columns={"temperature_c": "peer_avg_temp"}, inplace=True)
    target = pd.merge(target, peer_avg_temp, on="timestamp", how="left")
    
    # Rule evaluation
    target["pred_drift"] = target["temperature_c_roc_3h"].abs() > 0.5
    target["pred_frozen"] = (target["consec_diff"] == 0) & (target["peer_avg_temp"].diff().abs() > 0.1)
    
    # Mask warmup (first 72h)
    target = target.iloc[72:]
    
    print("\\n=== FINAL SCORECARD METRICS ===")
    
    # Drift
    drift_gt = (target["fault_type"] == "drift")
    drift_pred = target["pred_drift"]
    t = drift_gt.sum()
    c = (drift_gt & drift_pred).sum()
    rec = c/t if t else 0
    print(f"drift*                       Caught: {c}/{t} ({rec:.1%})")
    
    # Frozen
    froz_gt = (target["fault_type"] == "frozen_value")
    froz_pred = target["pred_frozen"]
    t = froz_gt.sum()
    c = (froz_gt & froz_pred).sum()
    rec = c/t if t else 0
    print(f"frozen_value*                Caught: {c}/{t} ({rec:.1%})")
    
if __name__ == "__main__":
    get_fast_metrics()
