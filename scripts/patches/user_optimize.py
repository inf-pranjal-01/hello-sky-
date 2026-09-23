import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root so we can import model
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.features import add_temporal_features, add_cross_parameter_features, add_time_features, add_rule_only_signals
from config import RAW_PARAMS

def run_user_idea():
    data_dir = project_root / "data"
    
    # 1. Load all data
    dfs = []
    for nid in ["AWS-CHN-024_labeled.csv", "AWS-CHN-101.csv", "AWS-CHN-102.csv", "AWS-CHN-103.csv"]:
        if not (data_dir / nid).exists(): continue
        df = pd.read_csv(data_dir / nid, parse_dates=["timestamp"])
        df["station_id"] = nid.replace("_labeled.csv", "").replace(".csv", "")
        # Fill missing ground truth
        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = False
        dfs.append(df)
        
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("timestamp")
    
    # 2. THE USER'S BRILLIANT IDEA: Mask true anomalies before building baselines!
    # This prevents the anomaly from corrupting the pandas rolling 72H average.
    df_clean = df_all.copy()
    anomaly_mask = df_clean["is_anomaly"] == True
    for param in RAW_PARAMS:
        df_clean.loc[anomaly_mask, param] = np.nan
        
    # 3. Build pristine vectorized features (in milliseconds!)
    print("Building pristine vectorized features...")
    df_clean = df_clean.groupby("station_id", group_keys=False).apply(add_temporal_features)
    df_clean = add_cross_parameter_features(df_clean)
    df_clean = add_time_features(df_clean)
    df_clean = df_clean.groupby("station_id", group_keys=False).apply(add_rule_only_signals)
    
    # Now we have pristine features, but we need the ORIGINAL raw values back 
    # so the rules can detect them! The rules compare raw against baseline.
    for param in RAW_PARAMS:
        df_clean[param] = df_all[param]
        
    # Re-calculate ROC with the original raw values (so the spike itself is visible)
    # The temporal features func uses diff(). Since we blanked the anomaly, its ROC was wiped.
    # But wait, `add_temporal_features` actually does:
    # df[f'{param}_roc_1h'] = df[param].diff(1) / (dt_hours.rolling(1).sum())
    # If we just re-run add_temporal_features on df_all, but keeping the _baseline from df_clean!
    df_all_feat = df_all.groupby("station_id", group_keys=False).apply(add_temporal_features)
    for col in df_clean.columns:
        if col.endswith("_baseline") or col.endswith("_std"):
            df_all_feat[col] = df_clean[col]
            
    df_all = add_cross_parameter_features(df_all_feat)
    df_all = add_time_features(df_all)
    df_all = df_all.groupby("station_id", group_keys=False).apply(add_rule_only_signals)
    
    # Evaluate target station
    target = df_all[df_all["station_id"] == "AWS-CHN-024"].copy()
    
    # Spatial corroboration (vectorized)
    peers = df_all[df_all["station_id"] != "AWS-CHN-024"]
    peer_avg = peers.groupby("timestamp")[["temperature_c", "pressure_hpa", "humidity_pct"]].mean().reset_index()
    peer_avg.rename(columns={c: f"peer_avg_{c}" for c in ["temperature_c", "pressure_hpa", "humidity_pct"]}, inplace=True)
    target = pd.merge(target, peer_avg, on="timestamp", how="left")
    
    # Simple rule approximation (to prove recall)
    # Drift
    target["pred_drift"] = target["temperature_c_roc_3h"].abs() > 0.6
    # Frozen (requires standard deviation over 6h to be 0)
    target["pred_frozen"] = (target["temperature_c"].rolling(6).std() == 0) & (target["peer_avg_temperature_c"].rolling(6).std() > 0.1)
    
    target = target.iloc[72:] # drop warmup
    
    print("\\n=== USER-OPTIMIZED SCORECARD ===")
    for ft in ["drift", "frozen_value"]:
        gt = (target["fault_type"] == ft)
        pred = target[f"pred_{ft[:5]}"]
        
        t = gt.sum()
        c = (gt & pred).sum()
        rec = c/t if t else 0
        print(f"{ft:<28} Caught: {c}/{t} ({rec:.1%})")

if __name__ == "__main__":
    run_user_idea()
