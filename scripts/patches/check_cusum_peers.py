import sys
import pandas as pd
from pathlib import Path
import joblib
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import scripts.patches.lightning_eval as ue
from model.features import build_feature_matrix
from model.engine import DecisionEngine

df_all = ue.pd.read_csv(project_root/"data/AWS-CHN-024_labeled.csv", parse_dates=["timestamp"])
df_all["station_id"] = "AWS-CHN-024"
dfs = [df_all]
for nid in ["AWS-CHN-101", "AWS-CHN-102", "AWS-CHN-103"]:
    df = ue.pd.read_csv(project_root/f"data/{nid}.csv", parse_dates=["timestamp"])
    df["station_id"] = nid
    dfs.append(df)
df_all_peers = pd.concat(dfs, ignore_index=True)
df_feat = build_feature_matrix(df_all_peers)

target_df = df_feat[df_feat["station_id"] == "AWS-CHN-024"].copy().reset_index(drop=True)
peers_df = df_feat[df_feat["station_id"] != "AWS-CHN-024"].copy()

artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")

# Find first drift
mask = target_df["fault_type"] == "drift"
drift_idx = target_df.index[mask][15]  # middle of a drift

start_idx = max(0, drift_idx - 1440)
history_df = target_df.iloc[start_idx : drift_idx+1]
raw = history_df.iloc[-1].to_dict()
ts = raw["timestamp"]

# Mock peers
precomp_n = {}
nbufs = {}
for nid in ["AWS-CHN-101", "AWS-CHN-102", "AWS-CHN-103"]:
    ndf = peers_df[(peers_df["station_id"] == nid) & (peers_df["timestamp"] <= ts)]
    if not ndf.empty:
        precomp_n[nid] = ndf.iloc[-1]
        n_start = max(0, len(ndf) - 1440)
        nbufs[nid] = ndf.iloc[n_start:].copy()[["temperature_c", "pressure_hpa", "humidity_pct", "timestamp", "station_id"]]

verdict = DecisionEngine.decide(
    raw, 
    history_df, 
    nbufs, 
    artifact, 
    state=None,
    precomputed_features=target_df.iloc[drift_idx],
    precomputed_neighbors=precomp_n,
    precomputed_history_featured=history_df
)
print("Verdict with peers:", verdict)
