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
df_feat = build_feature_matrix(df_all)
target_df = df_feat[df_feat["station_id"] == "AWS-CHN-024"].copy().reset_index(drop=True)
artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")

# Find first drift
mask = target_df["fault_type"] == "drift"
drift_idx = target_df.index[mask][15]  # middle of a drift

start_idx = max(0, drift_idx - 1440)
history_df = target_df.iloc[start_idx : drift_idx+1]
raw = history_df.iloc[-1].to_dict()

# Mock peers
precomp_n = {}
nbufs = {}

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
print("Verdict:", verdict)
