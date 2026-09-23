import sys
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from model.engine import DecisionEngine
from model.detect import _model_score_to_pct

artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")

import scripts.patches.ultimate_eval as ue
df_all = ue.pd.read_csv(project_root/"data/AWS-CHN-024_labeled.csv", parse_dates=["timestamp"])
df_all["station_id"] = "AWS-CHN-024"

# Use the EXACT way the actual evaluate.py builds features!
from model.features import build_features_for_latest

# find a drift
drift_idx = df_all[df_all["fault_type"] == "drift"].index[0]
raw = df_all.iloc[drift_idx].to_dict()

# history buffer (0 to drift_idx)
history_df = df_all.iloc[:drift_idx+1].copy()

# 1. How does the live engine score it?
feat_live = build_features_for_latest(history_df)
pct_live, stat_live = _model_score_to_pct(raw, feat_live, artifact, history_df)
print("LIVE PCT:", pct_live, "STATUS:", stat_live)

# 2. How did ultimate_eval score it?
df_feats = []
df_nid = df_all.copy()
df_feat = ue.add_temporal_features(df_nid.copy())
df_feat = ue.add_cross_parameter_features(df_feat)
df_feat = ue.add_time_features(df_feat)
df_feat = ue.add_rule_only_signals(df_feat)

feat_pre = df_feat.iloc[drift_idx]
pct_pre, stat_pre = _model_score_to_pct(raw, feat_pre, artifact, history_df)
print("PRE PCT:", pct_pre, "STATUS:", stat_pre)

