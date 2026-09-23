import sys
import pandas as pd
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import scripts.patches.lightning_eval as ue
from model.features import build_feature_matrix

df_all = ue.pd.read_csv(project_root/"data/AWS-CHN-024_labeled.csv", parse_dates=["timestamp"])
df_all["station_id"] = "AWS-CHN-024"
df_feat = build_feature_matrix(df_all)

for ft in ["frozen_value", "drift", "sensor_fail_low"]:
    mask = df_feat["fault_type"] == ft
    if not mask.any(): continue
    print(f"--- {ft} ---")
    print("temp_range_3h mean:", df_feat.loc[mask, "temp_range_3h"].mean())
    print("temp_range_3h min:", df_feat.loc[mask, "temp_range_3h"].min())
    print("temp_range_3h 50th:", df_feat.loc[mask, "temp_range_3h"].median())
