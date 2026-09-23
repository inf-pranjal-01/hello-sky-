import sys
import pandas as pd
import joblib
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from model.features import add_temporal_features, add_cross_parameter_features, add_time_features, add_rule_only_signals, build_features_for_latest
from config import MODEL_FEATURE_COLS

data_dir = project_root / "data"
artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")

df = pd.read_csv(data_dir / "AWS-CHN-024_labeled.csv", parse_dates=["timestamp"])
df["station_id"] = "AWS-CHN-024"

# Just process 1440 rows
df_hist = df.iloc[:1440].copy()
feat = build_features_for_latest(df_hist)

try:
    ml_feat = feat[MODEL_FEATURE_COLS].to_frame().T
    score = artifact["model"].decision_function(ml_feat)[0]
    print("SUCCESS", score)
except Exception as e:
    print("FAIL:", type(e), e)
