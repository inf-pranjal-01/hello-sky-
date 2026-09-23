import pandas as pd
import time
import pickle
from pathlib import Path
from model.features import build_feature_matrix
from model.detect import _model_score_to_pct, _rule_checks, _corroborate_network, _fuse_and_score

ARTIFACTS_PATH = Path("model_artifacts/isolation_forest.pkl")

def bench():
    df = pd.read_csv("data/AWS-CHN-024_labeled.csv").head(1000)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    
    with open(ARTIFACTS_PATH, "rb") as f:
        artifact = pickle.load(f)
        
    start = time.time()
    
    # Pre-featurize (simulating evaluate.py)
    featured = build_feature_matrix(df)
    
    count = 0
    for i in range(len(df)):
        raw = df.iloc[i].to_dict()
        feat = featured.iloc[i]
        
        hist = df.iloc[max(0, i-24):i+1] # 24 rows
        
        # Simulated decide_from_features
        model_pct, _ = _model_score_to_pct(raw, feat, artifact, hist)
        rules = _rule_checks(raw, feat, hist, artifact)
        # We skip _corroborate_network for bench simplicity or mock it
        res = _fuse_and_score(model_pct, rules["fired"])
        count += 1
        
    end = time.time()
    print(f"Processed {count} rows in {end - start:.2f} seconds")
    print(f"Estimated for 60k rows: {(end - start) * 60:.2f} seconds")

if __name__ == "__main__":
    bench()
