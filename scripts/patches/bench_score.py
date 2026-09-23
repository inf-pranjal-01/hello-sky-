import pandas as pd
import time
import os
import pickle
from pathlib import Path

# Local path since config doesn't have it
ARTIFACTS_PATH = Path("model_artifacts/isolation_forest.pkl")

from model.detect import score_reading

def bench():
    df = pd.read_csv("data/AWS-CHN-024_labeled.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    
    with open(ARTIFACTS_PATH, "rb") as f:
        artifact = pickle.load(f)
        
    start = time.time()
    
    buffer = []
    count = 0
    
    for _, row in df.head(100).iterrows():
        raw = row.to_dict()
        buffer.append(raw)
        if len(buffer) > 24:
            buffer.pop(0)
            
        hist_df = pd.DataFrame(buffer)
        
        # Call the unified detection engine core (detect.py's score_reading)
        res = score_reading(raw, hist_df, artifact, neighbor_buffers={})
        count += 1
        
    end = time.time()
    print(f"Processed {count} rows in {end - start:.2f} seconds")
    print(f"Estimated for 60k rows: {(end - start) * 600:.2f} seconds")

if __name__ == "__main__":
    bench()
