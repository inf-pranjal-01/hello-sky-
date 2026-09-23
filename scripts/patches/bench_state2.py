import pandas as pd
import time
import pickle
from pathlib import Path
from model.state import StateManager

ARTIFACTS_PATH = Path("model_artifacts/isolation_forest.pkl")

def bench():
    df = pd.read_csv("data/AWS-CHN-024_labeled.csv").head(2000)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    
    with open(ARTIFACTS_PATH, "rb") as f:
        artifact = pickle.load(f)
        
    state_manager = StateManager(artifact)
    state_manager.reset_all_buffers()
    
    start = time.time()
    
    count = 0
    for _, row in df.iterrows():
        raw = row.to_dict()
        res = state_manager.ingest_reading(raw)
        count += 1
        
    end = time.time()
    print(f"Processed {count} rows in {end - start:.2f} seconds")
    print(f"Estimated for 60k rows: {(end - start) * 30:.2f} seconds")

if __name__ == "__main__":
    bench()
