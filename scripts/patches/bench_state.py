import pandas as pd
import time
import pickle
from pathlib import Path
from model.state import StateManager

ARTIFACTS_PATH = Path("model_artifacts/isolation_forest.pkl")

def bench():
    df = pd.read_csv("data/AWS-CHN-024_labeled.csv").head(1000)
    
    with open(ARTIFACTS_PATH, "rb") as f:
        artifact = pickle.load(f)
        
    state_manager = StateManager(artifact, mock_history_store=True)
    state_manager.reset_all_buffers()
    
    start = time.time()
    
    count = 0
    for i in range(len(df)):
        raw = df.iloc[i].to_dict()
        res = state_manager.ingest_reading(raw)
        count += 1
        
    end = time.time()
    print(f"Processed {count} rows in {end - start:.2f} seconds")
    print(f"Estimated for 60k rows: {(end - start) * 60:.2f} seconds")

if __name__ == "__main__":
    bench()
