import pandas as pd
import time
from model.state import StateManager

def mock_load():
    class DummyModel:
        def predict_proba(self, X):
            import numpy as np
            return np.zeros((len(X), 2))
    return {
        "model": DummyModel(),
        "rule_thresholds": {
            "spike_deviation": 1.0,
            "multivariate_dev": 1.0,
            "vapor_pressure": 1.0
        }
    }

def bench():
    df = pd.read_csv("data/AWS-CHN-024_labeled.csv").head(1000)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    
    artifact = mock_load()
        
    state_manager = StateManager(artifact)
    state_manager.reset_all_buffers()
    
    start = time.time()
    
    count = 0
    for _, row in df.iterrows():
        raw = row.to_dict()
        try:
            res = state_manager.ingest_reading(raw)
        except Exception as e:
            pass # ignore errors from mock artifact
        count += 1
        
    end = time.time()
    print(f"Processed {count} rows in {end - start:.2f} seconds")
    print(f"Estimated for 60k rows: {(end - start) * 60:.2f} seconds")

if __name__ == "__main__":
    bench()
