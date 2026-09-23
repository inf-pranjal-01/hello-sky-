import pandas as pd
from model.detect import load_model, score_reading
from model.state import StateManager

def test():
    artifact = load_model()
    df = pd.read_csv("data/AWS-MUM-007.csv", parse_dates=["timestamp"])
    
    # Take first 100 hours (roughly 4 days)
    df = df.head(100)
    
    # Read metadata
    metadata = pd.read_csv("data/stations_metadata.csv")
    
    state = StateManager(metadata, artifact)
    
    for i, row in df.iterrows():
        reading = row.to_dict()
        reading["timestamp"] = reading["timestamp"].isoformat()
        
        # StateManager handles history persistence, buffer updates, and calls detect.py itself!
        verdict = state.ingest_reading(reading)
        
        if verdict["is_anomaly"]:
            print(f"Anomaly at {reading['timestamp']}: "
                  f"Score={verdict['anomaly_score_pct']}% "
                  f"Model={verdict['model_confidence_pct']}% "
                  f"Rules={verdict['rules_fired']} "
                  f"Fault={verdict['fault_type']}")

if __name__ == "__main__":
    test()
