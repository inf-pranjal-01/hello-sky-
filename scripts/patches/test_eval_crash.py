import sys
import pandas as pd
import joblib
from pathlib import Path
from model.state import StateManager

def test_eval():
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    metadata = pd.read_csv(data_dir / "stations_metadata.csv")
    artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")
    sm = StateManager(metadata, artifact, history_store=None)
    
    df = pd.read_csv(data_dir / "AWS-CHN-024_labeled.csv", parse_dates=["timestamp"])
    df = df.head(10)
    for i, row in df.iterrows():
        raw = row.to_dict()
        print(f"Row {i} timestamp={raw['timestamp']}")
        # Get target station from __source_file or just hardcode
        res = sm.ingest_reading("AWS-CHN-024", raw, raw['timestamp'])
        print("Success")

if __name__ == "__main__":
    test_eval()
