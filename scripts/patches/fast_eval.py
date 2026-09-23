import sys
import pandas as pd
import time
import joblib
from pathlib import Path

# Fix paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.evaluate import evaluate_all

def run_sample():
    data_dir = project_root / "data"
    # Only evaluate one injected station
    labeled_files = [data_dir / "AWS-CHN-024_labeled.csv"]
    
    # Also load its clean neighbors to make corroboration work
    for nid in ["AWS-CHN-101", "AWS-CHN-102", "AWS-CHN-103"]:
        labeled_files.append(data_dir / f"{nid}.csv")
        
    artifact = joblib.load(project_root / "model_artifacts" / "isolation_forest.pkl")
    
    print("Running evaluate_all on subset...")
    res = evaluate_all(labeled_files, artifact)
    print("Done")

if __name__ == "__main__":
    run_sample()
