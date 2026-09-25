import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark
from evaluation.fast_offline_eval import evaluate_all, ARTIFACTS_PATH

artifact = joblib.load(ARTIFACTS_PATH)

def test_system():
    print("Testing system...")

if __name__ == '__main__':
    test_system()
