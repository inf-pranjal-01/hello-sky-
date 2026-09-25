import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from evaluation.fast_offline_eval import evaluate_all, ARTIFACTS_PATH
from data.anomaly_injector import generate_network_benchmark

artifact = joblib.load(ARTIFACTS_PATH)

def test_dev_seeds():
    for seed in [42, 101, 202]:
        data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
        res = evaluate_all(data, artifact, silent=True)
        m = res['__overall__']
        ep = res['__episodic__']
        print(f"Seed {seed}: Precision={m['precision']*100:.2f}%, Recall={m['recall']*100:.2f}%, F1*={ep.latency_aware_f1:.4f}, Episodes={ep.detected_episodes}/{ep.total_episodes}")

if __name__ == '__main__':
    test_dev_seeds()
