import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from config import CLUSTERS, CUSUM_DRIFT_ALLOWANCE
from data.anomaly_injector import generate_network_benchmark

# Let's test CUSUM on diurnal residual across seeds 42, 101, 202
for seed in [42, 101, 202]:
    benchmark_b = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    tp_drift, fp_drift, total_drift = 0, 0, 0
    for sid, df in benchmark_b.items():
        is_drift_gt = (df['fault_type'] == 'drift')
        total_drift += is_drift_gt.sum()
        
    print(f"Seed {seed}: Total ground-truth drift rows = {total_drift}")
