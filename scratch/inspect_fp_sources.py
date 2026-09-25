import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark

STATION_TO_CLUSTER = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    for sid in [center] + neighbors:
        STATION_TO_CLUSTER[sid] = cid

def inspect_fp_sources(seed=42):
    benchmark_b = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    # Check clean stations vs target stations
    clean_stations = [s for s, df in benchmark_b.items() if df['is_anomaly'].sum() == 0]
    print(f"Clean stations count: {len(clean_stations)}")

if __name__ == '__main__':
    inspect_fp_sources(42)
