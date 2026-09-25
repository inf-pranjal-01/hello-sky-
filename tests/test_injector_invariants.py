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

def test_cluster_invariants_across_seeds(seeds=[42, 101, 202, 1101, 2024, 8888, 20260924]):
    print(f"Testing cluster-level concurrency invariant (<=1 faulty station/cluster/timestep) across {len(seeds)} seeds...")
    
    for seed in seeds:
        data = generate_network_benchmark(regime="benchmark_b", seed=seed, save_to_disk=False)
        
        # Build matrix of is_anomaly per station per timestamp
        station_series = {}
        for sid, df in data.items():
            station_series[sid] = df["is_anomaly"].fillna(False).astype(bool)
            
        full_df = pd.DataFrame(station_series)
        
        for cid, cinfo in CLUSTERS.items():
            cluster_stations = [cinfo["center"]["station_id"]] + [n["station_id"] for n in cinfo["neighbors"]]
            cluster_subset = full_df[[s for s in cluster_stations if s in full_df.columns]]
            
            # Count concurrent active faults per timestep in this cluster
            active_counts = cluster_subset.sum(axis=1)
            max_active = active_counts.max()
            
            assert max_active <= 1, f"FAILED: Seed {seed}, Cluster {cid} had {max_active} simultaneous faulty stations at timestep {active_counts.idxmax()}!"
            
        print(f"  [PASS] Seed {seed:<10}: Invariant strictly holds! Max active faulty stations per cluster <= 1.")
        
    print("\nALL INJECTOR INVARIANT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_cluster_invariants_across_seeds()
