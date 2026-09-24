"""
Tests for SkyGuard AI — PCL-Compatible Fault Model Invariant & Reproducibility.
Verifies max_faults_per_cluster_at_any_timestamp <= 1 under Benchmark B.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from config import CLUSTERS
from data.anomaly_injector import (
    inject_anomalies,
    generate_network_benchmark,
    RANDOM_SEED,
    DATA_DIR,
)

# Build station to cluster map
STATION_TO_CLUSTER = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    for sid in [center] + neighbors:
        STATION_TO_CLUSTER[sid] = cid


class TestPCLCompatibleFaultModel:

    @pytest.mark.parametrize("seed", [RANDOM_SEED, 42, 101, 2024, 8888])
    def test_cluster_concurrency_invariant_benchmark_b(self, seed):
        """Verify that at any evaluation timestamp, every cluster has at most 1 injected fault."""
        benchmark_b = generate_network_benchmark(
            regime="benchmark_b", seed=seed, save_to_disk=False
        )

        frames = []
        for sid, df in benchmark_b.items():
            sub = df[["timestamp", "is_anomaly", "fault_type"]].copy()
            sub["station_id"] = sid
            sub["cluster_id"] = STATION_TO_CLUSTER.get(sid, "UNKNOWN")
            frames.append(sub)
        combined = pd.concat(frames, ignore_index=True)

        anom = combined[combined["is_anomaly"] == True]
        cluster_concurrency = anom.groupby(["cluster_id", "timestamp"])["station_id"].nunique()

        max_concurrent = cluster_concurrency.max() if len(cluster_concurrency) > 0 else 0
        violations = cluster_concurrency[cluster_concurrency > 1]

        assert max_concurrent <= 1, f"Seed {seed} violated cluster concurrency: max={max_concurrent}"
        assert len(violations) == 0, f"Seed {seed} had {len(violations)} concurrent fault timestamps"

    def test_seed_reproducibility(self):
        """Verify that identical seeds produce bitwise identical benchmarks."""
        run1 = generate_network_benchmark(regime="benchmark_b", seed=42, save_to_disk=False)
        run2 = generate_network_benchmark(regime="benchmark_b", seed=42, save_to_disk=False)

        for sid in run1:
            assert sid in run2
            pd.testing.assert_frame_equal(run1[sid], run2[sid])

    def test_seed_diversity(self):
        """Verify that different seeds produce distinct, varied fault realizations."""
        run1 = generate_network_benchmark(regime="benchmark_b", seed=101, save_to_disk=False)
        run2 = generate_network_benchmark(regime="benchmark_b", seed=202, save_to_disk=False)

        differences = 0
        for sid in run1:
            if not run1[sid]["is_anomaly"].equals(run2[sid]["is_anomaly"]):
                differences += 1
        assert differences > 0, "Different seeds produced identical anomaly masks"

    def test_causal_timestamp_ordering(self):
        """Verify that timestamps in generated datasets are monotonically strictly increasing."""
        benchmark_b = generate_network_benchmark(
            regime="benchmark_b", seed=RANDOM_SEED, save_to_disk=False
        )
        for sid, df in benchmark_b.items():
            ts = pd.to_datetime(df["timestamp"])
            assert ts.is_monotonic_increasing, f"Timestamps not monotonic for station {sid}"
            assert df["timestamp"].duplicated().sum() == 0, f"Duplicate timestamps in {sid}"

    def test_benchmark_a_preserves_unrestricted_behavior(self):
        """Verify that Benchmark A regime still permits unrestricted multi-fault stress testing."""
        benchmark_a = generate_network_benchmark(
            regime="benchmark_a", seed=RANDOM_SEED, save_to_disk=False
        )
        frames = []
        for sid, df in benchmark_a.items():
            sub = df[["timestamp", "is_anomaly", "fault_type"]].copy()
            sub["station_id"] = sid
            sub["cluster_id"] = STATION_TO_CLUSTER.get(sid, "UNKNOWN")
            frames.append(sub)
        combined = pd.concat(frames, ignore_index=True)

        anom = combined[combined["is_anomaly"] == True]
        cluster_concurrency = anom.groupby(["cluster_id", "timestamp"])["station_id"].nunique()
        max_concurrent = cluster_concurrency.max() if len(cluster_concurrency) > 0 else 0

        # In Benchmark A, multiple simultaneous faults in the same cluster are permitted
        assert max_concurrent >= 2, "Benchmark A did not allow multi-fault stress test"
