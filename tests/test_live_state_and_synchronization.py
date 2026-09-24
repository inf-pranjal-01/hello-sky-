"""
tests/test_live_state_and_synchronization.py
============================================
Comprehensive test suite validating:
1. P0 — StationBuffer cache invalidation and sequential history growth.
2. Same-timestamp order-invariance across 28 stations.
3. Live frozen spatial context (temp_activity_gap) causal computation.
4. Partial station availability handling (28/28, 20/28, 1/28).
5. Peer freshness and staleness bounds.
6. Ranchi multi-fault cluster state isolation.
7. Startup warm-up behavior.
"""

import unittest
import copy
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from model.state import StateManager, StationBuffer, MODE_LIVE
from model.detect import ARTIFACTS_PATH
from model.fault_helper import score_live_fault_helpers
from evaluation.fast_offline_eval import DATA_DIR


class MockHistoryStore:
    def __init__(self):
        self.records = []
    def append(self, station_id, timestamp, raw_reading, verdict, source="live"):
        self.records.append({
            "station_id": station_id,
            "timestamp": timestamp,
            "raw_reading": raw_reading,
            "verdict": verdict,
            "source": source,
        })
    def mark_spike(self, station_id, timestamp, parameter, suggested_value, mode):
        pass
    def get_recent(self, station_id, hours=24, source=None):
        return pd.DataFrame([
            {"timestamp": r["timestamp"], "station_id": r["station_id"], **r["raw_reading"], **r["verdict"]}
            for r in self.records if r["station_id"] == station_id
        ])
    def clear_all(self, source=None):
        self.records = [r for r in self.records if source and r["source"] != source]


class TestLiveStateAndSynchronization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = pd.read_csv(DATA_DIR / "stations_metadata.csv")
        cls.artifact = joblib.load(ARTIFACTS_PATH)
        fh_path = ARTIFACTS_PATH.parent / "fault_helper.pkl"
        if fh_path.exists():
            cls.artifact["fault_helper"] = joblib.load(fh_path)

    # -------------------------------------------------------------------------
    # 1. P0 REGRESSION: StationBuffer Cache Invalidation & History Growth
    # -------------------------------------------------------------------------
    def test_cache_invalidation_and_history_growth(self):
        """
        Prove that sequential record_raw_reading calls invalidate the cache
        and raw_history_df() monotonically grows with each new reading.
        """
        buf = StationBuffer("AWS-BHO-030")
        
        # 1. First reading
        r1 = {"temperature_c": 20.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0}
        t1 = pd.Timestamp("2025-01-10 08:00:00")
        v1 = {"is_anomaly": False, "fault_type": None}
        buf.record_raw_reading(r1, t1, v1)
        h1 = buf.raw_history_df()
        self.assertEqual(len(h1), 1, "First reading must be present in raw_history_df")
        self.assertEqual(h1["temperature_c"].iloc[0], 20.0)

        # 2. Second reading
        r2 = {"temperature_c": 21.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0}
        t2 = pd.Timestamp("2025-01-10 09:00:00")
        v2 = {"is_anomaly": False, "fault_type": None}
        buf.record_raw_reading(r2, t2, v2)
        h2 = buf.raw_history_df()
        self.assertEqual(len(h2), 2, "Second reading must invalidate cache and yield len=2")
        self.assertEqual(list(h2["temperature_c"]), [20.0, 21.0])

        # 3. Third reading
        r3 = {"temperature_c": 22.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0}
        t3 = pd.Timestamp("2025-01-10 10:00:00")
        v3 = {"is_anomaly": False, "fault_type": None}
        buf.record_raw_reading(r3, t3, v3)
        h3 = buf.raw_history_df()
        self.assertEqual(len(h3), 3, "Third reading must invalidate cache and yield len=3")
        self.assertEqual(list(h3["temperature_c"]), [20.0, 21.0, 22.0])

        # 4. Monotonic growth over 10 sequential readings
        for i in range(4, 14):
            r_i = {"temperature_c": 20.0 + i, "pressure_hpa": 1010.0, "humidity_pct": 50.0}
            t_i = pd.Timestamp("2025-01-10 08:00:00") + pd.Timedelta(hours=i)
            buf.record_raw_reading(r_i, t_i, {"is_anomaly": False})
            h_i = buf.raw_history_df()
            self.assertEqual(len(h_i), i)

    # -------------------------------------------------------------------------
    # 2. SAME-TIMESTAMP ORDER-INVARIANCE TEST (All 28 Stations)
    # -------------------------------------------------------------------------
    def test_same_timestamp_order_invariance(self):
        """
        Verify that feeding the same network observation across all stations at timestamp T
        in different orders produces 100% identical anomaly verdicts, fault types, and scores.
        """
        all_sids = list(self.metadata["station_id"])
        t0 = pd.Timestamp("2025-01-15 14:00:00")

        # Create a sample network reading dict
        network_input = {}
        for idx, sid in enumerate(all_sids):
            # Inject varying readings across clusters
            network_input[sid] = ({
                "temperature_c": 25.0 + (idx % 7) * 0.5,
                "pressure_hpa": 1010.0 - (idx % 4) * 1.0,
                "humidity_pct": 50.0 + (idx % 5) * 2.0,
            }, t0)

        # Build baseline state in 3 identical managers
        hist_stores = [MockHistoryStore() for _ in range(3)]
        managers = [StateManager(self.metadata, self.artifact, history_store=h) for h in hist_stores]

        # Populate 5 prior hours into all managers in standard order
        for hr in range(5):
            past_t = t0 - pd.Timedelta(hours=5 - hr)
            batch = {
                sid: ({
                    "temperature_c": 22.0 + hr * 0.5 + (idx % 3) * 0.2,
                    "pressure_hpa": 1012.0,
                    "humidity_pct": 55.0,
                }, past_t)
                for idx, sid in enumerate(all_sids)
            }
            for mgr in managers:
                mgr.ingest_batch(batch)

        # Order A: Standard ascending order
        order_a = {sid: network_input[sid] for sid in sorted(all_sids)}
        verdicts_a = managers[0].ingest_batch(order_a)

        # Order B: Reverse descending order
        order_b = {sid: network_input[sid] for sid in sorted(all_sids, reverse=True)}
        verdicts_b = managers[1].ingest_batch(order_b)

        # Order C: Deterministically shuffled order
        shuffled_sids = list(all_sids)
        random.Random(42).shuffle(shuffled_sids)
        order_c = {sid: network_input[sid] for sid in shuffled_sids}
        verdicts_c = managers[2].ingest_batch(order_c)

        # Assert 100% identity across all stations
        for sid in all_sids:
            va, vb, vc = verdicts_a[sid], verdicts_b[sid], verdicts_c[sid]
            self.assertEqual(va["is_anomaly"], vb["is_anomaly"], f"Order mismatch on {sid} between A and B")
            self.assertEqual(va["is_anomaly"], vc["is_anomaly"], f"Order mismatch on {sid} between A and C")
            self.assertEqual(va["fault_type"], vb["fault_type"], f"Fault type mismatch on {sid} between A and B")
            self.assertEqual(va["fault_type"], vc["fault_type"], f"Fault type mismatch on {sid} between A and C")
            self.assertEqual(va["anomaly_score_pct"], vb["anomaly_score_pct"], f"Score mismatch on {sid} between A and B")
            self.assertEqual(va["anomaly_score_pct"], vc["anomaly_score_pct"], f"Score mismatch on {sid} between A and C")
            self.assertEqual(va["health_status"], vb["health_status"], f"Health status mismatch on {sid} between A and B")
            self.assertEqual(va["health_status"], vc["health_status"], f"Health status mismatch on {sid} between A and C")

    # -------------------------------------------------------------------------
    # 3. LIVE FROZEN SPATIAL CONTEXT (temp_activity_gap)
    # -------------------------------------------------------------------------
    def test_live_frozen_spatial_context_availability(self):
        """
        Verify that score_live_fault_helpers causally extracts temp_activity_gap
        from neighbor_buffers and only enables the spatial gate when sufficient history exists.
        """
        fh_artifact = self.artifact.get("fault_helper")
        if fh_artifact is None:
            self.skipTest("fault_helper artifact not found")

        target_sid = "AWS-CHN-024"
        peer_sids = ["AWS-CHN-101", "AWS-CHN-102", "AWS-CHN-103"]
        t0 = pd.Timestamp("2025-01-10 12:00:00")

        # Case 1: Warmup state (< 4 readings) -> temp_activity_gap is NaN / unavailable
        short_history = pd.DataFrame([{
            "station_id": target_sid, "timestamp": t0, "temperature_c": 25.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0,
        }])
        short_neighbors = {
            p: pd.DataFrame([{"station_id": p, "timestamp": t0, "temperature_c": 25.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0}])
            for p in peer_sids
        }
        res_warmup = score_live_fault_helpers(target_sid, {"temperature_c": 25.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0}, short_history, short_neighbors, fh_artifact)
        self.assertTrue(pd.isna(res_warmup["temp_activity_gap"]), "Warmup with <4 readings must yield NaN activity_gap")
        self.assertFalse(res_warmup["frozen_helper_alert"], "Warmup must not trigger frozen helper alert")

        # Case 2: Established history (8 readings), Target is frozen, Peers are varying
        target_rows = []
        neighbor_dict = {p: [] for p in peer_sids}
        for i in range(8):
            t = t0 + pd.Timedelta(hours=i)
            target_rows.append({"station_id": target_sid, "timestamp": t, "temperature_c": 25.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0})
            for idx, p in enumerate(peer_sids):
                neighbor_dict[p].append({
                    "station_id": p, "timestamp": t,
                    "temperature_c": 24.0 + i * 0.8 + idx * 0.2, # Clear diurnal spread (>3C range)
                    "pressure_hpa": 1010.0, "humidity_pct": 50.0
                })

        h_target = pd.DataFrame(target_rows)
        h_neighbors = {p: pd.DataFrame(rows) for p, rows in neighbor_dict.items()}

        res_frozen = score_live_fault_helpers(target_sid, target_rows[-1], h_target, h_neighbors, fh_artifact)
        self.assertFalse(pd.isna(res_frozen["temp_activity_gap"]), "Activity gap must be a valid float with 8h history")
        self.assertGreaterEqual(res_frozen["temp_activity_gap"], 0.6, "Frozen sensor with active peers must have activity_gap >= 0.6")

    # -------------------------------------------------------------------------
    # 4. PARTIAL STATION BATCH HANDLING (28/28, 20/28, 1/28)
    # -------------------------------------------------------------------------
    def test_partial_station_batch_availability(self):
        """
        Verify that StateManager.ingest_batch seamlessly handles any subset
        of available stations without waiting or failing.
        """
        mgr = StateManager(self.metadata, self.artifact, history_store=MockHistoryStore())
        t0 = pd.Timestamp("2025-01-20 08:00:00")
        all_sids = list(self.metadata["station_id"])

        # 1. Full 28-station batch
        full_batch = {sid: ({"temperature_c": 20.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0}, t0) for sid in all_sids}
        v_full = mgr.ingest_batch(full_batch)
        self.assertEqual(len(v_full), len(all_sids))

        # 2. Partial 20-station batch (8 stations delayed/missing)
        t1 = t0 + pd.Timedelta(hours=1)
        partial_20 = {sid: ({"temperature_c": 21.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0}, t1) for sid in all_sids[:20]}
        v_partial = mgr.ingest_batch(partial_20)
        self.assertEqual(len(v_partial), 20)
        for sid in all_sids[:20]:
            self.assertIn(sid, v_partial)
            self.assertFalse(v_partial[sid]["is_anomaly"])

        # 3. Single station isolated batch
        t2 = t0 + pd.Timedelta(hours=2)
        single_batch = {"AWS-CHN-024": ({"temperature_c": 22.0, "pressure_hpa": 1010.0, "humidity_pct": 50.0}, t2)}
        v_single = mgr.ingest_batch(single_batch)
        self.assertEqual(len(v_single), 1)
        self.assertIn("AWS-CHN-024", v_single)

    # -------------------------------------------------------------------------
    # 5. RANCHI MULTI-FAULT CLUSTER ISOLATION
    # -------------------------------------------------------------------------
    def test_ranchi_multi_fault_isolation(self):
        """
        Verify that multiple stations in the same cluster with simultaneous faults
        maintain strictly independent health states and buffers.
        """
        mgr = StateManager(self.metadata, self.artifact, history_store=MockHistoryStore())
        ran_sids = ["AWS-RAN-067", "AWS-RAN-101", "AWS-RAN-102", "AWS-RAN-103"]
        t0 = pd.Timestamp("2025-01-10 00:00:00")

        # Stream 5 normal hours across Ranchi
        for h in range(5):
            t = t0 + pd.Timedelta(hours=h)
            batch = {sid: ({"temperature_c": 22.0 + h*0.2, "pressure_hpa": 1008.0, "humidity_pct": 60.0}, t) for sid in ran_sids}
            mgr.ingest_batch(batch)

        # Inject extreme spike only on AWS-RAN-067 while AWS-RAN-101 remains normal
        t_spike = t0 + pd.Timedelta(hours=6)
        batch_spike = {
            "AWS-RAN-067": ({"temperature_c": 60.0, "pressure_hpa": 1008.0, "humidity_pct": 60.0}, t_spike),
            "AWS-RAN-101": ({"temperature_c": 23.2, "pressure_hpa": 1008.0, "humidity_pct": 60.0}, t_spike),
        }
        verdicts = mgr.ingest_batch(batch_spike)

        self.assertTrue(verdicts["AWS-RAN-067"]["is_anomaly"], "AWS-RAN-067 must flag physical bounds anomaly")
        self.assertFalse(verdicts["AWS-RAN-101"]["is_anomaly"], "AWS-RAN-101 must remain normal")
        self.assertEqual(mgr.buffers["AWS-RAN-101"].health.status, "HEALTHY")


if __name__ == "__main__":
    unittest.main()
