import unittest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

class TestAPIContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The simulator is initialized by FastAPI's lifespan handler.
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()
        import pandas as pd
        sim = app.state.sim
        for sid in sim.metadata["station_id"]:
            raw = sim._live_cache.get(sid, {"temperature_c": 25.0, "pressure_hpa": 1000.0, "humidity_pct": 60.0})
            sim.latest[sid] = {
                "raw_reading": raw,
                "verdict": {
                    "anomaly_score_pct": 5.0,
                    "is_anomaly": False,
                    "fault_type": None,
                    "severity": "none",
                    "model_confidence_pct": 5.0,
                    "rule_confidence_pct": 0.0,
                    "suggested_values": {},
                },
                "timestamp": pd.Timestamp.now(tz="UTC"),
            }

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_stations_endpoint(self):
        # We know we need to test /api/stations
        response = self.client.get("/api/stations")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        if len(data) > 0:
            self.assertIn("station_id", data[0])
            self.assertIn("name", data[0])

    def test_system_status(self):
        response = self.client.get("/api/system-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("mode", data)
        self.assertIn("live_poll_interval_seconds", data)

    def test_station_scoped_read_endpoints(self):
        stations = self.client.get("/api/stations").json()
        self.assertTrue(stations)
        station_id = stations[0]["station_id"]

        for path in (
            f"/api/current-reading?station_id={station_id}",
            f"/api/trends?station_id={station_id}&hours=6",
            f"/api/anomalies/latest?station_id={station_id}",
            f"/api/anomalies/recent?station_id={station_id}&limit=5",
            f"/api/sensor-health?station_id={station_id}",
            "/api/network-status",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

if __name__ == '__main__':
    unittest.main()
