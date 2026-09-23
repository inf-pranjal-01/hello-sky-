import unittest
from pathlib import Path

class TestArchitecture(unittest.TestCase):
    def test_canonical_engine_used(self):
        project_root = Path(__file__).parent.parent
        with open(project_root / "model" / "evaluate.py", "r", encoding="utf-8") as f:
            eval_code = f.read()
        self.assertIn("StateManager", eval_code)
        
        with open(project_root / "model" / "state.py", "r", encoding="utf-8") as f:
            state_code = f.read()
        self.assertIn("DecisionEngine.decide", state_code)

    def test_train_serve_parity(self):
        project_root = Path(__file__).parent.parent
        with open(project_root / "model" / "detect.py", "r", encoding="utf-8") as f:
            detect_code = f.read()
        self.assertNotIn("build_network_features", detect_code)

if __name__ == "__main__":
    unittest.main()
