import sys
sys.path.insert(0, '.')
import joblib
from evaluation.fast_offline_eval import ARTIFACTS_PATH
from config import SPIKE_DEVIATION_MULTIPLIER

artifact = joblib.load(ARTIFACTS_PATH)
thresholds = artifact.get("thresholds", {})
print("SPIKE_DEVIATION_MULTIPLIER:", SPIKE_DEVIATION_MULTIPLIER)
for p in ["temp", "pressure", "humidity"]:
    th = thresholds.get("spike", {}).get(p, {})
    print(f"Spike threshold for {p}: {th}")
