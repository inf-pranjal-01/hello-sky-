import sys
sys.path.insert(0, '.')
import joblib
from evaluation.fast_offline_eval import ARTIFACTS_PATH

artifact = joblib.load(ARTIFACTS_PATH)
rule_thresholds = artifact.get("rule_thresholds", {})
print("rule_thresholds keys:", list(rule_thresholds.keys()))
for k, v in rule_thresholds.items():
    print(k, type(v), list(v.keys()) if isinstance(v, dict) else v)
    if isinstance(v, dict) and "temp" in v:
        print("  temp sample:", list(v["temp"].items())[:3] if isinstance(v["temp"], dict) else v["temp"])
