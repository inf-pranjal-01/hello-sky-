import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

# Let's inspect running evaluate_all with the fix
import evaluation.fast_offline_eval as eval_mod

# Read fast_offline_eval.py content and modify in memory
with open("evaluation/fast_offline_eval.py", "r", encoding="utf-8") as f:
    code = f.read()

# Fix 1: direction_steps records residual instead of raw difference
code_patched = code.replace(
    'if previous_value is not None and not np.isnan(previous_value) and not np.isnan(value):\n                    st["direction_steps"].append(value - previous_value)\n                st["previous_value"] = value\n                \n                if not np.isnan(raw_roc):',
    'st["previous_value"] = value\n                \n                if not np.isnan(raw_roc):\n                    allowance = CUSUM_DRIFT_ALLOWANCE.get(col, 0.05) if isinstance(CUSUM_DRIFT_ALLOWANCE, dict) else CUSUM_DRIFT_ALLOWANCE\n                    min_scale = 1.0 if col in ("temperature_c", "humidity_pct") else 0.3\n                    eff_scale = max(float(scale_val), min_scale) if (scale_val is not None and np.isfinite(scale_val) and scale_val > 0) else min_scale\n                    expected = get_expected_roc(station_id, prefix, int(h))\n                    residual = float(np.clip((raw_roc - expected) / eff_scale, -3.0, 3.0))\n                    st["direction_steps"].append(residual)'
)

# Fix 2: remove duplicate residual calculation
code_patched = code_patched.replace(
    '                    allowance = CUSUM_DRIFT_ALLOWANCE.get(col, 0.05) if isinstance(CUSUM_DRIFT_ALLOWANCE, dict) else CUSUM_DRIFT_ALLOWANCE\n                    min_scale = 1.0 if col in ("temperature_c", "humidity_pct") else 0.3\n                    eff_scale = max(float(scale_val), min_scale) if (scale_val is not None and np.isfinite(scale_val) and scale_val > 0) else min_scale\n                    expected = get_expected_roc(station_id, prefix, int(h))\n                    residual = float(np.clip((raw_roc - expected) / eff_scale, -3.0, 3.0))\n                    \n                    st["splus"] = max(0.0, st["splus"] + residual - allowance)',
    '                    st["splus"] = max(0.0, st["splus"] + residual - allowance)'
)

# Fix 3: disable uncorroborated drift suppression
code_patched = code_patched.replace(
    'suppress_drift = (\n        uncorrob_drift\n        & (model_pct < UNCORROBORATED_DRIFT_MIN_MODEL_PCT)\n        & ~row_hard\n        & ~(model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD)\n    )\n    predicted = predicted & ~suppress_drift',
    'suppress_drift = pd.Series(False, index=featured.index)\n    # uncorroborated drift is true isolated drift confirmed by spatial gate'
)

with open("scratch/patched_fast_offline_eval.py", "w", encoding="utf-8") as f:
    f.write(code_patched)

print("Saved patched evaluator to scratch/patched_fast_offline_eval.py")
