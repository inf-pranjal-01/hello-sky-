import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from evaluation.fast_offline_eval import (
    evaluate_all, ARTIFACTS_PATH, PHYSICAL_BOUNDS,
    vectorized_model_scores, run_rule_engine_and_health,
    apply_spatial_corroboration, compute_episodic_result,
    _score_and_report, _featurize,
    MODEL_WEIGHT, RULE_WEIGHT, FUSION_ANOMALY_THRESHOLD,
    MODEL_ALONE_OVERRIDE_THRESHOLD, RULE_CONFIDENCE_BYPASS
)
from data.anomaly_injector import generate_network_benchmark

# Let's inspect how the rule engine and spatial corroboration work
print("Testing prototype...")
