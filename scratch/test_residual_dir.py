import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

# Let's test modifying fast_offline_eval in memory and running on seeds 42, 101, 202
import evaluation.fast_offline_eval as eval_mod

# Let's write a patched evaluation runner
def evaluate_patched(data_dict, artifact):
    # We test what happens when residual is used for direction_steps
    pass

print("Testing residual direction...")
