import pickle
import sys
import os
sys.path.append(os.getcwd())
with open("model_artifacts/fault_helper.pkl", "rb") as f:
    fh_model, cols = pickle.load(f)
print("Classes:", getattr(fh_model, "classes_", "No classes_ attribute"))

