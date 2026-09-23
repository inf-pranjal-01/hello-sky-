import pickle
import sys
import os
sys.path.append(os.getcwd())
with open("model_artifacts/isolation_forest.pkl", "rb") as f:
    artifact = pickle.load(f)
fault_helper_artifact = artifact.get("fault_helper")
if isinstance(fault_helper_artifact, dict):
    fh_model = fault_helper_artifact.get("helper_model")
else:
    fh_model = fault_helper_artifact[0]
print("Classes:", getattr(fh_model, "classes_", "No classes_ attribute"))

