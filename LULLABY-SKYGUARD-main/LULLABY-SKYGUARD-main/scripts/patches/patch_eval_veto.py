import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fetch helper_probability
old_helper = """    helper_lookup = helper_scored.set_index(["station_id", "timestamp"])["helper_alert"]
    helper_alert = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(helper_lookup).fillna(False).to_numpy(dtype=bool)"""

new_helper = """    helper_lookup = helper_scored.set_index(["station_id", "timestamp"])["helper_alert"]
    helper_alert = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(helper_lookup).fillna(False).to_numpy(dtype=bool)
    
    helper_prob_lookup = helper_scored.set_index(["station_id", "timestamp"])["helper_probability"]
    helper_prob = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(helper_prob_lookup).fillna(0.0).to_numpy(dtype=float)"""

content = content.replace(old_helper, new_helper)

# 2. Add the Veto logic
old_pred = """    predicted = predicted | helper_alert | frozen_helper_alert"""

new_pred = """    predicted = predicted | helper_alert | frozen_helper_alert
    
    # ExtraTrees Veto logic for multivariate_inconsistency and unstructured_anomaly
    # unstructured is when model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD and row_fault_type == "none"
    is_unstructured = (model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD) & (row_fault_type == "none")
    is_multivariate = (row_fault_type == "multivariate_inconsistency") & (row_rule_conf > 0)
    
    veto_mask = (is_unstructured | is_multivariate) & (helper_prob < 0.3)
    predicted = predicted & ~veto_mask"""

content = content.replace(old_pred, new_pred)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
