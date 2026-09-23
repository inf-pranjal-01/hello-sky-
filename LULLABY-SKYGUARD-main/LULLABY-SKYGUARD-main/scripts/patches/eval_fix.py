import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add pred_ft initialization and the missing merge logic back!
old_str = """    frozen_only = row_rule_conf == RULE_BASE_CONFIDENCE['frozen_value']
    predicted = predicted & ~(frozen_only & (model_pct < FROZEN_MIN_MODEL_CORROBORATION))

    has_rule_ft = (row_fault_type != None) & (row_fault_type != "none")"""

new_str = """    frozen_only = row_rule_conf == RULE_BASE_CONFIDENCE['frozen_value']
    predicted = predicted & ~(frozen_only & (model_pct < FROZEN_MIN_MODEL_CORROBORATION))

    featured = featured.merge(labels, on=["station_id", "timestamp"], how="left")
    featured["is_anomaly"] = featured["is_anomaly"].fillna(False).astype(bool)
    featured["fault_type"] = featured["fault_type"].fillna("none")
    featured["__source_file"] = featured["__source_file"].fillna("unknown")
    featured["__predicted"] = predicted
    featured["__model_pct"] = model_pct
    featured["__rule_confidence_pct"] = row_rule_conf
    featured["__score_pct"] = overall_confidence

    pred_ft = pd.Series("none", index=featured.index, dtype="object")
    has_rule_ft = (row_fault_type != None) & (row_fault_type != "none")"""

content = content.replace(old_str, new_str)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
