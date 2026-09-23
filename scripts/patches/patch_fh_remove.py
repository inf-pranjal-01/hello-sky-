import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the fault helper block
target = """    # Track A (blueprint A 1) ?" fault_helper live-path wiring.
    fault_helper_artifact = artifact.get("fault_helper")
    if fault_helper_artifact is not None:
        try:
            from config import HELPER_ALERT_THRESHOLD, PHYSICAL_BOUNDS
            from model.fault_helper import feature_columns, build_network_features
            single_row = pd.DataFrame([{**raw_reading, "is_anomaly": False, "fault_type": None}])
            if isinstance(fault_helper_artifact, dict):
                fh_model = fault_helper_artifact.get("helper_model")
                fh_cols = fault_helper_artifact.get("helper_columns")
            else:
                fh_model, fh_cols = fault_helper_artifact[:2]
            fh_featured = build_network_features(single_row)
            available_cols = [c for c in fh_cols if c in fh_featured.columns]
            if available_cols:
                fh_prob = fh_model.predict_proba(fh_featured[available_cols])[:, 1][0]
                
                # Case 1: Base pipeline missed it, but ExtraTrees catches it (Recall boost)
                if not is_anomaly and fh_prob >= HELPER_ALERT_THRESHOLD:
                    is_anomaly = True
                    if fault_type is None:
                        fault_type = "drift"
                    score_pct = max(score_pct, round(fh_prob * 100, 1))
                    severity = score_to_severity(score_pct)
                
                # Case 2: Base pipeline flagged it, but ExtraTrees says it's natural weather (Precision boost)
                # Only apply this veto to multivariate and unstructured as requested by user
                elif is_anomaly and fault_type in ["multivariate_inconsistency", "unstructured_anomaly"]:
                    # If ExtraTrees is very confident it's clean (< 0.3 probability of fault)
                    if fh_prob < 0.3:
                        is_anomaly = False
                        fault_type = None
                        score_pct = round(fh_prob * 100, 1)
                        severity = score_to_severity(score_pct)
                        decision_basis = "vetoed_by_fault_helper"
                        
        except Exception as _fh_exc:
            import logging
            logging.getLogger(__name__).debug("[detect] fault_helper scoring skipped: %s", _fh_exc)"""

if target in content:
    content = content.replace(target, "    # fault_helper removed from live path due to train/serve mismatch.")
else:
    # Use regex
    content = re.sub(r'# Track A \(blueprint A 1\).*?logging\.getLogger\(__name__\)\.debug.*?_fh_exc\)', '# fault_helper removed', content, flags=re.DOTALL)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
