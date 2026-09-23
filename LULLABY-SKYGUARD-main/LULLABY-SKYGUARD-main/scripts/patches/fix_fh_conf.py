import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the fault_helper block in detect.py
old_fh = """            if available_cols:
                fh_prob = fh_model.predict_proba(fh_featured[available_cols])[:, 1][0]
                # It's a confidence contribution, not an override. 
                # Max 89.9 so it can't independently bypass the rule-bypass threshold of 90.
                if fh_prob > 0.1:
                    fired.append({
                        "type": "network_helper",
                        "parameter": "all",
                        "observed_value": None,
                        "suggested_value": None,
                        "confidence": min(89.9, fh_prob * 100),
                        "reason": f"Supervised spatial network model flagged abnormality with {fh_prob:.1%} probability."
                    })"""

new_fh = """            if available_cols:
                fh_prob = fh_model.predict_proba(fh_featured[available_cols])[:, 1][0]
                # It's a SMALL confidence contribution, not an override. 
                # Capped at 40.0 so it only adds 16 points (0.4 * 40) to the fusion score,
                # meaning it can only tip the scales if the unsupervised model is already suspicious.
                if fh_prob >= HELPER_ALERT_THRESHOLD:
                    fh_conf = min(40.0, fh_prob * 50.0)
                    fired.append({
                        "type": "network_helper",
                        "parameter": "all",
                        "observed_value": None,
                        "suggested_value": None,
                        "confidence": fh_conf,
                        "reason": f"Supervised spatial network model flagged abnormality with {fh_prob:.1%} probability."
                    })"""

content = content.replace(old_fh, new_fh)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Now fix evaluate.py
with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    eval_content = f.read()

old_eval_fh = """    # Bounded confidence contribution inside the fusion logic!
    fh_conf = np.minimum(89.9, helper_prob * 100.0)
    helper_wins = (fh_conf > row_rule_conf) & (fh_conf > 10.0)
    row_rule_conf = np.maximum(row_rule_conf, fh_conf)
    row_fault_type = np.where(helper_wins, "network_helper", row_fault_type)
    
    frz_conf = np.minimum(89.9, frozen_prob * 100.0)
    frz_wins = (frz_conf > row_rule_conf) & (frz_conf > 10.0)
    row_rule_conf = np.maximum(row_rule_conf, frz_conf)
    row_fault_type = np.where(frz_wins, "frozen_value", row_fault_type)"""

new_eval_fh = """    # Bounded confidence contribution inside the fusion logic!
    # Max 40.0 confidence, only if above alert threshold, so it acts as a small nudge (16 points in fusion).
    fh_conf = np.where(helper_prob >= HELPER_ALERT_THRESHOLD, np.minimum(40.0, helper_prob * 50.0), 0.0)
    helper_wins = (fh_conf > row_rule_conf) & (fh_conf > 0.0)
    row_rule_conf = np.maximum(row_rule_conf, fh_conf)
    row_fault_type = np.where(helper_wins, "network_helper", row_fault_type)
    
    frz_conf = np.where(frozen_prob >= FROZEN_HELPER_ALERT_THRESHOLD, np.minimum(40.0, frozen_prob * 50.0), 0.0)
    frz_wins = (frz_conf > row_rule_conf) & (frz_conf > 0.0)
    row_rule_conf = np.maximum(row_rule_conf, frz_conf)
    row_fault_type = np.where(frz_wins, "frozen_value", row_fault_type)"""

eval_content = eval_content.replace(old_eval_fh, new_eval_fh)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(eval_content)
