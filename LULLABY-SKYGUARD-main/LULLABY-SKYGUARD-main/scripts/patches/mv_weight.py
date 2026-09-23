import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modify _fuse_and_score to use dynamic model weighting
old_fuse = """    if model_pct is None:
        # No usable model score (incomplete feature vector, e.g. still
        # in the rolling-window warm-up period) -- fall back to
        # rule-only evidence rather than silently treating a missing
        # model score as "definitely normal."
        overall = rule_confidence
    else:
        overall = MODEL_WEIGHT * model_pct + RULE_WEIGHT * rule_confidence"""

new_fuse = """    if model_pct is None:
        # No usable model score (incomplete feature vector, e.g. still
        # in the rolling-window warm-up period) -- fall back to
        # rule-only evidence rather than silently treating a missing
        # model score as "definitely normal."
        overall = rule_confidence
    else:
        # Give the ML model significantly higher authority for multivariate inconsistency
        if fault_type == "multivariate_inconsistency":
            m_weight = 0.85
            r_weight = 0.15
        else:
            m_weight = MODEL_WEIGHT
            r_weight = RULE_WEIGHT
            
        overall = m_weight * model_pct + r_weight * rule_confidence"""

content = content.replace(old_fuse, new_fuse)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    eval_content = f.read()

old_eval_fuse = """    model_pct = vectorized_model_scores(featured, artifact)

    overall_confidence = (
        MODEL_WEIGHT * model_pct
        + RULE_WEIGHT * row_rule_conf
    )"""

new_eval_fuse = """    model_pct = vectorized_model_scores(featured, artifact)

    # Apply higher model weighting specifically for multivariate inconsistency
    m_weight = np.where(row_fault_type == "multivariate_inconsistency", 0.85, MODEL_WEIGHT)
    r_weight = np.where(row_fault_type == "multivariate_inconsistency", 0.15, RULE_WEIGHT)

    overall_confidence = (
        m_weight * model_pct
        + r_weight * row_rule_conf
    )"""

eval_content = eval_content.replace(old_eval_fuse, new_eval_fuse)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(eval_content)
