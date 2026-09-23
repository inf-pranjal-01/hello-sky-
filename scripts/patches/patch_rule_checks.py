import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change _rule_checks signature
content = content.replace(
    'def _rule_checks(raw_reading: dict, feature_row: pd.Series, history_df: pd.DataFrame, artifact: dict) -> dict:',
    'def _rule_checks(raw_reading: dict, feature_row: pd.Series, history_df: pd.DataFrame, featured_buffer: pd.DataFrame, artifact: dict) -> dict:'
)

# Remove the internal featurize_buffer call inside _rule_checks
old_featurize_call = """    # drift (CUSUM, A 2) + multivariate_inconsistency (A 4) share one
    # featurized-buffer pass -- computed once, used by both.
    featured_buffer = _featurize_buffer(history_df)"""
content = content.replace(old_featurize_call, "    # featured_buffer is passed in directly.")

# Update score_reading to pass featured_buffer
old_score_reading_rules = 'rules = _rule_checks(raw_reading, feature_row, history_df, artifact)'
new_score_reading_rules = """    # Build the full featurized buffer ONCE for this reading
    featured_buffer = _featurize_buffer(history_df)
    rules = _rule_checks(raw_reading, feature_row, history_df, featured_buffer, artifact)"""
content = content.replace(old_score_reading_rules, new_score_reading_rules)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
