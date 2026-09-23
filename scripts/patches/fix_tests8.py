import re

with open('tests/test_rule_boundaries.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'rules_hi = _rule_checks(raw_reading_hi, feature_row, history_df_hi, artifact)',
    'rules_hi = _rule_checks(raw_reading_hi, feature_row, history_df_hi, history_df_hi, artifact)'
)

with open('tests/test_rule_boundaries.py', 'w', encoding='utf-8') as f:
    f.write(content)
