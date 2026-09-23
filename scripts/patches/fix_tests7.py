import re
import pandas as pd

with open('tests/test_rule_boundaries.py', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace(
    'rules = _rule_checks(raw_reading, feature_row, history_df, artifact)',
    'rules = _rule_checks(raw_reading, feature_row, history_df, history_df, artifact)'
)
with open('tests/test_rule_boundaries.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('tests/test_spatial_cluster.py', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace(
    '_corroborate_network(raw_reading, history_df, {}, "drift", ["temperature_c"])',
    '_corroborate_network(raw_reading, history_df, {}, {}, pd.Series({"temp_roc_1h": 0.0}), "drift", ["temperature_c"])'
)
content = content.replace(
    '_corroborate_network(raw_reading, history_df, {"peer_1": stale_peer_df}, "drift", ["temperature_c"])',
    '_corroborate_network(raw_reading, history_df, {"peer_1": stale_peer_df}, {"peer_1": pd.Series({"temp_roc_1h": 0.0})}, pd.Series({"temp_roc_1h": 0.0}), "drift", ["temperature_c"])'
)
with open('tests/test_spatial_cluster.py', 'w', encoding='utf-8') as f:
    f.write(content)
