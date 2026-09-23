import re

with open('tests/test_graduated_and_spatial.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "_corroborate_network(raw_reading, history_df, swing_neighbors, fault_type='multivariate_inconsistency', implicated_params=['temperature_c', 'humidity_pct'])",
    "_corroborate_network(raw_reading, history_df, swing_neighbors, {'AWS-CHN-101': pd.Series({'temp_roc_1h': 0.0, 'temp_robust_scale': 1.0}), 'AWS-CHN-103': pd.Series({'temp_roc_1h': 0.0, 'temp_robust_scale': 1.0}), 'AWS-CHN-104': pd.Series({'temp_roc_1h': 0.0, 'temp_robust_scale': 1.0})}, pd.Series({'temp_roc_1h': 5.0, 'temp_robust_scale': 1.0}), fault_type='multivariate_inconsistency', implicated_params=['temperature_c', 'humidity_pct'])"
)

# And fix the {} for neighbor features in the drift one
content = content.replace(
    "swing_neighbors, {}, pd.Series({'temp_roc_1h': 5.0, 'temp_robust_scale': 1.0}), fault_type='drift'",
    "swing_neighbors, {'AWS-CHN-101': pd.Series({'temp_roc_1h': 0.0, 'temp_robust_scale': 1.0}), 'AWS-CHN-103': pd.Series({'temp_roc_1h': 0.0, 'temp_robust_scale': 1.0}), 'AWS-CHN-104': pd.Series({'temp_roc_1h': 0.0, 'temp_robust_scale': 1.0})}, pd.Series({'temp_roc_1h': 5.0, 'temp_robust_scale': 1.0}), fault_type='drift'"
)
content = content.replace(
    "self.assertEqual(res_drift['confidence_bonus'], 3.0)",
    "self.assertEqual(res_drift['confidence_bonus'], 5.0)"
)

with open('tests/test_graduated_and_spatial.py', 'w', encoding='utf-8') as f:
    f.write(content)
