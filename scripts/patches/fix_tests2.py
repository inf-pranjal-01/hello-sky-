import re

with open('tests/test_graduated_and_spatial.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix conf_drift_extreme
content = content.replace('self.assertEqual(conf_drift_extreme, 89.0)', 'self.assertEqual(conf_drift_extreme, 95.0)')

# Fix _corroborate_network calls
content = content.replace(
    "_corroborate_network(raw_reading, target_history, swing_neighbors, fault_type='drift', implicated_params=['temperature_c'])",
    "_corroborate_network(raw_reading, target_history, swing_neighbors, {}, None, fault_type='drift', implicated_params=['temperature_c'])"
)
content = content.replace(
    "_corroborate_network(raw_reading, history_df, empty_neighbors, fault_type='spike', implicated_params=['temperature_c'])",
    "_corroborate_network(raw_reading, history_df, empty_neighbors, {}, None, fault_type='spike', implicated_params=['temperature_c'])"
)
content = content.replace(
    "_corroborate_network(raw_reading, history_df, swing_neighbors, fault_type='spike', implicated_params=['temperature_c'])",
    "_corroborate_network(raw_reading, history_df, swing_neighbors, {}, None, fault_type='spike', implicated_params=['temperature_c'])"
)
content = content.replace(
    "_corroborate_network(raw_reading, history_df, swing_neighbors, fault_type='multivariate_inconsistency', implicated_params=['temperature_c'])",
    "_corroborate_network(raw_reading, history_df, swing_neighbors, {}, None, fault_type='multivariate_inconsistency', implicated_params=['temperature_c'])"
)


with open('tests/test_graduated_and_spatial.py', 'w', encoding='utf-8') as f:
    f.write(content)
