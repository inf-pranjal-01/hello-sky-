import re
import pandas as pd

with open('tests/test_graduated_and_spatial.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("{}, None", "{}, pd.Series({'temp_roc_1h': 5.0, 'temp_robust_scale': 1.0})")

content = content.replace(
    "_corroborate_network(raw_reading, history_df, empty_neighbors, fault_type='multivariate_inconsistency', implicated_params=['temperature_c', 'humidity_pct'])",
    "_corroborate_network(raw_reading, history_df, empty_neighbors, {}, pd.Series({'temp_roc_1h': 5.0, 'temp_robust_scale': 1.0}), fault_type='multivariate_inconsistency', implicated_params=['temperature_c', 'humidity_pct'])"
)

# Also ensure pandas is imported if not
if "import pandas as pd" not in content:
    content = "import pandas as pd\n" + content

with open('tests/test_graduated_and_spatial.py', 'w', encoding='utf-8') as f:
    f.write(content)
