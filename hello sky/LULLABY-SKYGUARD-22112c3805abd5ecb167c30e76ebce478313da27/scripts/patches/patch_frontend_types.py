with open('FRONTEND/src/types/index.ts', 'r') as f:
    text = f.read()

import re

# Insert NetworkCorroborationState type
new_type = '''export type AnomalyType =
  | 'spike'
  | 'frozen_value'
  | 'drift'
  | 'dropout'
  | 'sensor_fail_low'
  | 'multivariate_inconsistency';

export type NetworkCorroborationState = 'LOCALIZED' | 'REGIONAL' | 'INSUFFICIENT_CORROBORATION';'''
text = text.replace('''export type AnomalyType =
  | 'spike'
  | 'frozen_value'
  | 'drift'
  | 'dropout'
  | 'sensor_fail_low'
  | 'multivariate_inconsistency';''', new_type)

# Add to LatestAnomaly
old_la = '''  observed_values?: Record<string, number | null>;
}'''
new_la = '''  observed_values?: Record<string, number | null>;
  regime?: string;
  network_corroboration?: NetworkCorroborationState;
}'''
# Need to be careful to replace only the first occurrence for LatestAnomaly or just all 3
# LatestAnomaly, RecentAnomalyItem, AnomalyExplanation all end with `observed_values` (Wait, AnomalyExplanation ends with `fault_type`)

# Let's just use regex
text = re.sub(
    r'(export interface LatestAnomaly \{[^}]+observed_values\?: Record<string, number \| null>;\n)\}',
    r'\g<1>  regime?: string;\n  network_corroboration?: NetworkCorroborationState;\n}',
    text
)

text = re.sub(
    r'(export interface RecentAnomalyItem \{[^}]+observed_values\?: Record<string, number \| null>;\n)\}',
    r'\g<1>  regime?: string;\n  network_corroboration?: NetworkCorroborationState;\n}',
    text
)

text = re.sub(
    r'(export interface AnomalyExplanation \{[^}]+fault_type\?: AnomalyType;\n)\}',
    r'\g<1>  regime?: string;\n  network_corroboration?: NetworkCorroborationState;\n}',
    text
)

with open('FRONTEND/src/types/index.ts', 'w') as f:
    f.write(text)
print("Patched types")
