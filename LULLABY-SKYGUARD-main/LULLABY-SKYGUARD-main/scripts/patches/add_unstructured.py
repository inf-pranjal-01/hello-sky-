import sys

with open('scripts/patches/patch_hybrid.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('episodic_faults = ["frozen_value", "drift", "spike", "multivariate_inconsistency"]', 'episodic_faults = ["frozen_value", "drift", "spike", "multivariate_inconsistency", "unstructured_anomaly"]')

with open('scripts/patches/patch_hybrid.py', 'w', encoding='utf-8') as f:
    f.write(content)
