import sys

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = 'episodic_faults = ["frozen_value", "drift"]'
replacement = 'episodic_faults = ["frozen_value", "drift", "spike", "multivariate_inconsistency", "unstructured_anomaly"]'

content = content.replace(target, replacement)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
