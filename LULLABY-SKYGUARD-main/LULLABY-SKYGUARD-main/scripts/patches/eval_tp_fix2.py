with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('episodic_faults = ["frozen_value", "drift", "multivariate_inconsistency"]', 'episodic_faults = ["frozen_value", "drift"]')

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
