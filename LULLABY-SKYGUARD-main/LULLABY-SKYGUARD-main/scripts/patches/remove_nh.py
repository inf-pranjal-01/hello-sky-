import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('if r["type"] not in ("physical_bounds", "dropout")]', 'if r["type"] not in ("physical_bounds", "dropout", "network_helper")]')

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('row_fault_type = np.where(helper_wins, "network_helper", row_fault_type)', '')

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
