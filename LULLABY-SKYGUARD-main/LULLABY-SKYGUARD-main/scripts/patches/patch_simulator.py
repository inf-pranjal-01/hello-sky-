with open('model/simulator.py', 'r') as f:
    text = f.read()

import re

# Block 1
old_block1 = '''                    "type": "spike",
                    "root_cause": ROOT_CAUSE_BY_FAULT_TYPE["spike"],
                })'''
new_block1 = '''                    "type": "spike",
                    "root_cause": ROOT_CAUSE_BY_FAULT_TYPE["spike"],
                    "regime": verdict.get("regime"),
                    "network_corroboration": verdict.get("network_corroboration"),
                })'''
if old_block1 in text:
    text = text.replace(old_block1, new_block1)
    print("Patched simulator block 1")
else:
    print("Simulator block 1 not found")

# Block 2
old_block2 = '''                    "root_cause": ROOT_CAUSE_BY_FAULT_TYPE.get(
                        verdict["fault_type"],
                        "Unusual reading pattern flagged by model"
                    ),
                })'''
new_block2 = '''                    "root_cause": ROOT_CAUSE_BY_FAULT_TYPE.get(
                        verdict["fault_type"],
                        "Unusual reading pattern flagged by model"
                    ),
                    "regime": verdict.get("regime"),
                    "network_corroboration": verdict.get("network_corroboration"),
                })'''
if old_block2 in text:
    text = text.replace(old_block2, new_block2)
    print("Patched simulator block 2")
else:
    print("Simulator block 2 not found")

with open('model/simulator.py', 'w') as f:
    f.write(text)
