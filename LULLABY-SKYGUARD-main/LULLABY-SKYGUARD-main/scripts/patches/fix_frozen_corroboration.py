import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modify frozen_value corroboration
old_frozen_corroboration = """        if ft == "frozen_value":
            if diverged_peers >= 1:
                row_rule_conf[idx] = min(89.5, row_rule_conf[idx] + 6.0)
                bonuses_awarded += 1"""

new_frozen_corroboration = """        if ft == "frozen_value":
            if diverged_peers >= 1:
                # Validated! Neighbors are changing while this sensor is stuck.
                row_rule_conf[idx] = 96.0
                bonuses_awarded += 1
            else:
                # Uncorroborated! Neighbors are also flat (or offline). Suppress natural stable weather FP.
                row_fault_type[idx] = "REGIONAL_EVENT"
                row_rule_conf[idx] = 0.0
                regional_events_found += 1"""

content = content.replace(old_frozen_corroboration, new_frozen_corroboration)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
