with open('main.py', 'r') as f:
    text = f.read()

import re

# Block 1
old_block1 = '''                "affected_parameters": a.get("affected_parameters", []),
            }'''
new_block1 = '''                "affected_parameters": a.get("affected_parameters", []),
                "regime": a.get("regime"),
                "network_corroboration": a.get("network_corroboration"),
            }'''
if old_block1 in text:
    text = text.replace(old_block1, new_block1)
    print("Patched main.py block 1")
else:
    print("main.py block 1 not found")

# Block 2
old_block2 = '''            "suggested_values": a.get("suggested_values"),
            "observed_values": a.get("observed_values"),
            "affected_parameters": a.get("affected_parameters", []),
        }
        for a in matches
    ]'''
new_block2 = '''            "suggested_values": a.get("suggested_values"),
            "observed_values": a.get("observed_values"),
            "affected_parameters": a.get("affected_parameters", []),
            "regime": a.get("regime"),
            "network_corroboration": a.get("network_corroboration"),
        }
        for a in matches
    ]'''
if old_block2 in text:
    text = text.replace(old_block2, new_block2)
    print("Patched main.py block 2")
else:
    print("main.py block 2 not found")

# Block 3
old_block3 = '''        "anomaly_score_pct": match.get("anomaly_score_pct"),
        "fault_type": match.get("type"),
    }'''
new_block3 = '''        "anomaly_score_pct": match.get("anomaly_score_pct"),
        "fault_type": match.get("type"),
        "regime": match.get("regime"),
        "network_corroboration": match.get("network_corroboration"),
    }'''
if old_block3 in text:
    text = text.replace(old_block3, new_block3)
    print("Patched main.py block 3")
else:
    print("main.py block 3 not found")

with open('main.py', 'w') as f:
    f.write(text)
