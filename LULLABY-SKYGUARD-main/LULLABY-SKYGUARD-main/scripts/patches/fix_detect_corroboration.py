import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_frozen_detect = '''        elif diverged_peers >= 1:
            state = "CONFIRMED_DIVERGENCE"
            interpretation = f"{diverged_peers} peer(s) diverged beyond calibrated envelope while sensor remained flat."
            confidence_bonus = 6.0
        elif flat_peers >= 1:
            state = "AMBIGUOUS_STABLE_REGION"
            interpretation = f"Peers are also stable/flat within calibrated envelope; regional meteorological stability."'''

new_frozen_detect = '''        elif diverged_peers >= 1:
            state = "CONFIRMED_DIVERGENCE"
            interpretation = f"{diverged_peers} peer(s) diverged beyond calibrated envelope while sensor remained flat."
            confidence_bonus = 16.0
        elif flat_peers >= 1:
            state = "AMBIGUOUS_STABLE_REGION"
            interpretation = f"Peers are also stable/flat within calibrated envelope; regional meteorological stability."
            suppress_alarm = True
        else:
            state = "UNCORROBORATED_STABLE"
            interpretation = "No peers diverged. Assuming regional stability."
            suppress_alarm = True'''

content = content.replace(old_frozen_detect, new_frozen_detect)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
