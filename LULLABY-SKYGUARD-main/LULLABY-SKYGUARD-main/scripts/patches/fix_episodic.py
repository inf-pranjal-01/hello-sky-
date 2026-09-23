import sys

with open('scripts/patches/patch_hybrid.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace whatever the episodic_faults line is with just frozen and drift
import re
content = re.sub(r'episodic_faults = \[.*?\]', 'episodic_faults = ["frozen_value", "drift"]', content)

with open('scripts/patches/patch_hybrid.py', 'w', encoding='utf-8') as f:
    f.write(content)
