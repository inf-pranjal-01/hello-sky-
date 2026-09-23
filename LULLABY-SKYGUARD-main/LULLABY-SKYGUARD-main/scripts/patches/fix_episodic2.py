import sys
import re

with open('scripts/patches/patch_hybrid.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'episodic_faults = \[.*?\]', 'episodic_faults = ["frozen_value", "drift", "spike", "multivariate_inconsistency"]', content)

with open('scripts/patches/patch_hybrid.py', 'w', encoding='utf-8') as f:
    f.write(content)
