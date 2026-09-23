import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Rip out fault_helper from score_reading
fh_block_pattern = r'# Track A \(blueprint A 1\) — fault_helper live-path wiring\..*?except Exception as _fh_exc:\n.*?logging\.getLogger\(__name__\)\.debug\("\[detect\] fault_helper scoring skipped: %s", _fh_exc\)'

content = re.sub(fh_block_pattern, '', content, flags=re.DOTALL)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
