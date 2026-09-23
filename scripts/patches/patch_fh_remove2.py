import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'# Track A.*?logging\.getLogger\(__name__\)\.debug\("\[detect\] fault_helper scoring skipped: \%s", _fh_exc\)', '# fault_helper removed', content, flags=re.DOTALL)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
