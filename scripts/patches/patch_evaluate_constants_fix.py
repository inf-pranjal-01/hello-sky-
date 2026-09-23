import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove local PHYSICAL_BOUNDS
content = re.sub(r'# Hard physical sanity bounds.*?PHYSICAL_BOUNDS = \{.*?\}\n', '', content, flags=re.DOTALL)

# Add PHYSICAL_BOUNDS to config imports
content = re.sub(r'from config import \(', 'from config import (\n    PHYSICAL_BOUNDS,\n    HELPER_ALERT_THRESHOLD,\n    FROZEN_HELPER_ALERT_THRESHOLD,', content)

# Remove local HELPER_ALERT_THRESHOLD
content = re.sub(r'HELPER_ALERT_THRESHOLD = 0\.92\n', '', content)

# Remove local FROZEN_HELPER_ALERT_THRESHOLD
content = re.sub(r'# This stricter specialist path.*?FROZEN_HELPER_ALERT_THRESHOLD = 0\.90\n', '', content, flags=re.DOTALL)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
