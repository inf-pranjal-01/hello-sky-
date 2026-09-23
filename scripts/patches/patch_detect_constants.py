import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove local PHYSICAL_BOUNDS
content = re.sub(r'# independent check.*?PHYSICAL_BOUNDS = \{.*?\}\n', '', content, flags=re.DOTALL)

# Add import for PHYSICAL_BOUNDS
content = re.sub(r'(from config import .*)', r'\1, PHYSICAL_BOUNDS', content)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
