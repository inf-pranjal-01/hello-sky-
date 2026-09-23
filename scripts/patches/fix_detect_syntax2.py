import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("from config import PHYSICAL_BOUNDS,\n    score_to_severity", "from config import (\n    PHYSICAL_BOUNDS,\n    score_to_severity")

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
