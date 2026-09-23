import sys
from pathlib import Path

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("DRIFT_MIN_MODEL_CORROBORATION = 25.0", "DRIFT_MIN_MODEL_CORROBORATION = 0.0")
content = content.replace("FROZEN_MIN_MODEL_CORROBORATION = 65.0", "FROZEN_MIN_MODEL_CORROBORATION = 0.0")

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
