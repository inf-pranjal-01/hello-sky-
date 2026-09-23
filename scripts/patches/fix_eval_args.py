import sys
import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'res = state_manager.ingest_reading(raw)',
    'res = state_manager.ingest_reading(raw["station_id"], raw, raw["timestamp"])'
)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
