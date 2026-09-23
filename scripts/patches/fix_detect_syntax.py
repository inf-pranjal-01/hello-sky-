import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("from config import (, PHYSICAL_BOUNDS", "from config import PHYSICAL_BOUNDS,")

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
