import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix FROZEN_CONSECUTIVE_REQUIRED
old_frozen = "FROZEN_CONSECUTIVE_REQUIRED = 6"
new_frozen = "FROZEN_CONSECUTIVE_REQUIRED = 4"
content = content.replace(old_frozen, new_frozen)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
