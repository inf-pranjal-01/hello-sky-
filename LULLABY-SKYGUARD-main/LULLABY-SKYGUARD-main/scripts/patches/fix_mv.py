import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix MULTIVARIATE_PERSISTENCE_REQUIRED
old_mv = "MULTIVARIATE_PERSISTENCE_REQUIRED = 2"
new_mv = "MULTIVARIATE_PERSISTENCE_REQUIRED = 3"
content = content.replace(old_mv, new_mv)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
