import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modify the TP condition
content = content.replace("if predicted[grp.index].mean() >= 0.25:", "if predicted[grp.index].any():")
content = content.replace("Episode must have a substantial portion of its readings flagged", "Episode must have at least one reading flagged (since arms take time)")

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
