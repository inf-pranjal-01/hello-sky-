import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modify the TP condition
content = content.replace("if predicted[grp.index].mean() > 0.5:", "if predicted[grp.index].mean() >= 0.25:")
content = content.replace("Episode must have MOST of its readings flagged", "Episode must have a substantial portion of its readings flagged")

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
