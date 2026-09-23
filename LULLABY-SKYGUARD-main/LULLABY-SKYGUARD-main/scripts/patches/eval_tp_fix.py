import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_tp = """                if predicted[grp.index].any():
                    ep_caught[ft] += 1
                    ep_tp += 1
                else:
                    ep_fn += 1"""

new_tp = """                # Episode must have MOST of its readings flagged to count as a TP
                if predicted[grp.index].mean() > 0.5:
                    ep_caught[ft] += 1
                    ep_tp += 1
                else:
                    ep_fn += 1"""

content = content.replace(old_tp, new_tp)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
