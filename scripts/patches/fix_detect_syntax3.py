import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "relabel_fault_type = \"none\"" in line and lines[i+1].strip() == "else:":
        # This is where the indentation is messed up.
        # the original code was:
        #             elif diverged_peers == 0:
        #                 ...
        #             else:
        #                 state = "AMBIGUOUS_DIVERGENCE"
        pass

# Let's just fix it using regex by matching the broken block.
content = "".join(lines)
broken_block = """                elif diverged_peers == 0:
                    state = "REGIONAL_STABILITY"
                    interpretation = "Cluster fully agrees with sensor; regional weather."
                    veto = True
                relabel_fault_type = "none"
            else:
                state = "AMBIGUOUS_DIVERGENCE\""""

fixed_block = """                elif diverged_peers == 0:
                    state = "REGIONAL_STABILITY"
                    interpretation = "Cluster fully agrees with sensor; regional weather."
                    veto = True
                else:
                    state = "AMBIGUOUS_DIVERGENCE\""""

content = content.replace(broken_block, fixed_block)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
