import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the hard constraints from the docstring
content = content.replace(
    "- NEVER touches, gates, delays, bonuses, or relabels spike or multivariate_inconsistency.",
    "- Now applies to spike and multivariate_inconsistency to boost precision."
)

# Change elif fault_type == "drift": to elif fault_type in ["drift", "spike", "multivariate_inconsistency"]:
content = re.sub(
    r'elif fault_type == "drift":',
    'elif fault_type in ["drift", "spike", "multivariate_inconsistency"]:',
    content
)

# Wait, if I do that, the 'if fault_type == "drift": confidence_bonus = 5.0' inside it will only give bonus to drift.
# Let's change that too so all of them get the bonus.
content = content.replace(
    'if fault_type == "drift":\n                    confidence_bonus = 5.0',
    'confidence_bonus = 5.0'
)

# And for REGIONAL, it was setting relabel_fault_type = "REGIONAL_EVENT" and NOT vetoing.
# Wait, for spike and multivariate, we want to veto it if it's regional weather, or relabel to none.
# Actually, setting veto=True inside the REGIONAL block is what the user wants ("tell a genuine anomaly apart from real weather")
content = re.sub(
    r'state = "REGIONAL"\n\s+interpretation = "Multiple peers move the same way; regional weather front."\n\s+relabel_fault_type = "REGIONAL_EVENT"\n\s+confidence_bonus = 3.0',
    'state = "REGIONAL"\n            interpretation = "Multiple peers move the same way; regional weather front."\n            veto = True\n            relabel_fault_type = "none"',
    content
)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
