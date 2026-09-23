import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add imports to detect.py
import_pattern = r'from config import \(\n'
new_imports = r'from config import (\n    NETWORK_MIN_ELIGIBLE_PEERS,\n    NETWORK_CORROBORATION_RATIO,\n'
content = re.sub(import_pattern, new_imports, content)

# Fix _corroborate_network logic
frozen_logic_old = """        if eligible_peers < 1:
            state = "INSUFFICIENT_CORROBORATION"
            interpretation = "No eligible peers with fresh data to corroborate frozen sensor."
        elif diverged_peers >= 1:
            state = "CONFIRMED_DIVERGENCE"
            interpretation = f"{diverged_peers} peer(s) diverged beyond calibrated envelope while sensor remained flat."
            confidence_bonus = 16.0
        elif flat_peers >= 1:
            state = "AMBIGUOUS_STABLE_REGION"
            veto = True
            relabel_fault_type = "none"
        else:"""

frozen_logic_new = """        if eligible_peers < NETWORK_MIN_ELIGIBLE_PEERS:
            state = "INSUFFICIENT_CORROBORATION"
            interpretation = "No eligible peers with fresh data to corroborate frozen sensor."
        elif (diverged_peers / eligible_peers) >= NETWORK_CORROBORATION_RATIO:
            state = "CONFIRMED_DIVERGENCE"
            interpretation = f"{diverged_peers}/{eligible_peers} peers diverged beyond envelope while sensor remained flat."
            confidence_bonus = 16.0
        elif (flat_peers / eligible_peers) >= NETWORK_CORROBORATION_RATIO:
            state = "AMBIGUOUS_STABLE_REGION"
            veto = True
            relabel_fault_type = "none"
        else:"""

content = content.replace(frozen_logic_old, frozen_logic_new)

# General logic for other faults also needs the fix if it uses `diverged_peers >= 1`
# Let's just fix it generally.
general_logic_old = """            diverge_ratio = diverged_peers / eligible_peers if eligible_peers > 0 else 0.0
            if diverge_ratio >= 0.5:
                state = "CONFIRMED_DIVERGENCE"
                interpretation = f"Sensor clearly diverging from {diverged_peers}/{eligible_peers} peers."
                confidence_bonus = 5.0
            elif diverged_peers == 0:
                state = "REGIONAL_STABILITY"
                interpretation = "Cluster fully agrees with sensor; regional weather."
                veto = True"""

general_logic_new = """            if eligible_peers < NETWORK_MIN_ELIGIBLE_PEERS:
                state = "INSUFFICIENT_CORROBORATION"
                interpretation = "Not enough eligible peers to corroborate."
                veto = False
            else:
                diverge_ratio = diverged_peers / eligible_peers
                if diverge_ratio >= NETWORK_CORROBORATION_RATIO:
                    state = "CONFIRMED_DIVERGENCE"
                    interpretation = f"Sensor clearly diverging from {diverged_peers}/{eligible_peers} peers."
                    confidence_bonus = 5.0
                elif diverged_peers == 0:
                    state = "REGIONAL_STABILITY"
                    interpretation = "Cluster fully agrees with sensor; regional weather."
                    veto = True"""

content = content.replace(general_logic_old, general_logic_new)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
