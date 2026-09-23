import re
import pandas as pd

with open('evaluation/fast_offline_eval.py', 'r') as f:
    text = f.read()

# 1. Add SPIKE to episodic eval
text = text.replace('for ft in ("frozen_value", "drift"):', 'for ft in ("frozen_value", "drift", "spike"):')

# 2. Add imports at the top
if 'from model.spike_tracker import' not in text:
    text = text.replace('import pandas as pd\n', 'import pandas as pd\nfrom model.spike_tracker import init_spike_state, step_spike_state\nfrom model.rules import graduated_confidence_spike\n')

# 3. Replace spike_confirmed dictionary initialization
old_spike_block = r'''        # Causal spike confirmation arrives one reading late, but the
        # detected event belongs to the extreme middle reading. Fully vectorized
        # via numpy array shifts for 100x evaluation speedup.
        spike_confirmed = {}
        for col, prefix in prefixes:
            vals = raw\[col\]
            if len\(vals\) < 3:
                spike_confirmed\[prefix\] = np\.zeros\(m, dtype=bool\)
                continue
            before = np\.empty_like\(vals\)
            before\[0\] = np\.nan
            before\[1:\] = vals\[:-1\]

            jump = np\.abs\(vals - before\)
            thresh = spike_thresh\[prefix\] \* SPIKE_DEVIATION_MULTIPLIER
            qualifies = \(jump > 0\) & \(np\.abs\(dev_col\[prefix\]\) > thresh\) & ~np\.isnan\(jump\) & ~np\.isnan\(dev_col\[prefix\]\)

            reversion = np\.zeros\(m, dtype=bool\)
            for step in \(1, 2, 3\):
                after = np\.empty_like\(vals\)
                after\[:-step\] = vals\[step:\]
                after\[-step:\] = np\.nan
                rev_step = np\.abs\(after - before\) <= \(jump \* SPIKE_REVERSION_RATIO\)
                reversion \|= \(rev_step & ~np\.isnan\(after\)\)

            sc = qualifies & reversion
            sc\[0\] = False
            sc\[-1\] = False
            spike_confirmed\[prefix\] = sc'''

new_spike_init = '''        spike_states = {prefix: init_spike_state() for col, prefix in prefixes}'''

text = re.sub(old_spike_block, new_spike_init, text, flags=re.DOTALL)

# 4. Replace spike evaluation inside the loop
old_spike_eval = r'''                # Spike: calibrated station/parameter threshold\.
                spike = spike_confirmed\[prefix\]\[i\]'''

new_spike_eval = '''                # Spike state machine
                conf, status, reason = step_spike_state(
                    raw[col][i], dev_col[prefix][i], spike_thresh[prefix], SPIKE_DEVIATION_MULTIPLIER, spike_states[prefix], graduated_confidence_spike
                )
                spike_conf_val = conf
                spike = (conf > 0)'''

text = re.sub(old_spike_eval, new_spike_eval, text)

# 5. Replace evidence append for spike
old_evidence_append = r'''                if spike:
                    evidence\.append\(
                        \(
                            "spike",
                            RULE_BASE_CONFIDENCE\["spike"\],
                        \)
                    \)'''

new_evidence_append = '''                if spike:
                    evidence.append(
                        (
                            "spike",
                            spike_conf_val,
                        )
                    )'''
text = re.sub(old_evidence_append, new_evidence_append, text)

with open('evaluation/fast_offline_eval.py', 'w') as f:
    f.write(text)

print("Eval patched!")
