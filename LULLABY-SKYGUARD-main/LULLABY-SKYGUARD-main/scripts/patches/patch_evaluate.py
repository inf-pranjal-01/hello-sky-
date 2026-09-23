import sys

with open('model/evaluate.py', 'r') as f:
    content = f.read()

old_block = """                if not np.isnan(nroc):
                    st["splus"] = max(0.0, st["splus"] + nroc - CUSUM_DRIFT_ALLOWANCE) if nroc > 0 else 0.0
                    st["sminus"] = max(0.0, st["sminus"] - nroc - CUSUM_DRIFT_ALLOWANCE) if nroc < 0 else 0.0"""

new_block = """                if not np.isnan(nroc):
                    allowance = CUSUM_DRIFT_ALLOWANCE.get(col, 0.05) if isinstance(CUSUM_DRIFT_ALLOWANCE, dict) else CUSUM_DRIFT_ALLOWANCE
                    st["splus"] = max(0.0, st["splus"] + nroc - allowance) if nroc > 0 else 0.0
                    st["sminus"] = max(0.0, st["sminus"] - nroc - allowance) if nroc < 0 else 0.0"""

if old_block not in content:
    print("Could not find the target block in evaluate.py")
    sys.exit(1)

content = content.replace(old_block, new_block)

with open('model/evaluate.py', 'w') as f:
    f.write(content)
print("Successfully patched model/evaluate.py")
