import subprocess
import re

def set_config(cusum, frozen, model_thresh, fusion_thresh):
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'CUSUM_THRESHOLD = [\d\.]+', f'CUSUM_THRESHOLD = {cusum}', content)
    content = re.sub(r'FROZEN_CONSECUTIVE_REQUIRED = \d+', f'FROZEN_CONSECUTIVE_REQUIRED = {frozen}', content)
    content = re.sub(r'MODEL_ALONE_OVERRIDE_THRESHOLD = [\d\.]+', f'MODEL_ALONE_OVERRIDE_THRESHOLD = {model_thresh}', content)
    content = re.sub(r'FUSION_ANOMALY_THRESHOLD = [\d\.]+', f'FUSION_ANOMALY_THRESHOLD = {fusion_thresh}', content)
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(content)

configs = [
    (7.0, 5, 95.0, 50.0),   # baseline
    (6.0, 5, 90.0, 55.0),
    (5.0, 5, 90.0, 55.0),
    (7.0, 6, 95.0, 60.0),
    (6.0, 6, 90.0, 60.0),
    (8.0, 5, 95.0, 55.0),
]

for c, f, m, fusion in configs:
    set_config(c, f, m, fusion)
    print(f"\n--- Testing CUSUM={c}, FROZEN={f}, MODEL={m}, FUSION={fusion} ---")
    result = subprocess.run(['.venv\\Scripts\\python', 'model\\evaluate.py'], capture_output=True, text=True)
    
    prec_match = re.search(r'Overall Precision \(Hybrid\):\s+([\d\.]+)%', result.stdout)
    rec_match = re.search(r'Overall Recall \(Hybrid\):\s+([\d\.]+)%', result.stdout)
    
    if prec_match and rec_match:
        print(f"Precision: {prec_match.group(1)}% | Recall: {rec_match.group(1)}%")
    else:
        print("Could not parse output!")
