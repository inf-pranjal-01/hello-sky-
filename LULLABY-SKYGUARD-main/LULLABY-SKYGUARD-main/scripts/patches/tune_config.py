import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Lower CUSUM_THRESHOLD to 4.0
content = re.sub(r'CUSUM_THRESHOLD\s*=\s*[\d\.]+', 'CUSUM_THRESHOLD = 4.0', content)

# 2. Raise MULTIVARIATE_PERSISTENCE_REQUIRED to 3
content = re.sub(r'MULTIVARIATE_PERSISTENCE_REQUIRED\s*=\s*\d+', 'MULTIVARIATE_PERSISTENCE_REQUIRED = 3', content)

# 3. Adjust tie-breaker ceilings in config.py
# Drift max to 97.0
content = content.replace("95.0 ceiling at 2x threshold", "97.0 ceiling at 2x threshold")
content = re.sub(r'return min\(95\.0, 85\.0 \+ 10\.0 \* ratio\)', 'return min(97.0, 85.0 + 12.0 * ratio)', content)

# Frozen max to 96.0
content = content.replace("95.0 ceiling for 2x threshold streak", "96.0 ceiling for 2x threshold streak")
content = re.sub(r'return min\(95\.0, 80\.0 \+ 15\.0 \* ratio\)', 'return min(96.0, 80.0 + 16.0 * ratio)', content)

# Spike max to 94.0
content = content.replace("85.0 floor up to 95.0 ceiling", "85.0 floor up to 94.0 ceiling")
content = re.sub(r'return min\(95\.0, 85\.0 \+ 10\.0 \* magnitude_ratio\)', 'return min(94.0, 85.0 + 9.0 * magnitude_ratio)', content)

# Multivariate max to 93.0
content = content.replace("Confirmed tier: 88.0 to 95.0", "Confirmed tier: 88.0 to 93.0")
content = re.sub(r'return min\(95\.0, 88\.0 \+ 7\.0 \* ratio\)', 'return min(93.0, 88.0 + 5.0 * ratio)', content)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
