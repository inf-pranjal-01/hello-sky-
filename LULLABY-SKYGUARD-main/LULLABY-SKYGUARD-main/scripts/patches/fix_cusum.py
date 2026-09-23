import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix CUSUM_DRIFT_ALLOWANCE
old_allowance = """CUSUM_DRIFT_ALLOWANCE = {
    "temperature_c": 0.25,
    "pressure_hpa": 0.05,
    "humidity_pct": 0.25
}"""
new_allowance = """CUSUM_DRIFT_ALLOWANCE = {
    "temperature_c": 0.05,
    "pressure_hpa": 0.02,
    "humidity_pct": 0.05
}"""
content = content.replace(old_allowance, new_allowance)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
