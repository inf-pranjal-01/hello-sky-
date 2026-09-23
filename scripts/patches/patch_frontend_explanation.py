import re

with open('frontend/src/components/analytics/ExplainabilityCommandCenter.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "'Physical inconsistency: Temperature and humidity violate Clausius-Clapeyron atmospheric limits.'",
    "'Thermodynamic consistency violation: Temperature and humidity violate Clausius-Clapeyron atmospheric limits.'"
)
content = content.replace(
    "Physical inconsistency: Temperature and humidity violate Clausius-Clapeyron atmospheric limits.",
    "Thermodynamic consistency violation: Temperature and humidity violate Clausius-Clapeyron atmospheric limits."
)

with open('frontend/src/components/analytics/ExplainabilityCommandCenter.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
