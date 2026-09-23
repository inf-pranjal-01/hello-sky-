import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_constants = """
# Diurnal Spike Suppression Parameters
SPIKE_DIURNAL_MIN_PEERS = 2
SPIKE_DIURNAL_CONSENSUS_FRACTION = 0.5
SPIKE_DIURNAL_SUPPRESSION_FACTOR = 0.38
SPIKE_DIURNAL_PEER_MIN_ROC = {
    "temperature_c": 0.5,
    "pressure_hpa": 0.2,
    "humidity_pct": 1.0
}

# New Architecture Quorum Constants
NETWORK_MIN_ELIGIBLE_PEERS = 2
NETWORK_CORROBORATION_RATIO = 0.5

# Centralized Physical Bounds (replacing duplicated values)
PHYSICAL_BOUNDS = {
    "temperature_c": (-50.0, 60.0),
    "pressure_hpa": (850.0, 1085.0),
    "humidity_pct": (0.0, 100.0)
}

# Centralized Fault Helper Alert Thresholds
HELPER_ALERT_THRESHOLD = 0.85
FROZEN_HELPER_ALERT_THRESHOLD = 0.85
"""

if "SPIKE_DIURNAL_MIN_PEERS" not in content:
    content += new_constants

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
