with open('main.py', 'r') as f:
    text = f.read()

import re

old_recent_block = '''        {
            "anomaly_id": a["anomaly_id"],
            "type": a["type"],
            "label": a["root_cause"],
            "station_id": a["station_id"],
            "timestamp": a["timestamp"].isoformat(),
            "score_pct": a["anomaly_score_pct"],
            "severity": a["severity"],
            "suggested_values": a.get("suggested_values"),
            "observed_values": a.get("observed_values"),
            "affected_parameters": a.get("affected_parameters", []),
            "regime": a.get("regime"),
            "network_corroboration": a.get("network_corroboration"),
        }'''

new_recent_block = '''        {
            "anomaly_id": a["anomaly_id"],
            "timestamp": a["timestamp"].isoformat(),
            "station_id": a["station_id"],
            "anomaly_score_pct": a["anomaly_score_pct"],
            "severity": a["severity"],
            "type": a["type"],
            "root_cause": a["root_cause"],
            "description": f"{a['root_cause']} detected at {a['station_id']}.",
            "suggested_values": a.get("suggested_values"),
            "observed_values": a.get("observed_values"),
            "affected_parameters": a.get("affected_parameters", []),
            "regime": a.get("regime"),
            "network_corroboration": a.get("network_corroboration"),
        }'''

if old_recent_block in text:
    text = text.replace(old_recent_block, new_recent_block)
    print("Patched recent anomalies return in main.py")
else:
    print("Block not found!")

with open('main.py', 'w') as f:
    f.write(text)
