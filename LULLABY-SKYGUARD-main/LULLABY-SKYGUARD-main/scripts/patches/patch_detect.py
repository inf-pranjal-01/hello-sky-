with open('model/detect.py', 'r') as f:
    text = f.read()

import re

# 1. Modify score_reading signature
old_sig = '''def score_reading(raw_reading: dict, history_df: pd.DataFrame, artifact: dict,
                  explainer=None) -> dict:'''

new_sig = '''def _corroborate_network(raw_reading: dict, history_df: pd.DataFrame, neighbor_buffers: dict, fault_type: str, implicated_params: list) -> str:
    if not neighbor_buffers:
        return "INSUFFICIENT_CORROBORATION"
    
    target_time = pd.to_datetime(raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1])
    
    eligible_peers = 0
    corroborating_peers = 0
    
    for nid, n_df in neighbor_buffers.items():
        if n_df.empty: continue
        n_latest = n_df.iloc[-1]
        n_time = pd.to_datetime(n_latest["timestamp"])
        
        # freshness / timestamp alignment
        if abs((n_time - target_time).total_seconds()) > 3600:
            continue
            
        eligible_peers += 1
        
        # For simplicity, if fault is related to a parameter, we check if neighbor has a similar extreme value
        # But wait, we can just run a quick deviation check
        n_features = build_features_for_latest(n_df)
        
        # If any implicated parameter has a deviation > 2.0 (or < -2.0) in the same direction, it's regional
        is_corroborating = False
        for param in implicated_params:
            prefix = PARAM_PREFIXES.get(param)
            if not prefix: continue
            
            target_dev = build_features_for_latest(history_df).get(f"{prefix}_deviation", 0)
            peer_dev = n_features.get(f"{prefix}_deviation", 0)
            
            if pd.notna(target_dev) and pd.notna(peer_dev):
                if abs(peer_dev) > 2.0 and (target_dev * peer_dev > 0):
                    is_corroborating = True
                    break
        
        if is_corroborating:
            corroborating_peers += 1

    if eligible_peers < 1:
        return "INSUFFICIENT_CORROBORATION"
    elif corroborating_peers > 0:
        return "REGIONAL"
    else:
        return "LOCALIZED"

def score_reading(raw_reading: dict, history_df: pd.DataFrame, artifact: dict,
                  neighbor_buffers: dict = None, explainer=None) -> dict:'''

if old_sig in text:
    text = text.replace(old_sig, new_sig)
    print("Patched signature and added network logic")
else:
    print("Signature not found!")

# 2. Add regime logic and network corroboration call inside score_reading
old_return = '''    return {
        "anomaly_score_pct": round(score_pct, 1),
        "model_confidence_pct": round(model_pct, 1) if model_pct is not None else None,
        "rule_confidence_pct": round(rule_confidence, 1),
        "is_anomaly": bool(is_anomaly),
        "severity": severity,
        "fault_type": fault_type,
        "rules_fired": rules["fired"],
        "fast_path_offline_params": rules["fast_path_offline_params"],
        "confirmed_spikes": rules["confirmed_spikes"],
        "suggested_values": suggested_values,
        "shap_features": shap_features_public,
        "likely_faulty_sensors": likely_sensors,
    }'''

new_return = '''    # Regime metadata (context-only, Option A)
    # Simple mockup: use the hour of day and temp to assign a regime.
    hour = pd.to_datetime(raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1]).hour
    temp = raw_reading.get("temperature_c", 25.0)
    
    if hour >= 6 and hour < 18:
        if temp > 35:
            regime = "SUMMER_DAY_HIGH_HEAT"
        else:
            regime = "NORMAL_DAY"
    else:
        if temp < 10:
            regime = "WINTER_NIGHT_COLD"
        else:
            regime = "NORMAL_NIGHT"

    network_state = "INSUFFICIENT_CORROBORATION"
    if is_anomaly:
        implicated = likely_sensors
        if not implicated and rules["fired"]:
            implicated = list(set(r[1] for r in rules["fired"]))
        network_state = _corroborate_network(raw_reading, history_df, neighbor_buffers, fault_type, implicated)

    return {
        "anomaly_score_pct": round(score_pct, 1),
        "model_confidence_pct": round(model_pct, 1) if model_pct is not None else None,
        "rule_confidence_pct": round(rule_confidence, 1),
        "is_anomaly": bool(is_anomaly),
        "severity": severity,
        "fault_type": fault_type,
        "rules_fired": rules["fired"],
        "fast_path_offline_params": rules["fast_path_offline_params"],
        "confirmed_spikes": rules["confirmed_spikes"],
        "suggested_values": suggested_values,
        "shap_features": shap_features_public,
        "likely_faulty_sensors": likely_sensors,
        "regime": regime,
        "network_corroboration": network_state,
    }'''

if old_return in text:
    text = text.replace(old_return, new_return)
    print("Patched return statement")
else:
    print("Return block not found!")

with open('model/detect.py', 'w') as f:
    f.write(text)
