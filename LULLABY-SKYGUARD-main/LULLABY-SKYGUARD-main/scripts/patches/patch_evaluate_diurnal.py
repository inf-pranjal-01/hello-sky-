import sys

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Pre-compute roc_lookup at the start of run_rule_engine_and_health
target_start = """    thresholds = artifact["rule_thresholds"]
    prefixes = RULE_ONLY_PREFIXES

    df = featured.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    n = len(df)"""

replacement_start = """    thresholds = artifact["rule_thresholds"]
    prefixes = RULE_ONLY_PREFIXES

    df = featured.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    n = len(df)

    # PRECOMPUTE ROC for Diurnal Consensus Filter
    print("Pre-indexing ROC values for Diurnal Consensus Filter in evaluation...")
    roc_lookup = {}
    for idx in df.index:
        t = df.at[idx, "timestamp"]
        s = df.at[idx, "station_id"]
        roc_lookup[(t, s)] = {
            "temp": df.at[idx, "temp_roc_1h"],
            "pressure": df.at[idx, "pressure_roc_1h"],
            "humidity": df.at[idx, "humidity_roc_1h"]
        }
"""

if target_start not in content:
    print("Could not find target_start in evaluate.py")
    sys.exit(1)
content = content.replace(target_start, replacement_start)

# 2. Apply the diurnal consensus filter in the spike block
target_spike = """                if spike:
                    spike_dev = abs(dev_col[prefix][i]) if not np.isnan(dev_col[prefix][i]) else spike_thresh[prefix]
                    spike_conf = graduated_confidence_spike(spike_dev, spike_thresh[prefix])
                    evidence.append(
                        (
                            "spike",
                            spike_conf,
                        )
                    )"""

replacement_spike = """                if spike:
                    spike_dev = abs(dev_col[prefix][i]) if not np.isnan(dev_col[prefix][i]) else spike_thresh[prefix]
                    spike_conf = graduated_confidence_spike(spike_dev, spike_thresh[prefix])
                    
                    # DIURNAL CONSENSUS FILTER
                    try:
                        from config import SPIKE_DIURNAL_MIN_PEERS, SPIKE_DIURNAL_CONSENSUS_FRACTION, SPIKE_DIURNAL_SUPPRESSION_FACTOR, SPIKE_DIURNAL_PEER_MIN_ROC, CLUSTERS
                        param_name = {"temp": "temperature_c", "pressure": "pressure_hpa", "humidity": "humidity_pct"}[prefix]
                        my_roc = roc_col[prefix][i]
                        if not np.isnan(my_roc):
                            my_dir = 1 if my_roc > 0 else -1
                            t_val = ts[i]
                            cluster_name = next((c for c, sids in CLUSTERS.items() if station_id in sids), None)
                            if cluster_name:
                                eligible_peers = 0
                                agreeing_peers = 0
                                min_roc = SPIKE_DIURNAL_PEER_MIN_ROC.get(param_name, 0.5)
                                for peer in CLUSTERS[cluster_name]:
                                    if peer == station_id: continue
                                    peer_data = roc_lookup.get((t_val, peer))
                                    if peer_data:
                                        peer_roc = peer_data[prefix]
                                        if pd.notna(peer_roc) and not np.isnan(peer_roc) and abs(peer_roc) >= min_roc:
                                            eligible_peers += 1
                                            peer_dir = 1 if peer_roc > 0 else -1
                                            if peer_dir == my_dir:
                                                agreeing_peers += 1
                                if eligible_peers >= SPIKE_DIURNAL_MIN_PEERS:
                                    if (agreeing_peers / eligible_peers) >= SPIKE_DIURNAL_CONSENSUS_FRACTION:
                                        spike_conf *= SPIKE_DIURNAL_SUPPRESSION_FACTOR
                    except Exception as e:
                        print(f"Diurnal consensus error in evaluate: {e}")
                        pass
                    
                    evidence.append(
                        (
                            "spike",
                            spike_conf,
                        )
                    )"""

if target_spike not in content:
    print("Could not find target_spike in evaluate.py")
    sys.exit(1)
content = content.replace(target_spike, replacement_spike)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully patched model/evaluate.py to include Diurnal Consensus Filter.")
