import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Insert dynamic threshold calculation BEFORE the peer loop
dynamic_logic = """    try:
        target_features = build_features_for_latest(history_df)
    except Exception:
        return {
            "state": "INSUFFICIENT_CORROBORATION",
            "eligible_peer_count": 0,
            "corroborating_peer_count": 0,
            "network_interpretation": "Target station features could not be built for comparison.",
            "confidence_bonus": 0.0,
            "relabel_fault_type": None,
            "veto": False,
        }

    # -- DYNAMIC THRESHOLD CALCULATION --
    # Calculate dynamic limits per reading to account for gradual weather changes
    dynamic_thresholds = {}
    all_dfs = [history_df] + [ndf for nid, ndf in neighbor_buffers.items() if ndf is not None and not ndf.empty]
    if len(all_dfs) >= 2:
        min_len = min(len(df) for df in all_dfs)
        if min_len >= 6:
            for param in implicated_params:
                prefix = PARAM_PREFIXES.get(param)
                if not prefix:
                    continue
                
                aligned_series = []
                st_means = []
                for df in all_dfs:
                    # Get last min_len valid readings
                    vals = df[param].dropna().astype(float).values[-min_len:]
                    if len(vals) == min_len:
                        aligned_series.append(vals)
                        st_means.append(vals.mean())
                        
                if len(aligned_series) == len(all_dfs):
                    deviations = []
                    for i in range(min_len):
                        step_residuals = [aligned_series[j][i] - st_means[j] for j in range(len(aligned_series))]
                        mean_residual = sum(step_residuals) / len(step_residuals)
                        for r in step_residuals:
                            deviations.append(abs(r - mean_residual))
                            
                    if len(deviations) >= 6:
                        deviations.sort()
                        p90_idx = int(len(deviations) * 0.90)
                        dyn_val = deviations[p90_idx]
                        
                        # Gradual onset adjustment
                        target_roc = target_features.get(f"{prefix}_roc_1h")
                        if pd.notna(target_roc) and dyn_val > 0:
                            if abs(float(target_roc)) < (dyn_val * 0.5):
                                dyn_val *= 3.0
                                
                        floor = {"temperature_c": 1.2, "pressure_hpa": 1.5, "humidity_pct": 2.0}.get(param, 1.2)
                        dynamic_thresholds[param] = max(floor, round(dyn_val, 2))
    # -----------------------------------
"""

content = content.replace(
    """    try:
        target_features = build_features_for_latest(history_df)
    except Exception:
        return {
            "state": "INSUFFICIENT_CORROBORATION",
            "eligible_peer_count": 0,
            "corroborating_peer_count": 0,
            "network_interpretation": "Target station features could not be built for comparison.",
            "confidence_bonus": 0.0,
            "relabel_fault_type": None,
            "veto": False,
        }""",
    dynamic_logic
)

# 2. Inside the peer loop, replace div_thresh assignment
peer_logic_old = """            # Peer delta over 1h
            peer_roc = n_features.get(f"{prefix}_roc_1h", 0.0)
            peer_delta = abs(float(peer_roc)) if pd.notna(peer_roc) else 0.0
            div_thresh = param_calib["divergence_threshold"] if param_calib else 1.0"""

peer_logic_new = """            # Peer delta over 1h
            peer_roc = n_features.get(f"{prefix}_roc_1h", 0.0)
            peer_delta = abs(float(peer_roc)) if pd.notna(peer_roc) else 0.0
            
            # Use dynamic threshold if enough data warm-up, otherwise fallback to static artifact
            if param in dynamic_thresholds:
                div_thresh = dynamic_thresholds[param]
            else:
                div_thresh = param_calib["divergence_threshold"] if param_calib else 1.0"""

content = content.replace(peer_logic_old, peer_logic_new)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
