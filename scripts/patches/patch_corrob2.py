import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change _corroborate_network signature
content = content.replace(
    'def _corroborate_network(raw_reading: dict, history_df: pd.DataFrame, neighbor_buffers: dict, fault_type: str, implicated_params: list, artifact: dict = None) -> dict:',
    'def _corroborate_network(raw_reading: dict, history_df: pd.DataFrame, neighbor_buffers: dict, neighbor_features: dict, target_features: pd.Series, fault_type: str, implicated_params: list, artifact: dict = None) -> dict:'
)

# Remove the target_features recomputation
target_feat_old = """    # Compute target station features ONCE before the peer loop
    try:
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
        }"""
content = content.replace(target_feat_old, "    # target_features is passed in directly.")

# Change neighbor feature fetching
neighbor_feat_old = """        try:
            n_features = build_features_for_latest(n_df)
        except Exception:
            eligible_peers -= 1
            continue"""
neighbor_feat_new = """        n_features = neighbor_features.get(nid)
        if n_features is None:
            eligible_peers -= 1
            continue"""
content = content.replace(neighbor_feat_old, neighbor_feat_new)

# Update score_reading to build neighbor features and pass them
score_reading_old = """            network_details = _corroborate_network(
                raw_reading, history_df, neighbor_buffers_safe, pre_fusion_fault, implicated, artifact=artifact
            )"""
score_reading_new = """            neighbor_features_safe = {}
            for nid, n_df in neighbor_buffers_safe.items():
                if n_df is not None and not n_df.empty:
                    try:
                        neighbor_features_safe[nid] = build_features_for_latest(n_df)
                    except Exception:
                        pass
            
            network_details = _corroborate_network(
                raw_reading, history_df, neighbor_buffers_safe, neighbor_features_safe, feature_row, pre_fusion_fault, implicated, artifact=artifact
            )"""
content = content.replace(score_reading_old, score_reading_new)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
