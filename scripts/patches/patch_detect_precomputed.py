import re

with open('model/detect.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """def score_reading(raw_reading: dict, history_df: pd.DataFrame, artifact: dict,
                  neighbor_buffers: dict = None, explainer=None) -> dict:"""
                  
replace1 = """def score_reading(raw_reading: dict, history_df: pd.DataFrame, artifact: dict,
                  neighbor_buffers: dict = None, explainer=None, precomputed_features=None, precomputed_neighbors=None) -> dict:"""

target2 = """    feature_row = build_features_for_latest(history_df)"""
replace2 = """    if precomputed_features is not None:
        feature_row = precomputed_features
    else:
        feature_row = build_features_for_latest(history_df)"""

target3 = """                    try:
                        neighbor_features_safe[nid] = build_features_for_latest(n_df)
                    except Exception:
                        pass"""
replace3 = """                    try:
                        if precomputed_neighbors and nid in precomputed_neighbors:
                            neighbor_features_safe[nid] = precomputed_neighbors[nid]
                        else:
                            neighbor_features_safe[nid] = build_features_for_latest(n_df)
                    except Exception:
                        pass"""

content = content.replace(target1, replace1)
content = content.replace(target2, replace2)
content = content.replace(target3, replace3)

with open('model/detect.py', 'w', encoding='utf-8') as f:
    f.write(content)
