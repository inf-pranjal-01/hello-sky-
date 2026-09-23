import re

with open('model/engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """               model_artifact: dict, 
               state: Any = None) -> dict:"""
replace1 = """               model_artifact: dict, 
               state: Any = None,
               precomputed_features: pd.Series = None,
               precomputed_neighbors: dict = None) -> dict:"""

target2 = """        return score_reading(reading, station_history, model_artifact, peer_snapshot, state)"""
replace2 = """        return score_reading(reading, station_history, model_artifact, peer_snapshot, state, precomputed_features, precomputed_neighbors)"""

content = content.replace(target1, replace1)
content = content.replace(target2, replace2)

with open('model/engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
