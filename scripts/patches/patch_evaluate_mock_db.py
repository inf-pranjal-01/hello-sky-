import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    metadata = pd.read_csv(DATA_DIR / "stations_metadata.csv")
    state_manager = StateManager(metadata, artifact, history_store=None)"""

replace = """    class MockHistoryStore:
        def append(self, *args, **kwargs): pass
        def mark_spike(self, *args, **kwargs): pass
        def get_all(self, *args, **kwargs): return pd.DataFrame()
        
    metadata = pd.read_csv(DATA_DIR / "stations_metadata.csv")
    state_manager = StateManager(metadata, artifact, history_store=MockHistoryStore())"""

content = content.replace(target, replace)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
