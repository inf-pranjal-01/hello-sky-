import sys
import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'state_manager = StateManager(artifact, mock_history_store=True)',
    'metadata = pd.read_csv(DATA_DIR / "stations_metadata.csv")\n    state_manager = StateManager(metadata, artifact, history_store=None)'
)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
