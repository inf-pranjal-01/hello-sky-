import re

with open('history_store.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'DATA_DIR = Path(__file__).parent / "data" / "history"',
    'DATA_DIR = Path(__file__).parent / "runtime" / "history"'
)

with open('history_store.py', 'w', encoding='utf-8') as f:
    f.write(content)
