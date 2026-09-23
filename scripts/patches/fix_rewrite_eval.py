import re

with open('scripts/patches/rewrite_eval.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("from config import ARTIFACTS_PATH", "ARTIFACTS_PATH = PROJECT_ROOT / 'model_artifacts' / 'isolation_forest.pkl'")

with open('scripts/patches/rewrite_eval.py', 'w', encoding='utf-8') as f:
    f.write(content)
