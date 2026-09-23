import re

with open('model/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

split_logic = """
    # True Temporal Split (P0/Phase 3): Ensure we don't mix future and past data randomly.
    # Train only on the first 70% of the dataset chronologically.
    df = df.sort_values("timestamp")
    cutoff_idx = int(len(df) * 0.7)
    cutoff_date = df.iloc[cutoff_idx]["timestamp"]
    print(f"Temporal Split: Training on data before {cutoff_date}")
    df = df[df["timestamp"] < cutoff_date]
"""

content = content.replace(
    'df = load_clean_training_data()',
    'df = load_clean_training_data()\n' + split_logic
)

with open('model/train.py', 'w', encoding='utf-8') as f:
    f.write(content)
