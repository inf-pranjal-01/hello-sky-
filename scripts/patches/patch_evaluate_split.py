import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

metric_split = """
    cutoff_idx = int(len(df_full) * 0.7)
    cutoff_date = df_full.iloc[cutoff_idx]["timestamp"]
    print(f"\\nEvaluating metrics strictly on Test Set (>= {cutoff_date})")
    
    res_df = res_df[res_df["timestamp"] >= cutoff_date]
"""

content = content.replace(
    'res_df = pd.DataFrame(results)',
    'res_df = pd.DataFrame(results)\n' + metric_split
)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
