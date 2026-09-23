import sys

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = 'featured = featured.merge(labels, on=["station_id", "timestamp"], how="left")'
replacement = 'featured["timestamp"] = pd.to_datetime(featured["timestamp"]).dt.tz_localize(None)\n    featured = featured.merge(labels, on=["station_id", "timestamp"], how="left")'

content = content.replace(target, replacement)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
