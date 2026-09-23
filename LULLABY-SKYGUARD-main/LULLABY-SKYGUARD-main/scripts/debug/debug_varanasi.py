import pandas as pd
from model.features import add_temporal_features
from model.seasonal_baseline import get_expected_roc
import sys

df = pd.read_csv('data/AWS-VAR-052_labeled.csv', parse_dates=['timestamp'])
df = add_temporal_features(df)
jan1 = df[(df['timestamp'].dt.day == 1) & (df['timestamp'].dt.hour >= 16) & (df['timestamp'].dt.hour <= 23)]
for i, row in jan1.iterrows():
    h = row['timestamp'].hour
    exp = get_expected_roc('AWS-VAR-052', 'temp', h)
    act = row['temp_roc_1h']
    std = row['temp_rolling_std']
    res = (act - exp) / std if std > 0 else 0
    print(f"{h}: exp={exp:.2f}, act={act:.2f}, std={std:.2f}, res={res:.2f}")
