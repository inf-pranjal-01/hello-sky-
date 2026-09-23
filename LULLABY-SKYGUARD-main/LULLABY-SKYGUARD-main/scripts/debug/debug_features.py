import pandas as pd
from model.features import add_temporal_features
import sys

df = pd.read_csv('data/AWS-RAN-067_labeled.csv', parse_dates=['timestamp'])
df = add_temporal_features(df)
jan1 = df[(df['timestamp'].dt.day == 1) & (df['timestamp'].dt.hour >= 16) & (df['timestamp'].dt.hour <= 21)]
for i, row in jan1.iterrows():
    print(f"{row['timestamp'].hour}: temp_roc_1h={row['temp_roc_1h']:.2f}, std={row['temp_rolling_std']:.2f}")
