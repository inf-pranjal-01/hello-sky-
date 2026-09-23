import pandas as pd
from model.features import add_temporal_features
import sys

df = pd.read_csv('data/AWS-RAN-067_labeled.csv', parse_dates=['timestamp'])
df = add_temporal_features(df)

# Look at Jan 01 from 12:00 to 23:00
jan1 = df[(df['timestamp'].dt.day == 1) & (df['timestamp'].dt.hour >= 12)]
for i, row in jan1.iterrows():
    t_roc = row.get('temp_roc_3h', 0)
    p_roc = row.get('pressure_roc_3h', 0)
    h_roc = row.get('humidity_roc_3h', 0)
    print(f"{row['timestamp']} | T={row['temperature_c']:.1f} P={row['pressure_hpa']:.1f} H={row['humidity_pct']:.1f} | T_ROC3={t_roc:.2f} P_ROC3={p_roc:.2f} H_ROC3={h_roc:.2f}")

