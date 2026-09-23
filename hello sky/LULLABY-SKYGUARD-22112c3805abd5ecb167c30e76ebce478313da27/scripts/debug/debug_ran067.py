import sys
sys.path.insert(0, '.')
import pandas as pd
from model.features import add_temporal_features
from model.detect import score_reading

# Load the model artifact for thresholds
import pickle
with open('model_artifacts/isolation_forest.pkl', 'rb') as f:
    artifact = pickle.load(f)

# Load data
df = pd.read_csv('data/AWS-RAN-067_labeled.csv', parse_dates=['timestamp'])
df['station_id'] = 'AWS-RAN-067'
df = df.head(100) # first few days
featured = add_temporal_features(df)

# We will simulate the causal buffer process
print("Evaluating AWS-RAN-067 for false positives...")
for i in range(10, min(100, len(df))):
    raw_reading = df.iloc[i].to_dict()
    # history_df is the past 10 rows including current
    history_df = df.iloc[i-10:i+1].copy()
    feature_row = featured.iloc[i]
    
    verdict = score_reading(
        raw_reading=raw_reading,
        history_df=history_df,
        artifact=artifact,
        is_warmup=False,
        neighbor_buffers=None
    )
    
    if verdict['is_anomaly']:
        print(f"\n{raw_reading['timestamp']}: ANOMALY DETECTED!")
        print(f"Severity: {verdict['severity']}, Score: {verdict['anomaly_score_pct']}%")
        print(f"Rules fired: {verdict['rules_fired']}")
        print(f"Reading: T={raw_reading['temperature_c']}C, P={raw_reading['pressure_hpa']}hPa, H={raw_reading['humidity_pct']}%")
