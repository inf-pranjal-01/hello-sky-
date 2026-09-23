import pandas as pd
import time
from model.features import add_temporal_features, add_cross_parameter_features, add_time_features, add_rule_only_signals

def _featurize_buffer(history_df):
    df = add_temporal_features(history_df)
    df = add_cross_parameter_features(df)
    df = add_time_features(df)
    df = add_rule_only_signals(df)
    return df

def bench():
    df = pd.read_csv("data/AWS-CHN-024_labeled.csv").head(1000)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    start = time.time()
    count = 0
    for i in range(len(df)):
        hist = df.iloc[max(0, i-24):i+1].copy() # 24 rows
        if not hist.empty:
            _ = _featurize_buffer(hist)
        count += 1
        
    end = time.time()
    print(f"Processed {count} rows in {end - start:.2f} seconds")
    print(f"Estimated for 60k rows: {(end - start) * 60:.2f} seconds")

if __name__ == "__main__":
    bench()
