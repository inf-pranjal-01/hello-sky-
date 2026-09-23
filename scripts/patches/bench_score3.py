import pandas as pd
import time

def bench():
    df = pd.read_csv("data/AWS-CHN-024_labeled.csv").head(1000)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    start = time.time()
    
    count = 0
    for i in range(len(df)):
        raw = df.iloc[i].to_dict()
        hist = df.iloc[max(0, i-24):i+1] # 24 rows
        count += 1
        
    end = time.time()
    print(f"Processed {count} rows in {end - start:.2f} seconds")
    print(f"Estimated for 60k rows: {(end - start) * 60:.2f} seconds")

if __name__ == "__main__":
    bench()
