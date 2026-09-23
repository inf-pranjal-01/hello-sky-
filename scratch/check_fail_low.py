import pandas as pd
df = pd.read_csv("data/eval_per_sensor_fault_log.csv")
fail_low = df[df["fault_type"] == "sensor_fail_low"]
print("Total fail low alerts:", len(fail_low))
df_full = pd.read_csv("data/eval_evidence_samples.csv")
merged = pd.merge(fail_low, df_full, on=["station_id", "timestamp"], how="left")
print(merged[["station_id", "timestamp", "parameter", "is_anomaly", "__decision_route"]].head(15))

