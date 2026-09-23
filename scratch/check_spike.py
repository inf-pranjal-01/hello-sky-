import pandas as pd
df = pd.read_csv("data/eval_per_sensor_fault_log.csv")
spikes = df[df["fault_type"] == "spike"]
df_full = pd.read_csv("data/eval_evidence_samples.csv")
merged = pd.merge(spikes, df_full, on=["station_id", "timestamp"], how="left")
pd.set_option("display.max_columns", None)
print(merged[["timestamp", "parameter", "is_anomaly", "__decision_route", "__score_pct", "fault_type_y"]].head(20))

