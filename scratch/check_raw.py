import pandas as pd
df = pd.read_csv("data/AWS-BHO-030_labeled.csv", parse_dates=["timestamp"])
fl_log = pd.read_csv("data/eval_per_sensor_fault_log.csv")
fl_log["timestamp"] = pd.to_datetime(fl_log["timestamp"])
bho_fl = fl_log[(fl_log["station_id"] == "AWS-BHO-030") & (fl_log["fault_type"] == "sensor_fail_low")]
merged = pd.merge(bho_fl, df, on=["station_id", "timestamp"], how="left")
print(merged[["timestamp", "parameter", "temperature_c", "pressure_hpa", "humidity_pct", "is_anomaly", "fault_type_y"]].head(20))

