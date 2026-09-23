import pandas as pd
df = pd.read_csv("data/AWS-BHO-030_labeled.csv", parse_dates=["timestamp"])
fl_log = df[(df["fault_type"] == "sensor_fail_low")]
print("True fail low stds (2 readings):")
for i, (_, row) in enumerate(fl_log.iterrows()):
    if i > 5: break
    idx = row.name
    window = df.loc[max(0, idx-1):idx]
    print("Temp std:", window["temperature_c"].std(), "Press std:", window["pressure_hpa"].std(), "Hum std:", window["humidity_pct"].std())

