import pandas as pd
df = pd.read_csv('data/eval_per_sensor_fault_log.csv')
print(df[df['Fault_Type'] == 'drift'][['Station_ID', 'Parameter']].value_counts())
