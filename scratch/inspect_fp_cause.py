import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# Inspect AWS-BHO-030 and its peers around 2025-01-02 11:00:00
bho_stations = ['AWS-BHO-030', 'AWS-BHO-101', 'AWS-BHO-102', 'AWS-BHO-103']
dfs = {}
for s in bho_stations:
    df = pd.read_csv(f'data/{s}.csv', parse_dates=['timestamp'])
    dfs[s] = df.set_index('timestamp')

ts_range = pd.date_range('2025-01-02 06:00:00', '2025-01-02 18:00:00', freq='h')
print("=== TEMPERATURE COMPARISON ACROSS BHOPAL CLUSTER ON 2025-01-02 ===")
temp_table = pd.DataFrame({s: dfs[s].loc[ts_range, 'temperature_c'] for s in bho_stations})
temp_table['Peer_Median'] = temp_table[['AWS-BHO-101', 'AWS-BHO-102', 'AWS-BHO-103']].median(axis=1)
temp_table['Target_vs_Peer'] = temp_table['AWS-BHO-030'] - temp_table['Peer_Median']
print(temp_table)

print("\n=== PRESSURE COMPARISON ACROSS BHOPAL CLUSTER ON 2025-01-02 ===")
p_table = pd.DataFrame({s: dfs[s].loc[ts_range, 'pressure_hpa'] for s in bho_stations})
p_table['Peer_Median'] = p_table[['AWS-BHO-101', 'AWS-BHO-102', 'AWS-BHO-103']].median(axis=1)
p_table['Target_vs_Peer'] = p_table['AWS-BHO-030'] - p_table['Peer_Median']
print(p_table)
