import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import CLUSTERS

STATION_TO_CLUSTER = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    for sid in [center] + neighbors:
        STATION_TO_CLUSTER[sid] = cid

def diagnose_all_stations():
    print("="*105)
    print("SECTION K: METEOROLOGICAL GENERATIVE STRUCTURE & DIURNAL CYCLE DIAGNOSTICS")
    print("="*105)
    
    records = []
    for sid, cid in STATION_TO_CLUSTER.items():
        df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        hours = df['timestamp'].dt.hour
        t_by_hour = df.groupby(hours)['temperature_c'].mean()
        
        min_hour = t_by_hour.idxmin()
        max_hour = t_by_hour.idxmax()
        daily_amp = t_by_hour.max() - t_by_hour.min()
        
        # Fit harmonic 1: sin(2pi*h/24), cos(2pi*h/24)
        sin1 = np.sin(2 * np.pi * hours / 24.0)
        cos1 = np.cos(2 * np.pi * hours / 24.0)
        X1 = np.column_stack([np.ones(len(df)), sin1, cos1])
        y = df['temperature_c'].values
        c1, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
        res1 = y - X1 @ c1
        std1 = np.std(res1)
        
        # Fit harmonic 2 (semidiurnal): sin(4pi*h/24), cos(4pi*h/24)
        sin2 = np.sin(4 * np.pi * hours / 24.0)
        cos2 = np.cos(4 * np.pi * hours / 24.0)
        X2 = np.column_stack([np.ones(len(df)), sin1, cos1, sin2, cos2])
        c2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
        res2 = y - X2 @ c2
        std2 = np.std(res2)
        
        records.append({
            "Station": sid,
            "Cluster": cid,
            "Daily Amp (°C)": daily_amp,
            "T_min Hour": f"{min_hour:02d}:00",
            "T_max Hour": f"{max_hour:02d}:00",
            "Harmonic 1 Res Std": std1,
            "Harmonic 1+2 Res Std": std2,
            "Harmonic Gain": std1 - std2
        })
        
    df_diag = pd.DataFrame(records)
    print(df_diag.to_string(index=False))
    print("="*105)

if __name__ == '__main__':
    diagnose_all_stations()
