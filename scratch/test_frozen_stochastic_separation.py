import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from data.anomaly_injector import generate_network_benchmark

def analyze_frozen_signatures(seed=20260924):
    print("="*105)
    print("SECTION M: STOCHASTIC SEPARATION OF FROZEN SENSORS VS NATURAL CALM PERIODS")
    print("="*105)
    
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    clean_stats = []
    frozen_stats = []
    
    for sid, df in data.items():
        t_vals = df['temperature_c'].values
        is_frozen = (df['fault_type'] == 'frozen_value').fillna(False).values
        is_clean = (~df['is_anomaly'].fillna(False).values)
        
        # Sliding window of 6 hours
        W = 6
        for i in range(W, len(t_vals)):
            win = t_vals[i - W:i]
            if np.isnan(win).any():
                continue
                
            mad = np.median(np.abs(win - np.median(win)))
            diff = np.diff(win)
            
            # Lag-1 autocorrelation of differences
            if np.std(diff) > 1e-4:
                rho1_diff = np.corrcoef(diff[:-1], diff[1:])[0, 1]
            else:
                rho1_diff = -0.50 # absolute zero variance floor
                
            # Turning point rate
            if len(diff) >= 2:
                turning_points = np.sum((diff[:-1] * diff[1:]) < 0)
                tp_rate = turning_points / (len(diff) - 1)
            else:
                tp_rate = 0.0
                
            stat_entry = {
                "mad": mad,
                "rho1_diff": rho1_diff,
                "tp_rate": tp_rate
            }
            
            if is_frozen[i]:
                frozen_stats.append(stat_entry)
            elif is_clean[i] and mad < 0.5: # low-variance natural calm
                clean_stats.append(stat_entry)
                
    df_clean = pd.DataFrame(clean_stats)
    df_frozen = pd.DataFrame(frozen_stats)
    
    print(f"Sample Size: Natural Low-Variance Calm Windows = {len(df_clean)}, Frozen Windows = {len(df_frozen)}")
    print("-" * 105)
    print(f"{'Feature / Metric':<30} {'Natural Calm (Low MAD)':<35} {'Injected Frozen Sensor':<35}")
    print("-" * 105)
    print(f"{'Mean Rolling MAD (°C)':<30} {df_clean['mad'].mean():<35.3f} {df_frozen['mad'].mean():<35.3f}")
    print(f"{'Mean rho1(diff)':<30} {df_clean['rho1_diff'].mean():<35.3f} {df_frozen['rho1_diff'].mean():<35.3f}")
    print(f"{'Median rho1(diff)':<30} {df_clean['rho1_diff'].median():<35.3f} {df_frozen['rho1_diff'].median():<35.3f}")
    print(f"{'Mean Turning-Point Rate':<30} {df_clean['tp_rate'].mean():<35.3f} {df_frozen['tp_rate'].mean():<35.3f}")
    print(f"{'Turning-Point Rate > 0.40':<30} {(df_clean['tp_rate'] > 0.40).mean()*100:<34.1f}% {(df_frozen['tp_rate'] > 0.40).mean()*100:<34.1f}%")
    print("=" * 105)

if __name__ == '__main__':
    analyze_frozen_signatures()
