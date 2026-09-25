import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import CLUSTERS
from data.anomaly_injector import generate_network_benchmark

STATION_TO_CLUSTER = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    for sid in [center] + neighbors:
        STATION_TO_CLUSTER[sid] = cid

def print_windows(seed=20260924):
    data = generate_network_benchmark(regime='benchmark_b', seed=seed, save_to_disk=False)
    
    clean_dfs = {}
    for sid in STATION_TO_CLUSTER:
        c_df = pd.read_csv(f"data/{sid}.csv", parse_dates=["timestamp"])
        c_df["timestamp"] = pd.to_datetime(c_df["timestamp"]).dt.tz_localize(None)
        clean_dfs[sid] = c_df.set_index("timestamp")
        
    for target_ft in ["drift", "frozen_value", "multivariate_inconsistency", "spike"]:
        print("\n" + "="*110)
        print(f"REPRESENTATIVE EXACT ROW WINDOW FOR FAULT TYPE: {target_ft.upper()}")
        print("="*110)
        
        # Find first station with this fault
        found = False
        for sid, df in data.items():
            df_inj = df.copy()
            df_inj["timestamp"] = pd.to_datetime(df_inj["timestamp"]).dt.tz_localize(None)
            matches = df_inj[df_inj["fault_type"] == target_ft]
            if len(matches) > 0:
                cid = STATION_TO_CLUSTER[sid]
                peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
                
                # Get start and end index of first episode
                idx_start = matches.index[0]
                idx_end = matches.index[0]
                while idx_end + 1 < len(df_inj) and df_inj.loc[idx_end + 1, "fault_type"] == target_ft:
                    idx_end += 1
                    
                win_start = max(0, idx_start - 10)
                win_end = min(len(df_inj) - 1, idx_end + 10)
                
                window_df = df_inj.loc[win_start:win_end].copy()
                
                print(f"Station: {sid} (Cluster: {cid}) | Fault: {target_ft} | Duration: {idx_end - idx_start + 1} hours")
                print(f"Peers: {peer_ids}")
                print("-" * 110)
                print(f"{'Timestamp':<20} {'Status':<12} {'Target Clean':<14} {'Target Injected':<16} {'Delta':<10} {'Peer Med':<10} {'Peer MAD':<10} {'Raw Resid':<10}")
                print("-" * 110)
                
                for idx, row in window_df.iterrows():
                    ts = row["timestamp"]
                    clean_t = clean_dfs[sid].loc[ts, "temperature_c"] if ts in clean_dfs[sid].index else np.nan
                    inj_t = row["temperature_c"]
                    delta = inj_t - clean_t
                    
                    p_vals = [clean_dfs[p].loc[ts, "temperature_c"] for p in peer_ids if ts in clean_dfs[p].index]
                    p_med = np.median(p_vals) if p_vals else np.nan
                    p_mad = np.median(np.abs(np.array(p_vals) - p_med)) if p_vals else np.nan
                    raw_res = inj_t - p_med
                    
                    status = "IN_FAULT" if (idx_start <= idx <= idx_end) else "NORMAL"
                    print(f"{str(ts):<20} {status:<12} {clean_t:<14.2f} {inj_t:<16.2f} {delta:<10.2f} {p_med:<10.2f} {p_mad:<10.2f} {raw_res:<10.2f}")
                    
                found = True
                break

if __name__ == "__main__":
    print_windows()
