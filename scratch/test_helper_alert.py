import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from evaluation.fast_offline_eval import ARTIFACTS_PATH, HELPER_ALERT_THRESHOLD, FROZEN_HELPER_ALERT_THRESHOLD
from model.fault_helper import predict_faults, score_frozen_channels
from data.anomaly_injector import generate_network_benchmark

helper_path = ARTIFACTS_PATH.parent / "fault_helper.pkl"
helper_artifact = joblib.load(helper_path)
helper_model = helper_artifact["helper_model"]
helper_columns = helper_artifact["helper_columns"]
frozen_helpers = helper_artifact["frozen_helpers"]

for s in [42, 101, 202]:
    data = generate_network_benchmark(regime='benchmark_b', seed=s, save_to_disk=False)
    df_full = pd.concat([df.assign(station_id=sid) for sid, df in data.items()], ignore_index=True)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp']).dt.tz_localize(None)
    
    scored = predict_faults(helper_model, helper_columns, df_full, HELPER_ALERT_THRESHOLD)
    scored = score_frozen_channels(scored, frozen_helpers, FROZEN_HELPER_ALERT_THRESHOLD)
    
    gt = df_full['is_anomaly'].astype(bool)
    h_alert = scored['helper_alert'].fillna(False).astype(bool)
    fz_alert = scored['frozen_helper_alert'].fillna(False).astype(bool)
    
    tp_h = (gt & h_alert).sum()
    fp_h = (~gt & h_alert).sum()
    tp_fz = (gt & fz_alert).sum()
    fp_fz = (~gt & fz_alert).sum()
    
    print(f"Seed {s}:")
    print(f"  helper_alert -> TP={tp_h}, FP={fp_h}, Precision={tp_h/(tp_h+fp_h)*100:.1f}%, Recall={tp_h/gt.sum()*100:.1f}%")
    print(f"  frozen_helper -> TP={tp_fz}, FP={fp_fz}, Precision={tp_fz/(tp_fz+fp_fz)*100:.1f}%, Recall={tp_fz/gt.sum()*100:.1f}%")
