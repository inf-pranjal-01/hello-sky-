import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))
sys.path.insert(0, str(Path('./scratch')))

import joblib
from scratch.patched_fast_offline_eval import evaluate_all, ARTIFACTS_PATH
from data.anomaly_injector import generate_network_benchmark

artifact = joblib.load(ARTIFACTS_PATH)

for s in [42, 101, 202]:
    data = generate_network_benchmark(regime='benchmark_b', seed=s, save_to_disk=False)
    res = evaluate_all(data, artifact, silent=True)
    m = res['__overall__']
    ep = res['__episodic__']
    print(f"Seed {s}: Precision={m['precision']*100:.2f}%, Recall={m['recall']*100:.2f}%, TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, F1*={ep.latency_aware_f1:.4f}, Episodes={ep.detected_episodes}/{ep.total_episodes}")
