import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from model.fault_helper import (
    make_sparse_training_replays, fit_fault_helper, predict_faults,
    add_frozen_channel_labels_from_reference, fit_frozen_channel_helpers,
    score_frozen_channels,
)

DATA_DIR = PROJECT_ROOT / "data"
TRAINING_SEEDS = [1101, 2202]
ALERT_THRESHOLD = 0.75
FROZEN_ALERT_THRESHOLD = 0.90


def main():
    labeled = sorted(DATA_DIR.glob("*_labeled.csv"))
    if not labeled:
        raise FileNotFoundError("No labelled replay files found.")
    test = pd.concat([pd.read_csv(path, parse_dates=["timestamp"]) for path in labeled], ignore_index=True)
    test = add_frozen_channel_labels_from_reference(test)
    held_out = set(test.loc[test["is_anomaly"].fillna(False), "station_id"])
    if not held_out:
        raise ValueError("The labelled replay has no injected anomalies.")

    print(f"Held-out faulty stations: {', '.join(sorted(held_out))}")
    print(f"Training sparse fresh replays with seeds: {TRAINING_SEEDS}")
    training = make_sparse_training_replays(held_out, TRAINING_SEEDS)
    model, columns = fit_fault_helper(training)
    scored = predict_faults(model, columns, test, ALERT_THRESHOLD)
    frozen_helpers = fit_frozen_channel_helpers(training)
    scored = score_frozen_channels(scored, frozen_helpers, FROZEN_ALERT_THRESHOLD)

    truth = scored["is_anomaly"].to_numpy(dtype=bool)
    predicted = (scored["helper_alert"] | scored["frozen_helper_alert"]).to_numpy(dtype=bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\nHelper-only @ {ALERT_THRESHOLD:.2f}; frozen specialist @ {FROZEN_ALERT_THRESHOLD:.2f}: TP={tp} FP={fp} FN={fn}")
    print(f"Precision={precision:.3%}  Recall={recall:.3%}")
    print("\nRecall by injected fault type:")
    for fault_type, group in scored[truth].groupby("fault_type"):
        caught = int(group["helper_alert"].sum())
        print(f"  {fault_type:<28} {caught}/{len(group)} ({caught / len(group):.1%})")


if __name__ == "__main__":
    main()
