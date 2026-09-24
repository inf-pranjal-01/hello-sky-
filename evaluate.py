"""
SkyGuard AI — Canonical Production Evaluator Entrypoint.

This is the single public entrypoint for executing end-to-end benchmark
evaluations across all operational and adversarial evaluation regimes.

Regimes evaluated:
1. Locked Baseline: Evaluated against the canonical benchmark dataset.
2. Benchmark A: Adversarial multi-fault stress test (unrestricted same-cluster faults).
3. Benchmark B: PCL-compatible operational benchmark (max 1 active fault per cluster).
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.fast_offline_eval import evaluate_all, ARTIFACTS_PATH, DATA_DIR
from data.anomaly_injector import generate_network_benchmark, RANDOM_SEED


def format_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def format_f1(val: float) -> str:
    return f"{val:.4f}"


def run_evaluation():
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {ARTIFACTS_PATH}. Run training first."
        )

    artifact = joblib.load(ARTIFACTS_PATH)
    print("=" * 68)
    print("SKYGUARD AI — FINAL EVALUATION")
    print("=" * 68)

    # 1. Locked Baseline Evaluation (from disk)
    labeled_files = sorted(DATA_DIR.glob("*_labeled.csv"))
    if not labeled_files:
        raise FileNotFoundError(f"No labeled CSV files found in {DATA_DIR}")

    print("\n--- 1. Evaluating Locked Baseline (Canonical Benchmark) ---")
    res_baseline = evaluate_all(labeled_files, artifact, silent=True)
    m_base = res_baseline["__overall__"]
    ep_base = res_baseline["__episodic__"]

    print(f"  TP:                      {m_base['tp']}")
    print(f"  FP:                      {m_base['fp']}")
    print(f"  FN:                      {m_base['fn']}")
    print(f"  TN:                      {m_base['tn']}")
    print(f"  Point Precision:         {format_pct(m_base['precision'])}")
    print(f"  Point Recall:            {format_pct(m_base['recall'])}")
    print(f"  Point F1:                {format_f1(m_base['f1'])}")
    print(f"  Latency-aware Recall*:   {format_pct(ep_base.latency_aware_recall)}")
    print(f"  Latency-aware F1*:       {format_f1(ep_base.latency_aware_f1)}")
    print(f"  Episode Detection Rate:  {ep_base.detected_episodes}/{ep_base.total_episodes} ({format_pct(ep_base.episode_detection_rate)})")
    lat_str = f"{ep_base.mean_latency_hours:.2f} hours" if ep_base.mean_latency_hours is not None else "N/A"
    print(f"  Mean Detection Latency:  {lat_str}")

    # 2. Benchmark A Evaluation (Adversarial Multi-Fault Stress Test)
    print("\n--- 2. Evaluating Benchmark A (Adversarial Multi-Fault Stress Test) ---")
    print("  Regime: Unrestricted Multi-Fault Stress Test (Concurrent cluster faults allowed)")
    data_a = generate_network_benchmark(
        regime="benchmark_a", seed=RANDOM_SEED, save_to_disk=False
    )
    res_a = evaluate_all(data_a, artifact, silent=True)
    m_a = res_a["__overall__"]
    ep_a = res_a["__episodic__"]

    print(f"  TP:                      {m_a['tp']}")
    print(f"  FP:                      {m_a['fp']}")
    print(f"  FN:                      {m_a['fn']}")
    print(f"  Point Precision:         {format_pct(m_a['precision'])}")
    print(f"  Point Recall:            {format_pct(m_a['recall'])}")
    print(f"  Point F1:                {format_f1(m_a['f1'])}")
    print(f"  Latency-aware F1*:       {format_f1(ep_a.latency_aware_f1)}")
    print(f"  Episode Detection Rate:  {ep_a.detected_episodes}/{ep_a.total_episodes} ({format_pct(ep_a.episode_detection_rate)})")

    # 3. Benchmark B Evaluation (PCL-Compatible Operational Benchmark)
    print("\n--- 3. Evaluating Benchmark B (PCL-Compatible Operational Benchmark) ---")
    print("  Regime: PCL-Compatible Operational Benchmark")
    print("  Constraint: Maximum 1 active fault per cluster per timestamp")
    data_b = generate_network_benchmark(
        regime="benchmark_b", seed=RANDOM_SEED, save_to_disk=False
    )
    res_b = evaluate_all(data_b, artifact, silent=True)
    m_b = res_b["__overall__"]
    ep_b = res_b["__episodic__"]

    print(f"  TP:                      {m_b['tp']}")
    print(f"  FP:                      {m_b['fp']}")
    print(f"  FN:                      {m_b['fn']}")
    print(f"  Point Precision:         {format_pct(m_b['precision'])}")
    print(f"  Point Recall:            {format_pct(m_b['recall'])}")
    print(f"  Point F1:                {format_f1(m_b['f1'])}")
    print(f"  Latency-aware F1*:       {format_f1(ep_b.latency_aware_f1)}")
    print(f"  Episode Detection Rate:  {ep_b.detected_episodes}/{ep_b.total_episodes} ({format_pct(ep_b.episode_detection_rate)})")

    # 4. Final Summary Table
    print("\n" + "=" * 68)
    print("SKYGUARD AI — FINAL BENCHMARK RESULTS")
    print("=" * 68)
    print(f"\n{'Regime':<30} {'Precision':<13} {'Recall':<12} {'F1*':<8}")
    print("-" * 68)
    print(f"{'Locked Baseline':<30} {format_pct(m_base['precision']):<13} {format_pct(m_base['recall']):<12} {format_f1(ep_base.latency_aware_f1):<8}")
    print(f"{'Benchmark A — Stress Test':<30} {format_pct(m_a['precision']):<13} {format_pct(m_a['recall']):<12} {format_f1(ep_a.latency_aware_f1):<8}")
    print(f"{'Benchmark B — Operational':<30} {format_pct(m_b['precision']):<13} {format_pct(m_b['recall']):<12} {format_f1(ep_b.latency_aware_f1):<8}")
    print("-" * 68)
    print("\nCANONICAL OPERATIONAL RESULT")
    print(f"Precision: {format_pct(m_b['precision'])}")
    print(f"Recall:    {format_pct(m_b['recall'])}")
    print("\n" + "=" * 68)


if __name__ == "__main__":
    run_evaluation()
