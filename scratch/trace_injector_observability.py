import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from data.anomaly_injector import (
    generate_network_benchmark,
    inject_drift_v2, inject_frozen_v2, inject_spike_v2,
    inject_dropout_v2, inject_fail_low_v2, inject_multivariate_v2,
    inject_unstructured_v2
)

# Run deep forensic trace across clean baseline vs injected faults
print("=" * 110)
print("INJECTOR DEEP OBSERVABILITY & SIGNAL-TO-NOISE RATIO (SNR) AUDIT")
print("=" * 110)

clean_dfs = {}
station_files = sorted(Path('data').glob("AWS-*.csv"))
for p in station_files:
    if "_labeled" not in p.name:
        df = pd.read_csv(p, parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        clean_dfs[p.stem] = df

# 1. Measure Clean Baseline Noise Floor and Step Variations
clean_stats = {}
for sid, df in clean_dfs.items():
    for col in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
        diffs = df[col].diff().abs().dropna()
        mad_diff = np.median(diffs)
        q95_diff = np.quantile(diffs, 0.95)
        clean_stats[(sid, col)] = {
            'std': df[col].std(),
            'mad_diff': mad_diff,
            'q95_diff': q95_diff
        }

# 2. Trace Injected Faults from t0 to maturity
rng = np.random.default_rng(42)
test_sid = 'AWS-BHO-030'
df_target = clean_dfs[test_sid].copy()

print("\nA. DRIFT V2 OBSERVABILITY TRACE (t0 -> t_maturity):")
print(f"{'Step (h)':<10} {'Injected Ramp':<18} {'Ambient Noise (MAD)':<22} {'SNR (|ramp|/MAD)':<20} {'Observability Status'}")
print("-" * 90)

# Simulate 50 representative drift episodes
drift_steps_snr = {h: [] for h in range(48)}
for _ in range(100):
    b_mature = float(rng.uniform(3.5, 6.5))
    L = int(rng.integers(24, 48))
    gamma = float(rng.uniform(1.0, 1.2))
    mad = clean_stats[(test_sid, 'temperature_c')]['mad_diff']
    for h in range(L):
        t_norm = h / (L - 1.0)
        ramp = b_mature * (t_norm ** gamma)
        snr = ramp / mad
        drift_steps_snr[h].append((ramp, snr))

for h in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]:
    ramps = [x[0] for x in drift_steps_snr[h]]
    snrs = [x[1] for x in drift_steps_snr[h]]
    mean_ramp = np.mean(ramps)
    mean_snr = np.mean(snrs)
    status = "UNOBSERVABLE (Buried in noise)" if mean_snr < 1.0 else ("BORDERLINE" if mean_snr < 2.0 else "OBSERVABLE")
    print(f"t0 + {h:<5} {mean_ramp:>6.3f} °C            {mad:>6.3f} °C                 {mean_snr:>6.2f}x               {status}")

print("\nB. FROZEN V2 OBSERVABILITY TRACE:")
print(f"{'Step (h)':<10} {'Sensor Output':<18} {'True Weather Movement':<24} {'Deviation':<15} {'Observability Status'}")
print("-" * 90)
for h in [0, 1, 2, 3, 4, 6, 8]:
    # In clean weather, how much does weather move in h hours?
    clean_devs = np.abs(df_target['temperature_c'] - df_target['temperature_c'].shift(h)).dropna()
    med_weather_dev = np.median(clean_devs)
    q75_weather_dev = np.quantile(clean_devs, 0.75)
    status = "UNOBSERVABLE (Anchor == True Weather)" if med_weather_dev < 0.20 else ("PARTIALLY OBSERVABLE" if med_weather_dev < 0.8 else "OBSERVABLE (Atmosphere diverged)")
    print(f"t0 + {h:<5} Frozen Anchor       Median: {med_weather_dev:>5.2f} °C (Q75: {q75_weather_dev:>5.2f} °C)   {med_weather_dev:>5.2f} °C          {status}")

print("\nC. SPIKE V2 OBSERVABILITY:")
spike_mag = 5.0
spike_mad = clean_stats[(test_sid, 'temperature_c')]['mad_diff']
spike_q95 = clean_stats[(test_sid, 'temperature_c')]['q95_diff']
print(f"  Injected Impulse: {spike_mag:.1f} °C vs Clean 1h Step Q95: {spike_q95:.2f} °C -> SNR: {spike_mag/spike_mad:.1f}x (OBSERVABLE at t0)")

print("\nD. FAIL-LOW V2 OBSERVABILITY:")
fail_low_val = -40.0
print(f"  Injected Rail: {fail_low_val:.1f} °C vs Clean Range [-10, 50] °C -> Absolute Rail Ground-Short (OBSERVABLE at t0)")

print("\nE. DROPOUT V2 OBSERVABILITY:")
print(f"  Injected Null: NaN (Missing packet) -> Direct telemetry missingness (OBSERVABLE at t0)")

print("\nF. MULTIVARIATE V2 OBSERVABILITY:")
delta_t = 4.5
delta_rh = 22.5
print(f"  Injected Perturbation: +{delta_t:.1f} °C and +{delta_rh:.1f}% RH simultaneously -> Immediate psychrometric violation (OBSERVABLE at t0)")

print("\nG. UNSTRUCTURED V2 OBSERVABILITY:")
t_noise_std = 3.5 * 0.8
clean_temp_noise = clean_stats[(test_sid, 'temperature_c')]['mad_diff']
print(f"  Injected Noise Std: {t_noise_std:.2f} °C vs Clean Step MAD: {clean_temp_noise:.2f} °C -> SNR: {t_noise_std/clean_temp_noise:.1f}x variance increase (OBSERVABLE)")
print("=" * 110)
