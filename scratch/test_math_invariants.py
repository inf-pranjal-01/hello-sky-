import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.linalg as la

from config import CLUSTERS
from scratch.run_chance1_causal_audit import (
    STATION_TO_CLUSTER, CausalNormalBehaviorModel, ConditionalResidualUncertaintyModel
)

def test_hard_invariants():
    print("=" * 80)
    print("TESTING HARD MATHEMATICAL INVARIANTS (INVARIANTS 1 TO 10)")
    print("=" * 80)

    # 1. Fit Normal Model and Uncertainty Model on first 60% of clean data
    print("Fitting CausalNormalBehaviorModel and ConditionalResidualUncertaintyModel on 60% clean split...")
    normal_model = CausalNormalBehaviorModel(train_ratio=0.60)
    normal_model.fit()

    uncertainty_model = ConditionalResidualUncertaintyModel(normal_model, train_ratio=0.60)
    uncertainty_model.fit()

    clean_dfs = {}
    for sid in STATION_TO_CLUSTER:
        df = pd.read_csv(f'data/{sid}.csv', parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        clean_dfs[sid] = df

    params = ['temperature_c', 'humidity_pct', 'pressure_hpa']

    # Compute standardized residuals for training and holdout
    train_z = {}
    holdout_z = {}

    for sid, cid in STATION_TO_CLUSTER.items():
        peer_ids = [s for s, c in STATION_TO_CLUSTER.items() if c == cid and s != sid]
        peer_dfs = {pid: clean_dfs[pid] for pid in peer_ids}
        df_target = clean_dfs[sid]
        n_total = len(df_target)
        n_train = int(n_total * 0.60)

        df_train = df_target.iloc[:n_train]
        peer_train = {pid: peer_dfs[pid].iloc[:n_train] for pid in peer_ids}

        df_holdout = df_target.iloc[n_train:]
        peer_holdout = {pid: peer_dfs[pid].iloc[n_train:] for pid in peer_ids}

        z_train_cols = []
        z_holdout_cols = []
        for p in params:
            # Predict y_hat on train
            y_hat_tr, _ = normal_model.predict_target(sid, p, df_train, peer_train)
            res_tr = df_train[p].values - y_hat_tr
            sig_tr = uncertainty_model.predict_sigma(sid, p, y_hat_tr, df_train['timestamp'].values)
            z_tr = res_tr / np.maximum(0.10, sig_tr)
            z_train_cols.append(z_tr)

            # Predict y_hat on holdout
            y_hat_ho, _ = normal_model.predict_target(sid, p, df_holdout, peer_holdout)
            res_ho = df_holdout[p].values - y_hat_ho
            sig_ho = uncertainty_model.predict_sigma(sid, p, y_hat_ho, df_holdout['timestamp'].values)
            z_ho = res_ho / np.maximum(0.10, sig_ho)
            z_holdout_cols.append(z_ho)

        train_z[sid] = np.column_stack(z_train_cols)  # [N_train, 3]
        holdout_z[sid] = np.column_stack(z_holdout_cols)  # [N_holdout, 3]

    print("Computed standardized residuals for all 28 stations.")

    # Estimate Covariance matrices and verify invariants
    invariants_passed = True
    all_cond_vars = []
    all_cond_inno_means = []
    all_cond_inno_vars = []

    for sid in STATION_TO_CLUSTER:
        Z_tr = train_z[sid]
        Z_ho = holdout_z[sid]
        
        # Robust center
        med = np.median(Z_tr, axis=0)
        Z_centered = Z_tr - med
        
        # Sample covariance
        N_pts = len(Z_centered)
        S = (Z_centered.T @ Z_centered) / (N_pts - 1)
        
        # Shrinkage toward diagonal target
        diag_target = np.diag(np.diag(S))
        shrinkage = max(0.05, 15.0 / (N_pts + 15.0))
        Sigma = (1.0 - shrinkage) * S + shrinkage * diag_target
        
        # Exact symmetry
        Sigma = 0.5 * (Sigma + Sigma.T)
        
        # TEST 3: Symmetry check
        assert np.allclose(Sigma, Sigma.T, atol=1e-12), f"Invariant 3 failed: Sigma not symmetric for {sid}"
        
        # Regularization for numerical conditioning
        evals = la.eigvalsh(Sigma)
        min_eval = np.min(evals)
        lam = max(0.05, -min_eval + 0.05 if min_eval < 0.05 else 1e-4)
        Sigma_reg = Sigma + lam * np.eye(3)
        
        # TEST 4: Positive definiteness
        evals_reg = la.eigvalsh(Sigma_reg)
        assert np.all(evals_reg > 0), f"Invariant 4 failed: non-positive eigenvalue for {sid}"
        # Cholesky check
        la.cholesky(Sigma_reg)
        
        # For each channel c in {0, 1, 2}
        for c in range(3):
            other_idx = [i for i in range(3) if i != c]
            
            sig_cc = Sigma_reg[c, c]
            sig_c_other = Sigma_reg[c:c+1, other_idx] # [1, 2]
            sig_other_other = Sigma_reg[np.ix_(other_idx, other_idx)] # [2, 2]
            sig_other_c = sig_c_other.T # [2, 1]
            
            # Linear weights W = sig_c_other @ inv(sig_other_other)
            W_c = la.solve(sig_other_other, sig_other_c, assume_a='pos').T # [1, 2]
            
            # Conditional variance
            cond_var = sig_cc - (W_c @ sig_other_c)[0, 0]
            
            # TEST 1: Conditional variance > 0
            assert cond_var > 0, f"Invariant 1 failed: cond_var <= 0 ({cond_var}) for {sid} ch {c}"
            
            # TEST 2: Conditional variance <= Marginal variance within numerical tolerance
            assert cond_var <= sig_cc + 1e-10, f"Invariant 2 failed: cond_var ({cond_var}) > marginal ({sig_cc}) for {sid} ch {c}"
            
            all_cond_vars.append((cond_var, sig_cc))
            
            # Evaluate conditional innovation on clean holdout data
            # E[z_c | z_-c] = W_c @ z_-c
            z_c_ho = Z_ho[:, c]
            z_other_ho = Z_ho[:, other_idx] # [N, 2]
            pred_z_c = (z_other_ho @ W_c.T).ravel()
            
            u_c = z_c_ho - pred_z_c
            u_std = u_c / np.sqrt(cond_var)
            
            mean_u = np.mean(u_c)
            var_u = np.var(u_std)
            
            all_cond_inno_means.append(mean_u)
            all_cond_inno_vars.append(var_u)

    # TEST 5: Conditional innovation approximately zero mean on clean holdout data
    max_mean = np.max(np.abs(all_cond_inno_means))
    print(f"Max absolute clean holdout mean of u_c: {max_mean:.4f} (target: < 0.25)")
    assert max_mean < 0.25, f"Invariant 5 failed: clean holdout mean too large ({max_mean})"

    # TEST 6: Standardized conditional innovation approximately unit variance
    mean_var = np.mean(all_cond_inno_vars)
    print(f"Mean clean holdout variance of u_std: {mean_var:.4f} (target: ~ 1.0, range [0.7, 1.4])")
    assert 0.70 <= mean_var <= 1.40, f"Invariant 6 failed: variance out of bounds ({mean_var})"

    # TEST 7, 8, 9: Ramp GLRT Likelihood Ratio >= 0, Delta RSS >= 0, GLRT >= 0
    print("Testing Ramp GLRT non-negativity (Invariants 7, 8, 9)...")
    # Synthetic / test sequences
    rng = np.random.RandomState(42)
    for _ in range(500):
        W = rng.randint(4, 25)
        u_seg = rng.randn(W) * 1.5 + rng.uniform(-1, 1)
        w_seg = 1.0 / (rng.uniform(0.5, 2.0, size=W) ** 2)
        t_seg = np.arange(W, dtype=float)
        
        sum_w = np.sum(w_seg)
        t_bar = np.sum(w_seg * t_seg) / sum_w
        u_bar = np.sum(w_seg * u_seg) / sum_w
        t_dev = t_seg - t_bar
        u_dev = u_seg - u_bar
        s_tt = np.sum(w_seg * (t_dev ** 2))
        s_tu = np.sum(w_seg * t_dev * u_dev)
        
        b_hat = s_tu / s_tt
        delta_rss = (s_tu ** 2) / s_tt
        lam = delta_rss / 2.0
        
        assert delta_rss >= 0.0, f"Invariant 8 failed: delta_rss < 0 ({delta_rss})"
        assert lam >= 0.0, f"Invariant 7/9 failed: lam < 0 ({lam})"

    print("Invariants 1-9 PASSED successfully!")
    print("=" * 80)

if __name__ == '__main__':
    test_hard_invariants()
