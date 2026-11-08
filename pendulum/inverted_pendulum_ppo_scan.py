#!/usr/bin/env python3
"""
Parameter scan for the shaped reward r = 1 - alpha * E on InvertedPendulum-v4.

Sweeps the energy coefficient alpha and selects the best value by the fewest
timesteps to first reach a >90% success rate (tie-break: higher final success
rate). Reuses the training / evaluation machinery from
`inverted_pendulum_ppo_paper.py`.

Run:
    python pendulum/inverted_pendulum_ppo_scan.py          # full scan
    python pendulum/inverted_pendulum_ppo_scan.py --quick  # tiny smoke test
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.utils import set_random_seed

import inverted_pendulum_ppo_paper as ip

# -------------------- Scan configuration --------------------
ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0]
SCAN_NUM_SEEDS = 3
SCAN_TOTAL_TIMESTEPS = 100_000
SCAN_EVAL_INTERVAL = 5_000
SCAN_EVAL_EPISODES = 100
SCAN_DIR = "runs/inverted_pendulum_ppo_scan"


def _seed_first_reaches_target(callback) -> float | None:
    """First eval timestep where success rate >= target, else None."""
    for t, r in zip(callback.timesteps, callback.success_rates):
        if r >= ip.SUCCESS_RATE_TARGET:
            return t
    return None


def run_scan():
    os.makedirs(SCAN_DIR, exist_ok=True)

    # Push scan config into the reused module.
    ip.NUM_SEEDS = SCAN_NUM_SEEDS
    ip.SEEDS = [ip.BASE_SEED + i for i in range(SCAN_NUM_SEEDS)]
    ip.TOTAL_TIMESTEPS = SCAN_TOTAL_TIMESTEPS
    ip.EVAL_INTERVAL = SCAN_EVAL_INTERVAL
    ip.EVAL_EPISODES = SCAN_EVAL_EPISODES
    ip.EVAL_RESET_SEEDS = [ip.BASE_SEED * 1000 + i for i in range(SCAN_EVAL_EPISODES)]

    rows = []                                   # per (alpha, seed)
    curves_by_alpha = {}                        # alpha -> list of callbacks

    for alpha in ALPHAS:
        ip.ENERGY_COEF = alpha
        print(f"\n{'='*64}\nScanning alpha = {alpha}\n{'='*64}")
        callbacks = []
        for seed in ip.SEEDS:
            set_random_seed(seed)
            model, cb = ip.train_one('Custom', seed)
            callbacks.append(cb)

            to_target = _seed_first_reaches_target(cb)
            final_sr = cb.success_rates[-1] if cb.success_rates else float('nan')
            rows.append({
                'alpha': alpha,
                'seed': seed,
                'timesteps_to_90': to_target if to_target is not None else np.nan,
                'reached_90': to_target is not None,
                'final_success_rate': final_sr,
            })
        curves_by_alpha[alpha] = callbacks

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SCAN_DIR, 'scan_per_seed.csv'), index=False)

    # -------- Aggregate per alpha --------
    summary = []
    for alpha in ALPHAS:
        sub = df[df['alpha'] == alpha]
        n_reached = int(sub['reached_90'].sum())
        reached_ts = sub.loc[sub['reached_90'], 'timesteps_to_90']
        mean_ts = float(reached_ts.mean()) if n_reached > 0 else np.nan
        std_ts = float(reached_ts.std(ddof=0)) if n_reached > 0 else np.nan
        mean_final = float(sub['final_success_rate'].mean())
        std_final = float(sub['final_success_rate'].std(ddof=0))
        summary.append({
            'alpha': alpha,
            'n_reached_90': n_reached,
            'n_seeds': len(sub),
            'mean_timesteps_to_90': mean_ts,
            'std_timesteps_to_90': std_ts,
            'mean_final_success_rate': mean_final,
            'std_final_success_rate': std_final,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(SCAN_DIR, 'scan_summary.csv'), index=False)

    # -------- Selection: fewest timesteps-to-90 (more seeds reaching wins first),
    #          tie-break higher final success rate --------
    def sort_key(r):
        # Prefer more seeds reaching target; then lower mean timesteps; then higher final SR.
        mean_ts = r['mean_timesteps_to_90']
        mean_ts = np.inf if np.isnan(mean_ts) else mean_ts
        return (-r['n_reached_90'], mean_ts, -r['mean_final_success_rate'])

    ranked = sorted(summary, key=sort_key)
    best = ranked[0]

    _plot_curves(curves_by_alpha)
    _plot_timesteps_to_90(summary)
    _print_summary(summary_df, best)
    return best


def _plot_curves(curves_by_alpha):
    grid = np.arange(SCAN_EVAL_INTERVAL, SCAN_TOTAL_TIMESTEPS + SCAN_EVAL_INTERVAL,
                     SCAN_EVAL_INTERVAL, dtype=np.float64)
    plt.figure(figsize=(10, 6))
    for alpha, callbacks in curves_by_alpha.items():
        interp = []
        for cb in callbacks:
            ts = np.asarray(cb.timesteps, dtype=np.float64)
            sr = np.asarray(cb.success_rates, dtype=np.float64)
            if len(ts) == 0:
                continue
            ts_u, idx = np.unique(ts, return_index=True)
            interp.append(np.interp(grid, ts_u, sr[idx], left=sr[idx][0], right=sr[idx][-1]))
        if not interp:
            continue
        stack = np.stack(interp, axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        plt.plot(grid, mean, linewidth=2, label=f'alpha={alpha}')
        plt.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15)
    plt.axhline(ip.SUCCESS_RATE_TARGET, color='gray', linestyle='--', linewidth=1,
                label=f'{int(ip.SUCCESS_RATE_TARGET*100)}% target')
    plt.xlabel('Timesteps')
    plt.ylabel('Success rate')
    plt.title('Shaped reward r = 1 - alpha*E: success rate vs timesteps')
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(SCAN_DIR, 'scan_success_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Curve plot saved to: {path}")


def _plot_timesteps_to_90(summary):
    alphas = [s['alpha'] for s in summary]
    means = [s['mean_timesteps_to_90'] for s in summary]
    stds = [s['std_timesteps_to_90'] for s in summary]
    x = np.arange(len(alphas))
    plt.figure(figsize=(9, 5))
    plt.bar(x, [m if not np.isnan(m) else 0 for m in means],
            yerr=[s if not np.isnan(s) else 0 for s in stds], capsize=5)
    for i, (m, s) in enumerate(zip(means, summary)):
        if np.isnan(m):
            plt.text(i, 0, 'never\nreached', ha='center', va='bottom', fontsize=8)
    plt.xticks(x, [str(a) for a in alphas])
    plt.xlabel('alpha (energy coefficient)')
    plt.ylabel(f'Timesteps to first >{int(ip.SUCCESS_RATE_TARGET*100)}% success')
    plt.title('Sample efficiency vs alpha (lower = better)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(SCAN_DIR, 'scan_timesteps_to_90.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Timesteps-to-90 plot saved to: {path}")


def _print_summary(summary_df, best):
    print(f"\n{'='*64}\nPARAMETER SCAN SUMMARY (r = 1 - alpha*E)\n{'='*64}")
    print(f"  Seeds/alpha   : {SCAN_NUM_SEEDS}")
    print(f"  Timesteps     : {SCAN_TOTAL_TIMESTEPS:,}")
    print(f"  Eval          : every {SCAN_EVAL_INTERVAL:,} steps, "
          f"{SCAN_EVAL_EPISODES} episodes, success = len > {ip.SUCCESS_THRESHOLD}\n")
    with pd.option_context('display.float_format', lambda v: f'{v:,.2f}'):
        print(summary_df.to_string(index=False))
    print(f"\n>>> BEST alpha = {best['alpha']}  "
          f"({best['n_reached_90']}/{best['n_seeds']} seeds reached "
          f"{int(ip.SUCCESS_RATE_TARGET*100)}%, "
          f"mean timesteps-to-90 = "
          f"{best['mean_timesteps_to_90']:,.0f}, "
          f"final success = {best['mean_final_success_rate']*100:.1f}%)")
    print(f"\nResults saved in: {SCAN_DIR}")


def _apply_quick_config():
    global ALPHAS, SCAN_NUM_SEEDS, SCAN_TOTAL_TIMESTEPS, SCAN_EVAL_INTERVAL
    global SCAN_EVAL_EPISODES, SCAN_DIR
    ALPHAS = [0.1, 1.0]
    SCAN_NUM_SEEDS = 1
    SCAN_TOTAL_TIMESTEPS = 4_000
    SCAN_EVAL_INTERVAL = 2_000
    SCAN_EVAL_EPISODES = 5
    SCAN_DIR = "runs/inverted_pendulum_ppo_scan_quicktest"
    ip.SUCCESS_THRESHOLD = 50   # low bar so smoke test can show non-zero rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Tiny smoke test.')
    args = parser.parse_args()
    if args.quick:
        _apply_quick_config()
    run_scan()
