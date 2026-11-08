#!/usr/bin/env python3
"""
Combined success-rate plot for inverted_pendulum_ppo_paper.py and
inverted_pendulum_potential_paper.py run directories.
================================================================================

Given one or more run directories (each containing the
`per_seed_success_rates.csv` written by either training script), this script:

  1. Plots the success rate vs timesteps of every policy found across all
     run directories (mean over seeds, +/- std shaded).
  2. Prints, for each policy, the earliest timestep at which it first reaches
     the target success rate (default 90%): the per-seed mean +/- std, the
     single earliest seed, and how many seeds ever reached the target.

Usage:
    python pendulum/paper_plot.py <run_dir> [<run_dir> ...]
    python pendulum/paper_plot.py runs/inverted_pendulum_ppo_standard_vs_energy_vs_effort_100_000 \
                                 runs/inverted_pendulum_ppo_potential_energy_vs_effort_150_000

Options:
    --target  Success-rate target (default 0.90).
    --out     Output plot path (default combined_success_rate_vs_timesteps.pdf
              in the first run directory).
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PER_SEED_CSV = 'per_seed_success_rates.csv'

LABEL_MAP = {
    'Standard': 'Standard reward',
    'Custom': 'Energy-based reward',
    'Effort': 'control effort penalized reward',
    'PBRS_Energy': 'PBRS energy',
    'PBRS_Effort': 'PBRS effort (Phi = -beta*||a||^2)',
}


def load_per_seed(run_dirs: List[str]) -> pd.DataFrame:
    """Load and concatenate per-seed success-rate tables from all run dirs.

    If the same policy name appears in more than one run dir, later
    occurrences are suffixed with the run dir name to keep them distinct.
    """
    frames = []
    seen_policies: Dict[str, str] = {}   # policy name -> run dir it came from
    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, PER_SEED_CSV)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing {PER_SEED_CSV} in {run_dir}")
        df = pd.read_csv(csv_path)
        required = {'environment', 'seed', 'timesteps', 'success_rate'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        rename = {}
        for policy in df['environment'].unique():
            if policy in seen_policies and seen_policies[policy] != run_dir:
                rename[policy] = f"{policy} [{os.path.basename(os.path.normpath(run_dir))}]"
            else:
                seen_policies[policy] = run_dir
        if rename:
            df['environment'] = df['environment'].replace(rename)
            for old, new in rename.items():
                print(f"Note: policy '{old}' appears in multiple run dirs; "
                      f"renamed to '{new}' for {run_dir}")

        df['run_dir'] = run_dir
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_curves(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """Per policy: mean/std success rate over seeds on that policy's eval grid.

    Seeds of the same policy share the eval grid, but different run dirs may
    use different eval intervals or horizons, so each policy keeps its own
    timestep grid (union of its seeds' eval points, seed curves interpolated).
    """
    aggregated: Dict[str, Dict[str, np.ndarray]] = {}
    for policy, group in df.groupby('environment'):
        grid = np.sort(group['timesteps'].unique()).astype(np.float64)
        curves = []
        for _seed, seed_df in group.groupby('seed'):
            seed_df = seed_df.sort_values('timesteps')
            ts = seed_df['timesteps'].to_numpy(dtype=np.float64)
            sr = seed_df['success_rate'].to_numpy(dtype=np.float64)
            ts_u, idx = np.unique(ts, return_index=True)
            curves.append(np.interp(grid, ts_u, sr[idx],
                                    left=sr[idx][0], right=sr[idx][-1]))
        stack = np.stack(curves, axis=0)
        aggregated[policy] = {
            'timesteps': grid,
            'mean_success': np.mean(stack, axis=0),
            'std_success': np.std(stack, axis=0),
            'n_seeds': stack.shape[0],
        }
    return aggregated


def timesteps_to_target(df: pd.DataFrame, target: float
                        ) -> Dict[str, Tuple[float, float, float, int, int]]:
    """Per policy: (mean, std, earliest) first timestep with success_rate >= target,
    plus (n_reached, n_total) seed counts."""
    result: Dict[str, Tuple[float, float, float, int, int]] = {}
    for policy, group in df.groupby('environment'):
        reached = []
        n_total = group['seed'].nunique()
        for _seed, seed_df in group.groupby('seed'):
            seed_df = seed_df.sort_values('timesteps')
            hits = seed_df.loc[seed_df['success_rate'] >= target, 'timesteps']
            if not hits.empty:
                reached.append(float(hits.iloc[0]))
        if reached:
            result[policy] = (float(np.mean(reached)), float(np.std(reached)),
                              float(np.min(reached)), len(reached), n_total)
        else:
            result[policy] = (float('nan'), float('nan'), float('nan'), 0, n_total)
    return result


def plot_curves(aggregated: Dict[str, Dict[str, np.ndarray]], target: float,
                save_path: str):
    plt.figure(figsize=(7, 4.2))
    for policy in sorted(aggregated):
        stats = aggregated[policy]
        ts = stats['timesteps']
        mean = stats['mean_success']
        std = stats['std_success']
        plt.plot(ts, mean, label=LABEL_MAP.get(policy, policy), linewidth=2)
        plt.fill_between(ts, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                         alpha=0.2)
    plt.axhline(target, color='gray', linestyle='--', linewidth=1,
                label=f'{int(target*100)}% target')
    plt.xlabel('Timesteps')
    plt.ylabel('Success rate')
    plt.title('InvertedPendulum: Success rate vs timesteps (success = episode length > 950)')
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCombined success-rate plot saved to: {save_path}")


def print_target_summary(to_target: Dict[str, Tuple[float, float, float, int, int]],
                         target: float):
    width = max(len(LABEL_MAP.get(p, p)) for p in to_target)
    print(f"\nEarliest timesteps to reach {int(target*100)}% success rate:")
    for policy in sorted(to_target):
        mean, std, earliest, n_reached, n_total = to_target[policy]
        label = LABEL_MAP.get(policy, policy)
        if n_reached == 0:
            print(f"  - {label:{width}s}: never reached (0/{n_total} seeds)")
        else:
            print(f"  - {label:{width}s}: earliest {earliest:10,.0f}  |  "
                  f"mean {mean:10,.0f} +/- {std:8,.0f} timesteps  "
                  f"({n_reached}/{n_total} seeds)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot combined success-rate curves from run directories of "
                    "inverted_pendulum_ppo_paper.py and "
                    "inverted_pendulum_potential_paper.py.")
    parser.add_argument('run_dirs', nargs='+',
                        help='Run directories containing per_seed_success_rates.csv')
    parser.add_argument('--target', type=float, default=0.90,
                        help='Success-rate target (default: 0.90)')
    parser.add_argument('--out', default=None,
                        help='Output plot path (default: '
                             'combined_success_rate_vs_timesteps.pdf in first run dir)')
    args = parser.parse_args()

    df = load_per_seed(args.run_dirs)
    policies = sorted(df['environment'].unique())
    print(f"Loaded {len(policies)} policies from {len(args.run_dirs)} run dir(s): "
          f"{policies}")

    aggregated = aggregate_curves(df)
    out_path = args.out or os.path.join(args.run_dirs[0],
                                        'combined_success_rate_vs_timesteps.pdf')
    plot_curves(aggregated, args.target, out_path)

    to_target = timesteps_to_target(df, args.target)
    print_target_summary(to_target, args.target)


if __name__ == "__main__":
    main()
