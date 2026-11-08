"""Evaluation script for policies trained in cartpole2_safe_rl.py.

This script loads the trained policies (PPO, PPO-Lagrangian, CPO, and PPO+CBF)
and evaluates each of them for a fixed number of episodes inside the
`ConstrainedCartPoleWrapper`. During evaluation we do not enforce any
`steps_per_epoch` limit—episodes terminate only when the environment ends or
the constraint is violated. Per-episode returns, accumulated costs, and
violation counts are reported and saved, together with comparison plots.
"""

import os

# Set environment variable to use legacy Keras (Keras 2) instead of Keras 3
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
from typing import Any, Dict, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Import TensorFlow 1.x compatibility mode (required by safe_rl policies)
try:
    import tensorflow.compat.v1 as tf  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "TensorFlow (v1 compat) is required to load safe_rl policies."
    ) from exc

tf.disable_v2_behavior()

# Ensure safe_rl package is discoverable
SAFE_RL_ROOT = '/home/dmy/gymtest/safety-starter-agents'
if SAFE_RL_ROOT not in sys.path:
    sys.path.append(SAFE_RL_ROOT)

try:
    from safe_rl.utils.load_utils import load_policy  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "safe_rl package not found. Please ensure safety-starter-agents is available."
    ) from exc

from cartpole2_safe_rl import (
    ConstrainedCartPoleWrapper,
    ConstraintViolationCounter,
    MAX_X_DISPLACEMENT,
    RUN_DIR,
    save_index,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EVAL_EPISODES = 60
# Use a very large steps_per_epoch so the wrapper never triggers epoch resets
EVAL_STEPS_PER_EPOCH = int(1e9)

POLICY_SPECS = [
    ("PPO", os.path.join(RUN_DIR, f"ppo_{save_index}")),
    ("PPO-Lagrangian", os.path.join(RUN_DIR, f"ppo_lagrangian_{save_index}")),
    ("CPO", os.path.join(RUN_DIR, f"cpo_{save_index}")),
    ("PPO+CBF", os.path.join(RUN_DIR, f"ppo_cbf_{save_index}")),
    ("PPO+CBF (Reward)", os.path.join(RUN_DIR, f"ppo_cbf_reward_shaping_{save_index}")),
]

EVAL_OUTPUT_DIR = os.path.join(RUN_DIR, f"evaluation_{save_index}")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def gymnasium_reset_compat(env: gym.Env) -> np.ndarray:
    """Handle both gym and gymnasium reset() signatures."""

    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def gymnasium_step_compat(env: gym.Env, action: Any) -> tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """Handle both gym and gymnasium step() signatures."""

    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    obs, reward, done, info = result  # Legacy API
    return obs, float(reward), bool(done), info


def build_evaluation_env() -> ConstrainedCartPoleWrapper:
    """Create a constrained environment without epoch boundaries."""

    counter = ConstraintViolationCounter(x_threshold=MAX_X_DISPLACEMENT)
    base_env = gym.make('InvertedPendulum-v4')

    # Not setting a max episode length ensures termination only on env done or
    # constraint violation. The wrapper still tracks costs/violations.
    env = ConstrainedCartPoleWrapper(
        base_env,
        counter,
        steps_per_epoch=EVAL_STEPS_PER_EPOCH,
    )
    return env


def evaluate_policy(policy_name: str, policy_dir: str, episodes: int = EVAL_EPISODES) -> Optional[Dict[str, Any]]:
    """Evaluate a single trained policy and return detailed statistics."""

    print(f"\nEvaluating {policy_name} policy")
    print(f"  Loading from: {policy_dir}")

    if not os.path.isdir(policy_dir):
        print("  Skipping: directory not found.")
        return None

    try:
        _, get_action, sess = load_policy(policy_dir, itr='last')
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Skipping: could not load policy ({exc}).")
        return None

    env = build_evaluation_env()

    episode_returns: List[float] = []
    episode_costs: List[float] = []
    episode_lengths: List[int] = []
    violation_count = 0

    try:
        obs = gymnasium_reset_compat(env)
        for ep in range(episodes):
            ep_ret = 0.0
            ep_cost = 0.0
            ep_len = 0
            done = False

            while not done:
                action = get_action(obs)
                obs, reward, done, info = gymnasium_step_compat(env, action)
                ep_ret += reward
                ep_cost += float(info.get('cost', 0.0))
                ep_len += 1

            if getattr(env, 'episode_had_violation', False):
                violation_count += 1

            episode_returns.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(ep_len)

            if ep < episodes - 1:
                obs = gymnasium_reset_compat(env)

        # Record final episode inside the counter for completeness
        if hasattr(env, 'counter'):
            env.counter.episode_ended(getattr(env, 'episode_had_violation', False))

    finally:
        env.close()
        sess.close()
        tf.reset_default_graph()

    episode_indices = np.arange(1, len(episode_returns) + 1)
    returns_np = np.asarray(episode_returns, dtype=np.float64)
    costs_np = np.asarray(episode_costs, dtype=np.float64)
    lengths_np = np.asarray(episode_lengths, dtype=np.int32)

    policy_df = pd.DataFrame(
        {
            'Episode': episode_indices,
            'Return': returns_np,
            'Cost': costs_np,
            'Length': lengths_np,
            'Violation': [1 if i < violation_count else 0 for i in range(len(episode_indices))],
        }
    )

    policy_csv = os.path.join(EVAL_OUTPUT_DIR, f"{policy_name.lower().replace(' ', '_')}_episodes_{save_index}.csv")
    policy_df.to_csv(policy_csv, index=False)

    summary = {
        'policy': policy_name,
        'policy_dir': policy_dir,
        'episodes': len(returns_np),
        'mean_return': float(returns_np.mean()) if len(returns_np) else 0.0,
        'std_return': float(returns_np.std()) if len(returns_np) else 0.0,
        'total_cost': float(costs_np.sum()) if len(costs_np) else 0.0,
        'mean_cost': float(costs_np.mean()) if len(costs_np) else 0.0,
        'violations': violation_count,
        'episode_returns': returns_np,
        'episode_costs': costs_np,
        'episode_lengths': lengths_np,
    }

    print(
        "  Results: "
        f"mean return {summary['mean_return']:.2f} ± {summary['std_return']:.2f}, "
        f"total cost {summary['total_cost']:.2f}, violations {violation_count}/{summary['episodes']}"
    )

    return summary


def generate_plots(results: List[Dict[str, Any]]) -> None:
    """Generate comparison plots for returns and accumulated costs."""

    if not results:
        print("No evaluation results to plot.")
        return

    episodes_axis = np.arange(1, EVAL_EPISODES + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for res in results:
        returns = res['episode_returns']
        costs = res['episode_costs']

        axes[0].plot(episodes_axis[: len(returns)], returns, label=res['policy'], linewidth=2)
        axes[1].plot(episodes_axis[: len(costs)], np.cumsum(costs), label=res['policy'], linewidth=2)

    axes[0].set_ylabel('Episode Return')
    axes[0].set_title('Return per Episode')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Cumulative Cost')
    axes[1].set_title('Accumulated Cost across Episodes')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(EVAL_OUTPUT_DIR, f'evaluation_returns_costs_{save_index}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Plots saved to: {plot_path}")


def save_summary(results: List[Dict[str, Any]]) -> None:
    """Save overall summary statistics to CSV."""

    if not results:
        return

    rows = []
    for res in results:
        rows.append(
            {
                'Policy': res['policy'],
                'Episodes': res['episodes'],
                'MeanReturn': res['mean_return'],
                'StdReturn': res['std_return'],
                'TotalCost': res['total_cost'],
                'MeanCost': res['mean_cost'],
                'Violations': res['violations'],
                'PolicyDir': res['policy_dir'],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(EVAL_OUTPUT_DIR, f'evaluation_summary_{save_index}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    """Run evaluation for all trained policies."""

    print("\n" + "=" * 60)
    print("Evaluating trained policies from cartpole2_safe_rl.py")
    print("=" * 60)

    results: List[Dict[str, Any]] = []
    for policy_name, policy_dir in POLICY_SPECS:
        res = evaluate_policy(policy_name, policy_dir)
        if res is not None:
            results.append(res)

    if not results:
        print("No policies were evaluated successfully.")
        return

    print("\nViolation summary:")
    for res in results:
        print(f"  {res['policy']}: {res['mean_return']:.2f} ± {res['std_return']:.2f}")
        print(f"  {res['policy']}: {res['violations']} / {res['episodes']} episodes")

    save_summary(results)
    generate_plots(results)


if __name__ == '__main__':
    main()