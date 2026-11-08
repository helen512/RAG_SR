#!/usr/bin/env python3
"""
InvertedPendulum PPO: Standard reward vs physics-based shaped reward (r = 1 - E).
================================================================================

Compares two reward designs on the OpenAI/Gymnasium MuJoCo InvertedPendulum-v4
environment using PPO from Stable Baselines3:

  1. Standard reward: +1 per timestep while the pole stays within the allowed angle.
  2. Shaped reward:   r = 1 - E, where E is the total mechanical energy of the
     cart-pole system (potential energy zero at upright).

Total energy:
    E = 1/2 [ (M + m) x_dot^2
              + m x_dot l theta_dot
              + 1/3 m l^2 theta_dot^2
              + m l g (1 - cos(theta)) ]

A task is considered a SUCCESS if the episode lasts longer than 950 timesteps.

Every EVAL_INTERVAL training steps, the current policy is evaluated over
EVAL_EPISODES episodes and the success rate is recorded, producing a
success-rate-vs-timesteps curve. Each experiment runs across NUM_SEEDS seeds;
the shaded area in the plot is the standard deviation across seeds.

After training, the final policy of every seed is evaluated and we report the
mean success rate +/- std, plus how many timesteps the policy needs to first
reach a >90% success rate.

PPO hyperparameters are taken from cartpole_ppo2.py.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Tuple, Dict, Any, List

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# -------------------- Configuration --------------------
BASE_SEED = 42
NUM_SEEDS = 5
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]
TOTAL_TIMESTEPS = 150_000          # training timesteps per environment per seed
EVAL_INTERVAL = 5_000             # evaluate the policy every this many train steps
EVAL_EPISODES = 100                # episodes per evaluation
SUCCESS_THRESHOLD = 950            # episode length strictly greater => success
SUCCESS_RATE_TARGET = 0.90         # report timesteps to first reach this success rate
MAX_EPISODE_LENGTH = 1000          # InvertedPendulum-v4 truncates here

# Fixed reset seeds for evaluation so the success-rate curve is comparable over time.
EVAL_RESET_SEEDS = [BASE_SEED * 1000 + i for i in range(EVAL_EPISODES)]

RUN_DIR = "runs/inverted_pendulum_ppo_standard_vs_energy_vs_effort_100_000"

# Physical constants for the energy term (match cartpole_ppo2.py / InvertedPendulum-v4).
G = 9.81
CART_MASS = 10.472   # M
POLE_MASS = 5.019    # m
POLE_LEN = 0.3       # l (half-pole length used in cartpole_ppo2.py)

ENERGY_COEF = 0.1    # alpha in shaped reward r = 1 - alpha * E
# beta in effort-penalized reward r = standard_reward - beta * ||action||^2.
# Chosen so beta * mean(action^2) ~ ENERGY_COEF * mean(E) over trajectories:
# empirically the effort term then matches the energy term in magnitude
# (measured scalar ~0.10 for a random policy, ~0.18 for a trained policy).
EFFORT_COEF = 0.1


# -------------------- Energy / shaped reward --------------------
def total_energy(observation: np.ndarray) -> float:
    """Total mechanical energy E of the cart-pole (potential zero at upright)."""
    _x, x_dot, theta, theta_dot = observation
    return 0.5 * (
        (CART_MASS + POLE_MASS) * x_dot**2
        + POLE_MASS * x_dot * POLE_LEN * theta_dot
        + (1.0 / 3.0) * POLE_MASS * POLE_LEN**2 * theta_dot**2
        + POLE_MASS * POLE_LEN * G * (1.0 - np.cos(theta))
    )


class ShapedRewardInvertedPendulum(gym.Wrapper):
    """Replaces the +1 reward with the physics-based shaped reward r = 1 - alpha*E."""

    def step(self, action):
        observation, _orig_reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = 1.0 - ENERGY_COEF * total_energy(observation)
        return observation, shaped_reward, terminated, truncated, info


class EffortPenalizedInvertedPendulum(gym.Wrapper):
    """Standard reward minus a control-effort penalty: r = r_std - beta*||action||^2."""

    def step(self, action):
        observation, orig_reward, terminated, truncated, info = self.env.step(action)
        control_effort = float(np.sum(np.square(action)))
        shaped_reward = orig_reward - EFFORT_COEF * control_effort
        return observation, shaped_reward, terminated, truncated, info


# -------------------- PPO params (from cartpole_ppo2.py) --------------------
def get_optimal_ppo_params(seed: int) -> Dict[str, Any]:
    return {
        'seed': seed,
        'verbose': 0,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(net_arch=[64, 64]),
    }


# -------------------- Evaluation helpers --------------------
def make_train_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make('InvertedPendulum-v4')
    if env_name == 'Custom':
        env = ShapedRewardInvertedPendulum(env)
    elif env_name == 'Effort':
        env = EffortPenalizedInvertedPendulum(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def evaluate_success_rate(model: PPO, eval_env: gym.Env, eval_episodes: int,
                          eval_seeds: List[int]) -> float:
    """Fraction of episodes whose length is strictly greater than SUCCESS_THRESHOLD.

    Termination dynamics are identical for both reward designs, so a plain
    InvertedPendulum env is used to measure survival length.
    """
    successes = 0
    for i in range(eval_episodes):
        obs, _ = eval_env.reset(seed=eval_seeds[i])
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _ = eval_env.step(action)
            steps += 1
            done = terminated or truncated
        if steps > SUCCESS_THRESHOLD:
            successes += 1
    return successes / eval_episodes


class SuccessRateCallback(BaseCallback):
    """Evaluates success rate every EVAL_INTERVAL steps during training."""

    def __init__(self, eval_interval: int, eval_episodes: int,
                 eval_seeds: List[int], verbose: int = 0):
        super().__init__(verbose)
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_seeds = eval_seeds
        self.timesteps: List[int] = []
        self.success_rates: List[float] = []
        self._eval_env: gym.Env | None = None
        self._next_eval = eval_interval

    def _init_callback(self) -> None:
        # Plain env (survival length only) reused across evaluations.
        self._eval_env = gym.make('InvertedPendulum-v4')

    def _run_eval(self) -> None:
        rate = evaluate_success_rate(
            self.model, self._eval_env, self.eval_episodes, self.eval_seeds
        )
        self.timesteps.append(self.num_timesteps)
        self.success_rates.append(rate)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._run_eval()
            self._next_eval += self.eval_interval
        return True

    def _on_training_end(self) -> None:
        # Ensure a final evaluation at the end of training.
        if not self.timesteps or self.timesteps[-1] != self.num_timesteps:
            self._run_eval()
        if self._eval_env is not None:
            self._eval_env.close()


# -------------------- Training --------------------
def train_one(env_name: str, seed: int) -> Tuple[PPO, SuccessRateCallback]:
    print(f"  Training {env_name} (seed {seed})")
    env = make_train_env(env_name, seed)
    model = PPO(policy='MlpPolicy', env=env, **get_optimal_ppo_params(seed))
    callback = SuccessRateCallback(EVAL_INTERVAL, EVAL_EPISODES, EVAL_RESET_SEEDS)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    env.close()
    return model, callback


# -------------------- Aggregation --------------------
def aggregate_success_curves(callbacks: Dict[str, List[SuccessRateCallback]]
                             ) -> Dict[str, Dict[str, np.ndarray]]:
    grid = np.arange(EVAL_INTERVAL, TOTAL_TIMESTEPS + EVAL_INTERVAL, EVAL_INTERVAL,
                     dtype=np.float64)
    aggregated: Dict[str, Dict[str, np.ndarray]] = {}
    for env_name, env_cbs in callbacks.items():
        curves = []
        for cb in env_cbs:
            ts = np.asarray(cb.timesteps, dtype=np.float64)
            sr = np.asarray(cb.success_rates, dtype=np.float64)
            order = np.argsort(ts)
            ts, sr = ts[order], sr[order]
            ts_u, idx = np.unique(ts, return_index=True)
            curves.append(np.interp(grid, ts_u, sr[idx],
                                    left=sr[idx][0], right=sr[idx][-1]))
        stack = np.stack(curves, axis=0)
        aggregated[env_name] = {
            'timesteps': grid,
            'mean_success': np.mean(stack, axis=0),
            'std_success': np.std(stack, axis=0),
        }
    return aggregated


def timesteps_to_target(callbacks: Dict[str, List[SuccessRateCallback]]
                        ) -> Dict[str, Tuple[float, float, int, int]]:
    """First eval timestep reaching SUCCESS_RATE_TARGET, per env: mean, std, n_reached, n_total."""
    result: Dict[str, Tuple[float, float, int, int]] = {}
    for env_name, env_cbs in callbacks.items():
        reached = []
        for cb in env_cbs:
            hit = next((t for t, r in zip(cb.timesteps, cb.success_rates)
                        if r >= SUCCESS_RATE_TARGET), None)
            if hit is not None:
                reached.append(hit)
        if reached:
            result[env_name] = (float(np.mean(reached)), float(np.std(reached)),
                                len(reached), len(env_cbs))
        else:
            result[env_name] = (float('nan'), float('nan'), 0, len(env_cbs))
    return result


# -------------------- Plot / save --------------------
def plot_success_curves(aggregated: Dict[str, Dict[str, np.ndarray]], save_path: str):
    plt.figure(figsize=(10, 6))
    label_map = {'Standard': 'Standard reward', 'Custom': 'Shaped reward (1 - E)',
                 'Effort': 'Effort-penalized (r_std - beta*u^2)'}
    for env_name, stats in aggregated.items():
        ts = stats['timesteps']
        mean = stats['mean_success']
        std = stats['std_success']
        plt.plot(ts, mean, label=label_map.get(env_name, env_name), linewidth=2)
        plt.fill_between(ts, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
    plt.axhline(SUCCESS_RATE_TARGET, color='gray', linestyle='--', linewidth=1,
                label=f'{int(SUCCESS_RATE_TARGET*100)}% target')
    plt.xlabel('Timesteps')
    plt.ylabel('Success rate')
    plt.title(f'InvertedPendulum PPO: Success rate over {NUM_SEEDS} seeds '
              f'(success = episode > {SUCCESS_THRESHOLD} steps)')
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Success-rate plot saved to: {save_path}")


def save_results(callbacks: Dict[str, List[SuccessRateCallback]],
                 aggregated: Dict[str, Dict[str, np.ndarray]],
                 seeds: List[int]):
    rows = []
    for env_name, env_cbs in callbacks.items():
        for seed_value, cb in zip(seeds, env_cbs):
            for t, r in zip(cb.timesteps, cb.success_rates):
                rows.append({'environment': env_name, 'seed': seed_value,
                             'timesteps': t, 'success_rate': r})
    per_seed_path = os.path.join(RUN_DIR, 'per_seed_success_rates.csv')
    pd.DataFrame(rows).to_csv(per_seed_path, index=False)
    print(f"Per-seed success rates saved to: {per_seed_path}")

    agg_rows = []
    for env_name, stats in aggregated.items():
        for t, m, s in zip(stats['timesteps'], stats['mean_success'], stats['std_success']):
            agg_rows.append({'environment': env_name, 'timesteps': t,
                             'mean_success_rate': m, 'std_success_rate': s})
    agg_path = os.path.join(RUN_DIR, 'aggregated_success_rates.csv')
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    print(f"Aggregated success rates saved to: {agg_path}")


# -------------------- Final evaluation --------------------
def final_evaluation(trained_models: Dict[str, List[PPO]]) -> Dict[str, Tuple[float, float]]:
    """Evaluate each seed's final policy; return mean & std of per-seed success rate."""
    eval_env = gym.make('InvertedPendulum-v4')
    results: Dict[str, Tuple[float, float]] = {}
    for env_name, models in trained_models.items():
        per_seed_rates = [
            evaluate_success_rate(m, eval_env, EVAL_EPISODES, EVAL_RESET_SEEDS)
            for m in models
        ]
        results[env_name] = (float(np.mean(per_seed_rates)), float(np.std(per_seed_rates)))
    eval_env.close()
    return results


def print_summary(final_eval: Dict[str, Tuple[float, float]],
                  to_target: Dict[str, Tuple[float, float, int, int]]):
    print(f"\n{'='*64}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*64}")
    print(f"  Training timesteps/seed : {TOTAL_TIMESTEPS:,}")
    print(f"  Seeds                   : {SEEDS}")
    print(f"  Eval interval           : {EVAL_INTERVAL:,} steps, {EVAL_EPISODES} episodes")
    print(f"  Success threshold       : episode length > {SUCCESS_THRESHOLD}")

    label_map = {'Standard': 'Standard reward', 'Custom': 'Shaped reward (1 - E)',
                 'Effort': 'Effort-penalized (r_std - beta*u^2)'}

    print(f"\nFinal policy success rate (over {NUM_SEEDS} seeds x {EVAL_EPISODES} episodes):")
    for env_name, (mean, std) in final_eval.items():
        print(f"  - {label_map.get(env_name, env_name):24s}: {mean*100:6.2f}% +/- {std*100:5.2f}%")

    print(f"\nTimesteps to first reach >{int(SUCCESS_RATE_TARGET*100)}% success rate:")
    for env_name, (mean, std, n_reached, n_total) in to_target.items():
        if n_reached == 0:
            print(f"  - {label_map.get(env_name, env_name):24s}: never reached "
                  f"(0/{n_total} seeds)")
        else:
            print(f"  - {label_map.get(env_name, env_name):24s}: {mean:10,.0f} +/- {std:8,.0f} "
                  f"timesteps  ({n_reached}/{n_total} seeds)")


# -------------------- Main --------------------
def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    callbacks: Dict[str, List[SuccessRateCallback]] = {'Standard': [], 'Custom': [], 'Effort': []}
    trained_models: Dict[str, List[PPO]] = {'Standard': [], 'Custom': [], 'Effort': []}

    for seed in SEEDS:
        print(f"\n{'='*64}\nRunning seed {seed}\n{'='*64}")
        set_random_seed(seed)
        for env_name in ('Standard', 'Custom', 'Effort'):
            model, cb = train_one(env_name, seed)
            callbacks[env_name].append(cb)
            trained_models[env_name].append(model)
            seed_dir = os.path.join(RUN_DIR, f'seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            model.save(os.path.join(seed_dir, f'ppo_{env_name.lower()}_inverted_pendulum'))

    aggregated = aggregate_success_curves(callbacks)
    plot_success_curves(aggregated, os.path.join(RUN_DIR, 'success_rate_vs_timesteps.png'))
    save_results(callbacks, aggregated, SEEDS)

    final_eval = final_evaluation(trained_models)
    to_target = timesteps_to_target(callbacks)
    print_summary(final_eval, to_target)

    print(f"\nDone. Results in: {RUN_DIR}")


def _apply_quick_config():
    """Tiny config for a smoke test."""
    global NUM_SEEDS, SEEDS, TOTAL_TIMESTEPS, EVAL_INTERVAL, EVAL_EPISODES
    global SUCCESS_THRESHOLD, EVAL_RESET_SEEDS, RUN_DIR
    NUM_SEEDS = 1
    SEEDS = [BASE_SEED]
    TOTAL_TIMESTEPS = 4_000
    EVAL_INTERVAL = 2_000
    EVAL_EPISODES = 5
    SUCCESS_THRESHOLD = 50          # low bar so the smoke test can show non-zero rates
    EVAL_RESET_SEEDS = [BASE_SEED * 1000 + i for i in range(EVAL_EPISODES)]
    RUN_DIR = "runs/inverted_pendulum_ppo_paper_quicktest"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Tiny smoke-test run (1 seed, 4k steps).')
    args = parser.parse_args()
    if args.quick:
        _apply_quick_config()
    main()
