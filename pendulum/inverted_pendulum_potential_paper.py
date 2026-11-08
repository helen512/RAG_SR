#!/usr/bin/env python3
"""
InvertedPendulum PPO: Potential-based reward shaping (PBRS) variants.
================================================================================

Companion to inverted_pendulum_ppo_paper.py. Trains PPO on the Gymnasium
MuJoCo InvertedPendulum-v4 environment with two potential-based shaped rewards:

    F(s, a, s') = gamma * Phi(s') - Phi(s)
    r' = r_standard + F        (r_standard = +1 per surviving timestep)

  1. PBRS_Energy: Phi(s) = -ENERGY_COEF * E(s), where E is the total mechanical
     energy of the cart-pole system (potential energy zero at upright).
  2. PBRS_Effort: Phi(s) = -EFFORT_COEF * ||a_prev||^2, where a_prev is the
     action that led into state s. The control effort is not a pure function of
     the environment state, so the last action is treated as part of an
     augmented state; this is an approximate PBRS.

All settings, PPO hyperparameters, evaluation protocol, and output formats
match inverted_pendulum_ppo_paper.py, so paper_plot.py can combine the run
directories of both scripts. The Standard baseline is NOT retrained here;
paper_plot.py picks it up from the paper script's run directory.

A task is considered a SUCCESS if the episode lasts longer than 950 timesteps.
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

# -------------------- Configuration (matches inverted_pendulum_ppo_paper.py) ----
BASE_SEED = 42
NUM_SEEDS = 5
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]
TOTAL_TIMESTEPS = 150_000          # training timesteps per environment per seed
EVAL_INTERVAL = 5_000              # evaluate the policy every this many train steps
EVAL_EPISODES = 100                # episodes per evaluation
SUCCESS_THRESHOLD = 950            # episode length strictly greater => success
SUCCESS_RATE_TARGET = 0.90         # report timesteps to first reach this success rate
MAX_EPISODE_LENGTH = 1000          # InvertedPendulum-v4 truncates here

# Fixed reset seeds for evaluation so the success-rate curve is comparable over time.
EVAL_RESET_SEEDS = [BASE_SEED * 1000 + i for i in range(EVAL_EPISODES)]

RUN_DIR = "runs/inverted_pendulum_ppo_potential_energy_150_000"

# Physical constants for the energy term (match cartpole_ppo2.py / InvertedPendulum-v4).
G = 9.81
CART_MASS = 10.472   # M
POLE_MASS = 5.019    # m
POLE_LEN = 0.3       # l (half-pole length used in cartpole_ppo2.py)

ENERGY_COEF = 0.1    # alpha in Phi(s) = -alpha * E(s)
EFFORT_COEF = 0.1    # beta in Phi(s) = -beta * ||a_prev||^2

# Discount used in the shaping term; must equal PPO's gamma for the PBRS
# optimal-policy invariance to hold.
SHAPING_GAMMA = 0.99

ENV_NAMES = ('PBRS_Energy',)


# -------------------- Energy --------------------
def total_energy(observation: np.ndarray) -> float:
    """Total mechanical energy E of the cart-pole (potential zero at upright)."""
    _x, x_dot, theta, theta_dot = observation
    return 0.5 * (
        (CART_MASS + POLE_MASS) * x_dot**2
        + POLE_MASS * x_dot * POLE_LEN * theta_dot
        + (1.0 / 3.0) * POLE_MASS * POLE_LEN**2 * theta_dot**2
        + POLE_MASS * POLE_LEN * G * (1.0 - np.cos(theta))
    )


# -------------------- PBRS wrappers --------------------
class EnergyPBRSInvertedPendulum(gym.Wrapper):
    """Standard reward plus PBRS with energy potential.

    Phi(s) = -ENERGY_COEF * E(s)
    r' = r + SHAPING_GAMMA * Phi(s') - Phi(s)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._potential = 0.0

    @staticmethod
    def _phi(observation: np.ndarray) -> float:
        return -ENERGY_COEF * total_energy(observation)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._potential = self._phi(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        next_potential = self._phi(observation)
        shaping = SHAPING_GAMMA * next_potential - self._potential
        self._potential = next_potential
        return observation, reward + shaping, terminated, truncated, info


class EffortPBRSInvertedPendulum(gym.Wrapper):
    """Standard reward plus PBRS with a control-effort potential.

    The control effort depends on the action, not the state, so the last
    action is treated as part of an augmented state:

    Phi(s) = -EFFORT_COEF * ||a_prev||^2   (a_prev = action leading into s)
    r' = r + SHAPING_GAMMA * Phi(s') - Phi(s)

    At reset there is no previous action, so Phi = 0.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._potential = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._potential = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        next_potential = -EFFORT_COEF * float(np.sum(np.square(action)))
        shaping = SHAPING_GAMMA * next_potential - self._potential
        self._potential = next_potential
        return observation, reward + shaping, terminated, truncated, info


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
    if env_name == 'PBRS_Energy':
        env = EnergyPBRSInvertedPendulum(env)
    elif env_name == 'PBRS_Effort':
        env = EffortPBRSInvertedPendulum(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def evaluate_success_rate(model: PPO, eval_env: gym.Env, eval_episodes: int,
                          eval_seeds: List[int]) -> float:
    """Fraction of episodes whose length is strictly greater than SUCCESS_THRESHOLD.

    Termination dynamics are identical for all reward designs, so a plain
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
LABEL_MAP = {
    'PBRS_Energy': 'PBRS energy (r + gamma*Phi(s\') - Phi(s), Phi = -alpha*E)',
    'PBRS_Effort': 'PBRS effort (Phi = -beta*||a||^2)',
}


def plot_success_curves(aggregated: Dict[str, Dict[str, np.ndarray]], save_path: str):
    plt.figure(figsize=(10, 6))
    for env_name, stats in aggregated.items():
        ts = stats['timesteps']
        mean = stats['mean_success']
        std = stats['std_success']
        plt.plot(ts, mean, label=LABEL_MAP.get(env_name, env_name), linewidth=2)
        plt.fill_between(ts, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
    plt.axhline(SUCCESS_RATE_TARGET, color='gray', linestyle='--', linewidth=1,
                label=f'{int(SUCCESS_RATE_TARGET*100)}% target')
    plt.xlabel('Timesteps')
    plt.ylabel('Success rate')
    plt.title(f'InvertedPendulum PPO (PBRS): Success rate over {NUM_SEEDS} seeds '
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
    print("EXPERIMENT SUMMARY (PBRS)")
    print(f"{'='*64}")
    print(f"  Training timesteps/seed : {TOTAL_TIMESTEPS:,}")
    print(f"  Seeds                   : {SEEDS}")
    print(f"  Eval interval           : {EVAL_INTERVAL:,} steps, {EVAL_EPISODES} episodes")
    print(f"  Success threshold       : episode length > {SUCCESS_THRESHOLD}")

    print(f"\nFinal policy success rate (over {NUM_SEEDS} seeds x {EVAL_EPISODES} episodes):")
    for env_name, (mean, std) in final_eval.items():
        print(f"  - {LABEL_MAP.get(env_name, env_name):48s}: {mean*100:6.2f}% +/- {std*100:5.2f}%")

    print(f"\nTimesteps to first reach >{int(SUCCESS_RATE_TARGET*100)}% success rate:")
    for env_name, (mean, std, n_reached, n_total) in to_target.items():
        if n_reached == 0:
            print(f"  - {LABEL_MAP.get(env_name, env_name):48s}: never reached "
                  f"(0/{n_total} seeds)")
        else:
            print(f"  - {LABEL_MAP.get(env_name, env_name):48s}: {mean:10,.0f} +/- {std:8,.0f} "
                  f"timesteps  ({n_reached}/{n_total} seeds)")


# -------------------- Main --------------------
def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    callbacks: Dict[str, List[SuccessRateCallback]] = {name: [] for name in ENV_NAMES}
    trained_models: Dict[str, List[PPO]] = {name: [] for name in ENV_NAMES}

    for seed in SEEDS:
        print(f"\n{'='*64}\nRunning seed {seed}\n{'='*64}")
        set_random_seed(seed)
        for env_name in ENV_NAMES:
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
    RUN_DIR = "runs/inverted_pendulum_potential_paper_quicktest"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Tiny smoke-test run (1 seed, 4k steps).')
    args = parser.parse_args()
    if args.quick:
        _apply_quick_config()
    main()
