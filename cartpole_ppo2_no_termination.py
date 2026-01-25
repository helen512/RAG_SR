#!/usr/bin/env python3
"""
CartPole PPO Training: Standard vs Custom Reward Functions (Continuous Action Space)
=====================================================================================

This script creates and trains two InvertedPendulum (MuJoCo CartPole) environments using PPO from Stable Baselines3:
1. Standard environment with modified reward: +1 when angle <= 0.2 rad, else 0. Termination removed.
2. Custom environment with energy-based reward. Termination removed. Records the standard metric (+1 when angle <= 0.2) for comparison.

Both environments use continuous action space and optimal PPO parameters.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Tuple, Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm

# Configuration
BASE_SEED = 42
NUM_SEEDS = 1
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]
TOTAL_TIMESTEPS = 250_000  # Training timesteps for each environment
TIMESTEP_INTERVAL = 2_000
EVAL_EPISODES = 60
EVAL_SEEDS = [BASE_SEED * 100 + i for i in range(EVAL_EPISODES)]
MAX_EPISODE_LENGTH = 1000  # Maximum episode length for InvertedPendulum-v4

RUN_DIR = "inverted_pendulum_ppo_no_termination_PBRS2"
os.makedirs(RUN_DIR, exist_ok=True)


class EpisodicLogger(BaseCallback):
    """
    Callback for logging the 'Standard Metric' (count of steps with angle <= 0.2).
    """
    def __init__(self):
        super().__init__()
        self.metric_returns = []  # Stores the accumulated metric (region count) per episode
        self.timesteps = []
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        # Check if an episode just finished
        # Monitor wrapper adds 'episode' dict to info when done/truncated
        if "episode" in self.locals.get("infos", [{}])[-1]:
            info = self.locals["infos"][-1]
            
            # We look for 'region_metric' which our wrapper injects
            if "region_metric" in info:
                self.metric_returns.append(info["region_metric"])
                self.timesteps.append(self.num_timesteps)
                self.total_episodes += 1
                
        return True


class ModifiedCartPoleWrapper(gym.Wrapper):
    """
    Modified InvertedPendulum environment:
    - Removes termination (except truncation).
    - Tracks a 'region_metric': +1 per step if |theta| <= 0.2.
    - If use_custom_reward=False (Standard): Reward is the region_metric (+1 if valid, 0 else).
    - If use_custom_reward=True (Custom): Uses Potential-Based Reward Shaping (PBRS).
      Base reward (+1 if valid, 0 else) + F(s, s').
      F(s, s') = gamma * Phi(s') - Phi(s), where Phi(s) = 0.01 * (-Energy(s)).
    """
    
    def __init__(self, env, use_custom_reward=False, gamma=0.99):
        super().__init__(env)
        self.use_custom_reward = use_custom_reward
        self.gamma = gamma
        self.ep_region_count = 0.0
        self.prev_potential = 0.0
        
    def reset(self, **kwargs):
        self.ep_region_count = 0.0
        # reset returns (obs, info) in Gymnasium
        obs, info = self.env.reset(**kwargs)
        
        if self.use_custom_reward:
            self.prev_potential = self._calculate_potential(obs)
            
        return obs, info

    def _calculate_potential(self, observation: np.ndarray) -> float:
        # Unpack observation based on InvertedPendulum-v4 standard:
        # 0: position (x)
        # 1: angle (theta)
        # 2: velocity (x_dot)
        # 3: angular velocity (theta_dot)
        x, theta, x_dot, theta_dot = observation
        
        g = 9.81
        mc, mp, l = 10.472, 5.019, 0.3  # half-pole length
        total_mass = mc + mp
        mp_l = mp * l

        # Energy-based potential function (Phi)
        # We use scaled negative total energy
        # Previous custom reward was 0.01 * (-Energy)
        # So we define Phi(s) = 0.01 * (-Energy)
        
        energy = 1/2 * (total_mass * x_dot**2 + mp * x_dot * l * theta_dot + 1/3 * mp * l**2 * theta_dot**2 + mp_l * g * (1 - np.cos(theta)))
        potential = -energy
        
        return potential
    
    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Remove Termination (allow pendulum to spin/fall without resetting until timeout)
        terminated = False
        
        # 2. Track Standard Metric (+1 when angle <= 0.2 rad)
        theta = observation[1] # Index 1 is angle
        in_region = abs(theta) <= 0.2
        if in_region:
            self.ep_region_count += 1.0
            
        # 3. Determine Training Reward
        base_reward = 1.0 if in_region else 0.0
        
        if self.use_custom_reward:
            current_potential = self._calculate_potential(observation)
            # Potential-Based Reward Shaping: F = gamma * Phi(s') - Phi(s)
            shaping = self.gamma * current_potential - self.prev_potential
            reward = base_reward + shaping
            self.prev_potential = current_potential
        else:
            reward = base_reward
            
        # Pass the metric to logger via info dict
        info['region_metric'] = self.ep_region_count
        
        return observation, reward, terminated, truncated, info


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


def create_environments(seed: int) -> Tuple[gym.Env, gym.Env]:
    """Create and return both standard and custom reward environments."""
    # Standard environment (Modified reward: 0/1 based on region)
    env_standard = gym.make('InvertedPendulum-v4')
    env_standard = ModifiedCartPoleWrapper(env_standard, use_custom_reward=False)
    env_standard = Monitor(env_standard)
    env_standard.reset(seed=seed)
    
    # Custom environment (Energy reward)
    env_custom = gym.make('InvertedPendulum-v4')
    env_custom = ModifiedCartPoleWrapper(env_custom, use_custom_reward=True)
    env_custom = Monitor(env_custom)
    env_custom.reset(seed=seed)
    
    return env_standard, env_custom


def train_ppo_model(env: gym.Env, model_name: str, seed: int) -> Tuple[PPO, EpisodicLogger]:
    """Train a PPO model on the given environment."""
    print(f"Training PPO on {model_name} environment")
    ppo_params = get_optimal_ppo_params(seed)
    model = PPO(
        policy='MlpPolicy',
        env=env,
        **ppo_params
    )
    callback = EpisodicLogger()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )
    print(f"Training completed for {model_name}")
    return model, callback


def _interpolate_metrics(callback: EpisodicLogger, grid: np.ndarray) -> np.ndarray:
    if len(callback.timesteps) == 0 or len(callback.metric_returns) == 0:
        return np.zeros_like(grid) # Handle empty case gracefully

    timesteps = np.asarray(callback.timesteps, dtype=np.float64)
    metrics = np.asarray(callback.metric_returns, dtype=np.float64)

    order = np.argsort(timesteps)
    timesteps = timesteps[order]
    metrics = metrics[order]

    unique_timesteps, unique_indices = np.unique(timesteps, return_index=True)
    unique_metrics = metrics[unique_indices]

    return np.interp(
        grid,
        unique_timesteps,
        unique_metrics,
        left=unique_metrics[0] if len(unique_metrics) > 0 else 0,
        right=unique_metrics[-1] if len(unique_metrics) > 0 else 0,
    )


def aggregate_metrics_by_timestep(callbacks: Dict[str, list[EpisodicLogger]]) -> Dict[str, Dict[str, np.ndarray]]:
    grid = np.arange(0, TOTAL_TIMESTEPS + TIMESTEP_INTERVAL, TIMESTEP_INTERVAL, dtype=np.float64)
    aggregated: Dict[str, Dict[str, np.ndarray]] = {}

    for env_name, env_callbacks in callbacks.items():
        if len(env_callbacks) == 0:
            continue
        interpolated_curves = []
        for cb in env_callbacks:
            interpolated_curves.append(_interpolate_metrics(cb, grid))

        if not interpolated_curves:
            continue
            
        curves = np.stack(interpolated_curves, axis=0)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)

        aggregated[env_name] = {
            'timesteps': grid,
            'mean_metric': mean_curve,
            'std_metric': std_curve,
        }

    return aggregated


def plot_average_metrics(aggregated_data: Dict[str, Dict[str, np.ndarray]], save_path: str):
    plt.figure(figsize=(10, 6))

    for env_name, stats in aggregated_data.items():
        timesteps = stats['timesteps']
        mean = stats['mean_metric']
        std = stats['std_metric']

        plt.plot(timesteps, mean, label=env_name, linewidth=2)
        plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2)

    plt.xlabel('Timesteps')
    plt.ylabel('Accumulated Metric (Steps with |θ| <= 0.2)')
    plt.title(f'Comparison: Standard (0/1) vs Custom (Energy) Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")


def save_results(
    callbacks: Dict[str, list[EpisodicLogger]],
    aggregated_data: Dict[str, Dict[str, np.ndarray]],
    seeds: list[int],
):
    per_seed_rows = []
    for env_name, env_callbacks in callbacks.items():
        for seed_value, callback in zip(seeds, env_callbacks):
            per_seed_rows.extend(
                {
                    'environment': env_name,
                    'seed': seed_value,
                    'timesteps': t,
                    'metric_return': r,
                }
                for t, r in zip(callback.timesteps, callback.metric_returns)
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(RUN_DIR, 'per_seed_metrics.csv')
    per_seed_df.to_csv(per_seed_path, index=False)
    print(f"Per-seed metrics saved to: {per_seed_path}")

    aggregated_rows = []
    for env_name, stats in aggregated_data.items():
        for timestep, mean, std in zip(stats['timesteps'], stats['mean_metric'], stats['std_metric']):
            aggregated_rows.append(
                {
                    'environment': env_name,
                    'timesteps': timestep,
                    'mean_metric': mean,
                    'std_metric': std,
                }
            )

    aggregated_df = pd.DataFrame(aggregated_rows)
    aggregated_path = os.path.join(RUN_DIR, 'aggregated_metrics.csv')
    aggregated_df.to_csv(aggregated_path, index=False)
    print(f"Aggregated metrics saved to: {aggregated_path}")


def main():
    """Main execution function."""

    callbacks_per_env: Dict[str, list[EpisodicLogger]] = {'Standard': [], 'Custom': []}
    
    # We will save models here
    trained_models = {'Standard': [], 'Custom': []}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")

        set_random_seed(seed)
        env_standard, env_custom = create_environments(seed)

        # Train Standard
        model_standard, callback_standard = train_ppo_model(env_standard, 'Standard', seed)
        callbacks_per_env['Standard'].append(callback_standard)
        trained_models['Standard'].append(model_standard)
        
        # Train Custom
        model_custom, callback_custom = train_ppo_model(env_custom, 'Custom', seed)
        callbacks_per_env['Custom'].append(callback_custom)
        trained_models['Custom'].append(model_custom)

        # Save models
        seed_dir = os.path.join(RUN_DIR, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        model_standard.save(os.path.join(seed_dir, 'ppo_standard_inverted_pendulum'))
        model_custom.save(os.path.join(seed_dir, 'ppo_custom_inverted_pendulum'))

        env_standard.close()
        env_custom.close()

    aggregated_data = aggregate_metrics_by_timestep(callbacks_per_env)

    plot_path = os.path.join(RUN_DIR, 'metric_vs_timesteps.png')
    plot_average_metrics(aggregated_data, plot_path)

    save_results(callbacks_per_env, aggregated_data, SEEDS)

    print(f"\nExperiment completed successfully!")
    print(f"All results saved in: {RUN_DIR}")


if __name__ == "__main__":
    main()
