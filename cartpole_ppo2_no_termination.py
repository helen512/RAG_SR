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
from typing import Tuple, Dict, Any, List

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm

# Configuration
BASE_SEED = 42
NUM_SEEDS = 3
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]
TOTAL_TIMESTEPS = 250_000  # Training timesteps for each environment
TIMESTEP_INTERVAL = 2_000
EVAL_EPISODES = 60
EVAL_SEEDS = [BASE_SEED * 100 + i for i in range(EVAL_EPISODES)]
MAX_EPISODE_LENGTH = 1000  # Maximum episode length for InvertedPendulum-v4

RUN_DIR = "inverted_pendulum_ppo_no_termination_PBRS_0.1energy_multiple_seeds"
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


class EnergyLoggerCallback(BaseCallback):
    """
    Callback for logging the energy and shaping term at each step.
    """
    def __init__(self, run_dir: str, env_name: str, seed: int):
        super().__init__()
        self.energies = {'total': [], 'kinetic': [], 'potential': [], 'shaping': []}
        self.save_path = os.path.join(run_dir, f"energy_log_{env_name}_seed_{seed}.csv")
        self.plot_path = os.path.join(run_dir, f"energy_plot_{env_name}_seed_{seed}.png")
        self.env_name = env_name

    def _on_step(self) -> bool:
        # Access the info dict from the environments
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if 'total_energy' in info:
                self.energies['total'].append(info['total_energy'])
                self.energies['kinetic'].append(info.get('kinetic_energy', 0.0))
                self.energies['potential'].append(info.get('potential_energy', 0.0))
                self.energies['shaping'].append(info.get('shaping_reward', 0.0))
        return True

    def _on_training_end(self):
        # Save to CSV
        df = pd.DataFrame({
            'total_energy': self.energies['total'],
            'kinetic_energy': self.energies['kinetic'],
            'potential_energy': self.energies['potential'],
            'shaping_reward': self.energies['shaping']
        })
        df.to_csv(self.save_path, index=False)
        
        # Calculate and print max energy
        if self.energies['total']:
            max_total = np.max(self.energies['total'])
            max_kinetic = np.max(self.energies['kinetic'])
            max_potential = np.max(self.energies['potential'])
            max_shaping = np.max(self.energies['shaping'])
            
            print(f"[{self.env_name}] Max total energy: {max_total:.4f}")
            print(f"[{self.env_name}] Max kinetic energy: {max_kinetic:.4f}")
            print(f"[{self.env_name}] Max potential energy: {max_potential:.4f}")
            print(f"[{self.env_name}] Max shaping reward: {max_shaping:.4f}")
            print(f"[{self.env_name}] Energy log saved to: {self.save_path}")

            # Plot energies
            plt.figure(figsize=(12, 6))
            timesteps = np.arange(len(self.energies['total']))
            plt.plot(timesteps, self.energies['total'], label='Total Energy', alpha=0.7)
            plt.plot(timesteps, self.energies['kinetic'], label='Kinetic Energy', alpha=0.7)
            plt.plot(timesteps, self.energies['potential'], label='Potential Energy', alpha=0.7)
            plt.plot(timesteps, self.energies['shaping'], label='Shaping Reward', alpha=0.7)
            plt.xlabel('Timesteps')
            plt.ylabel('Energy / Reward')
            plt.title(f'Energy and Shaping vs Timesteps ({self.env_name})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()
            print(f"[{self.env_name}] Energy plot saved to: {self.plot_path}")


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
        self.current_potential = 0.0
        
    def reset(self, **kwargs):
        self.ep_region_count = 0.0
        # reset returns (obs, info) in Gymnasium
        obs, info = self.env.reset(**kwargs)
        
        self.current_potential, _, _, _ = self._calculate_potential(obs)
            
        return obs, info

    def _calculate_potential(self, observation: np.ndarray) -> Tuple[float, float]:
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

        kinetic_energy = 1/2 * (total_mass * x_dot**2 + mp * x_dot * l * theta_dot + 1/3 * mp * l**2 * theta_dot**2)
        potential_energy = 1/2 * mp_l * g * (1 - np.cos(theta))
        total_energy = kinetic_energy + potential_energy
        
    
        potential = - total_energy
        
        return potential, total_energy, kinetic_energy, potential_energy
    
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
        
        # Always calculate potential and energy for logging
        future_potential, current_energy, current_kinetic, current_potential_energy = self._calculate_potential(observation)

        # if truncated or terminated:
        #     future_potential = 0.0

        shaping = self.gamma * future_potential - self.current_potential

        if self.use_custom_reward:
            # Potential-Based Reward Shaping: F = gamma * Phi(s') - Phi(s)
            reward = base_reward + 0.1* shaping
            self.current_potential = future_potential
        else:
            reward = base_reward
            self.current_potential = future_potential
            
        # Pass the metric to logger via info dict
        info['region_metric'] = self.ep_region_count
        info['total_energy'] = current_energy
        info['kinetic_energy'] = current_kinetic
        info['potential_energy'] = current_potential_energy
        info['potential_phi'] = future_potential
        info['shaping_reward'] = 0.1*shaping
        
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
    
    episodic_callback = EpisodicLogger()
    energy_callback = EnergyLoggerCallback(RUN_DIR, model_name, seed)
    
    callbacks = [episodic_callback, energy_callback]

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True
    )
    print(f"Training completed for {model_name}")
    return model, episodic_callback


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
