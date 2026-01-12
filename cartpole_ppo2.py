#!/usr/bin/env python3
"""
CartPole PPO Training: Standard vs Custom Reward Functions (Continuous Action Space)
=====================================================================================

This script creates and trains two InvertedPendulum (MuJoCo CartPole) environments using PPO from Stable Baselines3:
1. Standard environment with original InvertedPendulum reward function
2. Custom environment with a reward function that takes action and state as inputs

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
NUM_SEEDS = 5
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]
TOTAL_TIMESTEPS = 200_000  # Training timesteps for each environment
TIMESTEP_INTERVAL = 2_000
EVAL_EPISODES = 60
EVAL_SEEDS = [BASE_SEED * 100 + i for i in range(EVAL_EPISODES)]
MAX_EPISODE_LENGTH = 1000  # Maximum episode length for InvertedPendulum-v4

RUN_DIR = "runs_cartpole2_ppo_test"
os.makedirs(RUN_DIR, exist_ok=True)


class EpisodicLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.returns = []
        self.timesteps = []
        self.episode_lengths = []
        self.total_episodes = 0
        
        # Tracking when maximum episode length is first reached
        self.timestep_reached_max_length = None  # timestep when max episode length first reached
        self.has_reached_max_length = False  # flag to track if max length was reached

    def _on_step(self) -> bool:
        if "episode" in self.locals.get("infos", [{}])[-1]:
            ep_info = self.locals["infos"][-1]["episode"]
            ep_return = ep_info["r"]
            ep_length = ep_info["l"]
            
            self.returns.append(ep_return)
            self.episode_lengths.append(ep_length)
            self.timesteps.append(self.num_timesteps)
            self.total_episodes += 1
            
            # Check if this episode reached maximum length (and we haven't recorded it yet)
            if not self.has_reached_max_length and ep_length >= MAX_EPISODE_LENGTH:
                self.timestep_reached_max_length = self.num_timesteps
                self.has_reached_max_length = True
            
            # Check if episode failed due to x position exceeding threshold
            # Get the current observation to check the cart position
            obs = self.locals.get("new_obs", None)
            if obs is not None:
                # For CartPole, obs[0] is the cart position x
                x_pos = obs[0] if hasattr(obs, '__len__') and len(obs) > 0 else obs
                if hasattr(x_pos, '__len__'):
                    x_pos = x_pos[0]
        return True
 
        


class CustomRewardCartPole(gym.Wrapper):
    """
    Custom InvertedPendulum environment with reward function that takes action and state as inputs.
    The custom reward function encourages:
    1. Keeping the pole upright (small angle)
    2. Keeping the cart centered (small position)
    3. Minimizing velocities for stability
    4. Minimizing action magnitude
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def custom_reward_function(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        Custom reward function that takes both action and state as inputs.
        
        Args:
            observation: State vector [x (cart position), x_dot (cart velocity), theta (pole angle), theta_dot (pole angular velocity)]
            action: Action taken (continuous force value)
        
        Returns:
            Custom reward value
        """
        x, x_dot, theta, theta_dot = observation
        
        
        g = 9.81
        gamma = 0.1
        mc, mp, l = 10.472, 5.019, 0.3  # half-pole length
        total_mass = mc + mp
        mp_l = mp * l
        gear = 100.0

        reward = 1-gamma * 1/2 * (total_mass * x_dot**2 + mp * x_dot * l * theta_dot + 1/3 * mp * l**2 * theta_dot**2 + mp_l * g * (1 - np.cos(theta)))
        
        return reward
    
    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Replace original reward with custom reward
        custom_reward = self.custom_reward_function(observation, action)
        
        return observation, custom_reward, terminated, truncated, info


def get_optimal_ppo_params(seed: int) -> Dict[str, Any]:
    """
    Get optimal PPO hyperparameters for InvertedPendulum-v4 (continuous action space).
    These parameters are tuned for optimal performance on MuJoCo environments.
    """
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
        'ent_coef': 0.0,  # Will be set automatically for continuous actions
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(net_arch=[64, 64]),
    }


def create_environments(seed: int) -> Tuple[gym.Env, gym.Env]:
    """Create and return both standard and custom reward environments."""
    # Standard environment
    env_standard = gym.make('InvertedPendulum-v4')
    env_standard = Monitor(env_standard)
    env_standard.reset(seed=seed)
    
    # Custom reward environment
    env_custom = gym.make('InvertedPendulum-v4')
    env_custom = CustomRewardCartPole(env_custom)
    env_custom = Monitor(env_custom)
    env_custom.reset(seed=seed)
    
    return env_standard, env_custom


def train_ppo_model(env: gym.Env, model_name: str, seed: int) -> Tuple[PPO, EpisodicLogger]:
    """Train a PPO model on the given environment."""
    print(f"Training PPO on {model_name} environment")
    ppo_params = get_optimal_ppo_params(seed)
    # Create PPO model - MlpPolicy automatically handles continuous/discrete actions
    model = PPO(
        policy='MlpPolicy',
        env=env,
        **ppo_params
    )
    # Create callback for tracking training (pass n_steps for epoch calculation)
    callback = EpisodicLogger()
    # Train the model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )
    print(f"Training completed for {model_name}")
    return model, callback


def _interpolate_episode_lengths(callback: EpisodicLogger, grid: np.ndarray) -> np.ndarray:
    if len(callback.timesteps) == 0 or len(callback.episode_lengths) == 0:
        raise ValueError("Timesteps or episode lengths are empty")
        

    timesteps = np.asarray(callback.timesteps, dtype=np.float64)
    lengths = np.asarray(callback.episode_lengths, dtype=np.float64)

    order = np.argsort(timesteps)
    timesteps = timesteps[order]
    lengths = lengths[order]

    unique_timesteps, unique_indices = np.unique(timesteps, return_index=True)
    unique_lengths = lengths[unique_indices]

    return np.interp(
        grid,
        unique_timesteps,
        unique_lengths,
        left=unique_lengths[0],
        right=unique_lengths[-1],
    )


def aggregate_returns_by_timestep(callbacks: Dict[str, list[EpisodicLogger]]) -> Dict[str, Dict[str, np.ndarray]]:
    grid = np.arange(0, TOTAL_TIMESTEPS + TIMESTEP_INTERVAL, TIMESTEP_INTERVAL, dtype=np.float64)
    aggregated: Dict[str, Dict[str, np.ndarray]] = {}

    for env_name, env_callbacks in callbacks.items():
        if len(env_callbacks) == 0:
            continue
        interpolated_curves = []
        for cb in env_callbacks:
            interpolated_curves.append(_interpolate_episode_lengths(cb, grid))

        curves = np.stack(interpolated_curves, axis=0)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)

        aggregated[env_name] = {
            'timesteps': grid,
            'mean_length': mean_curve,
            'std_length': std_curve,
        }

    return aggregated


def _make_evaluation_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make('InvertedPendulum-v4')
    if env_name == 'Custom':
        env = CustomRewardCartPole(env)
    env.reset(seed=seed)
    return env


def _evaluate_single_model(model: PPO, env_name: str, progress_bar: tqdm | None = None) -> list[int]:
    episode_lengths: list[int] = []
    for seed in EVAL_SEEDS:
        env = _make_evaluation_env(env_name, seed)
        obs, info = env.reset(seed=seed)
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        episode_lengths.append(steps)
        env.close()
        if progress_bar is not None:
            progress_bar.update(1)
    return episode_lengths


def evaluate_models(trained_models: Dict[str, list[PPO]]) -> Dict[str, list[int]]:
    evaluation_results: Dict[str, list[int]] = {env_name: [] for env_name in trained_models}
    total_evals = sum(len(models) for models in trained_models.values()) * len(EVAL_SEEDS)
    with tqdm(total=total_evals, desc="Evaluating policies", unit="episode") as progress_bar:
        for env_name, models in trained_models.items():
            for model in models:
                evaluation_results[env_name].extend(
                    _evaluate_single_model(model, env_name, progress_bar)
                )
    return evaluation_results


def plot_average_returns(aggregated_data: Dict[str, Dict[str, np.ndarray]], save_path: str):
    plt.figure(figsize=(10, 6))

    for env_name, stats in aggregated_data.items():
        timesteps = stats['timesteps']
        mean = stats['mean_length']
        std = stats['std_length']

        plt.plot(timesteps, mean, label=env_name, linewidth=2)
        plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2)

    plt.xlabel('Timesteps')
    plt.ylabel('Average Episode Length')
    plt.title(f'InvertedPendulum PPO: Average Episode Length over {NUM_SEEDS} Seeds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()

    print(f"Average episode length plot saved to: {save_path}")


def save_results(
    callbacks: Dict[str, list[EpisodicLogger]],
    aggregated_data: Dict[str, Dict[str, np.ndarray]],
    seeds: list[int],
):
    """Save per-seed episode returns and aggregated statistics to CSV."""

    per_seed_rows = []
    for env_name, env_callbacks in callbacks.items():
        for seed_value, callback in zip(seeds, env_callbacks):
            per_seed_rows.extend(
                {
                    'environment': env_name,
                    'seed': seed_value,
                    'timesteps': t,
                    'episode_return': r,
                    'episode_length': l,
                }
                for t, r, l in zip(callback.timesteps, callback.returns, callback.episode_lengths)
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(RUN_DIR, 'per_seed_episode_returns.csv')
    per_seed_df.to_csv(per_seed_path, index=False)
    print(f"Per-seed episode returns saved to: {per_seed_path}")

    aggregated_rows = []
    for env_name, stats in aggregated_data.items():
        for timestep, mean, std in zip(stats['timesteps'], stats['mean_length'], stats['std_length']):
            aggregated_rows.append(
                {
                    'environment': env_name,
                    'timesteps': timestep,
                    'mean_episode_length': mean,
                    'std_episode_length': std,
                }
            )

    aggregated_df = pd.DataFrame(aggregated_rows)
    aggregated_path = os.path.join(RUN_DIR, 'aggregated_episode_lengths.csv')
    aggregated_df.to_csv(aggregated_path, index=False)
    print(f"Aggregated episode lengths saved to: {aggregated_path}")

    return per_seed_df, aggregated_df


def print_summary(aggregated_data: Dict[str, Dict[str, np.ndarray]], evaluation_results: Dict[str, list[float]]):
    """Print a summary of the experiment."""

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    print("Configuration:")
    print(f"  - Total training timesteps per seed: {TOTAL_TIMESTEPS:,}")
    print(f"  - Number of seeds: {NUM_SEEDS}")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Results directory: {RUN_DIR}")

    print("\nFinal averaged returns:")
    for env_name, stats in aggregated_data.items():
        final_mean = stats['mean_length'][-1]
        final_std = stats['std_length'][-1]
        print(f"  - {env_name:15s}: {final_mean:8.2f} ± {final_std:6.2f} (episode length)")

    if len(aggregated_data) == 2:
        env_names = list(aggregated_data.keys())
        improvement = aggregated_data[env_names[1]]['mean_length'][-1] - aggregated_data[env_names[0]]['mean_length'][-1]
        print(f"\nRelative episode-length difference ({env_names[1]} - {env_names[0]}): {improvement:+.2f}")

    print(f"\nEvaluation (episode lengths over {NUM_SEEDS} trained policies × {EVAL_EPISODES} evaluation seeds):")
    for env_name, lengths in evaluation_results.items():
        if len(lengths) == 0:
            continue
        mean_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))
        total_episodes = len(lengths)
        print(f"  - {env_name:15s}: mean length={mean_length:8.2f}, std={std_length:6.2f} over {total_episodes} episodes")


def main():
    """Main execution function."""

    callbacks_per_env: Dict[str, list[EpisodicLogger]] = {'Standard': [], 'Custom': []}
    trained_models: Dict[str, list[PPO]] = {'Standard': [], 'Custom': []}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")

        set_random_seed(seed)
        env_standard, env_custom = create_environments(seed)

        models = {}
        models['Standard'], callback_standard = train_ppo_model(env_standard, 'Standard', seed)
        models['Custom'], callback_custom = train_ppo_model(env_custom, 'Custom', seed)

        callbacks_per_env['Standard'].append(callback_standard)
        callbacks_per_env['Custom'].append(callback_custom)
        trained_models['Standard'].append(models['Standard'])
        trained_models['Custom'].append(models['Custom'])

        seed_dir = os.path.join(RUN_DIR, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        models['Standard'].save(os.path.join(seed_dir, 'ppo_standard_inverted_pendulum'))
        models['Custom'].save(os.path.join(seed_dir, 'ppo_custom_inverted_pendulum'))

        env_standard.close()
        env_custom.close()

    aggregated_data = aggregate_returns_by_timestep(callbacks_per_env)

    plot_path = os.path.join(RUN_DIR, 'average_episode_length_vs_timesteps.png')
    plot_average_returns(aggregated_data, plot_path)

    save_results(callbacks_per_env, aggregated_data, SEEDS)

    evaluation_results = evaluate_models(trained_models)
    

    print_summary(aggregated_data, evaluation_results)
    
    # Print summary for timesteps to reach maximum episode length
    print("\n" + "=" * 60)
    print("TIMESTEPS TO REACH MAXIMUM EPISODE LENGTH")
    print("=" * 60)
    print(f"Maximum episode length: {MAX_EPISODE_LENGTH} steps")
    
    for env_name, env_callbacks in callbacks_per_env.items():
        if len(env_callbacks) == 0:
            continue
        
        # Extract timesteps when max episode length was first reached for each seed
        timesteps_to_max = []
        for callback in env_callbacks:
            if callback.timestep_reached_max_length is not None:
                timesteps_to_max.append(callback.timestep_reached_max_length)
        
        if len(timesteps_to_max) == 0:
            print(f"\n{env_name:15s}: Never reached maximum episode length")
        else:
            mean_timesteps = np.mean(timesteps_to_max)
            std_timesteps = np.std(timesteps_to_max)
            print(f"\n{env_name:15s}: {mean_timesteps:10.0f} ± {std_timesteps:10.0f} timesteps")
            print(f"  ({len(timesteps_to_max)}/{len(env_callbacks)} seeds reached max length)")

    print(f"\nExperiment completed successfully!")
    print(f"All results saved in: {RUN_DIR}")


if __name__ == "__main__":
    main()
