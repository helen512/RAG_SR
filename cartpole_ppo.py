#!/usr/bin/env python3
"""
CartPole PPO Training: Standard vs Custom Reward Functions
==========================================================

This script creates and trains two CartPole environments using PPO from Stable Baselines3:
1. Standard environment with original CartPole reward function
2. Custom environment with a reward function that takes action and state as inputs

Both environments use the same optimal PPO parameters from SB3 zoo.
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
from stable_baselines3.common.evaluation import evaluate_policy

# Configuration
SEED = 42
TOTAL_TIMESTEPS = 100_000  # Training timesteps for each environment
RUN_DIR = "runs_cartpole_ppo"
os.makedirs(RUN_DIR, exist_ok=True)
set_random_seed(SEED)


class TrainingCallback(BaseCallback):
    """Callback to track episodic returns during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []
        self.timesteps_log = []
    
    def _on_step(self) -> bool:
        # self.locals contains info of the current step
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            # The 'episode' key is automatically added by the Monitor wrapper when an episode ends
            if 'episode' in info:
                # Log episode statistics
                self.episode_returns.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                self.timesteps_log.append(self.num_timesteps)
        # return False to stop training
        return True


class CustomRewardCartPole(gym.Wrapper):
    """
    Custom CartPole environment with reward function that takes action and state as inputs.
    The custom reward function encourages:
    1. Keeping the pole upright (small angle)
    2. Keeping the cart centered (small position)
    3. Minimizing velocities for stability
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def custom_reward_function(self, observation: np.ndarray, action: int) -> float:
        """
        Custom reward function that takes both action and state as inputs.
        
        Args:
            observation: State vector [x, x_dot, theta, theta_dot]
            action: Action taken (0 or 1)
        
        Returns:
            Custom reward value
        """
        x, x_dot, theta, theta_dot = observation
        
        # Normalize state variables
        x_threshold = self.env.unwrapped.x_threshold  # ±2.4
        theta_threshold = self.env.unwrapped.theta_threshold_radians  # ±12°
        
        x_norm = x / x_threshold
        theta_norm = theta / theta_threshold
        
        # Custom reward components
        # 1. Penalty for pole angle (primary objective)
        angle_penalty = theta_norm ** 2
        
        # 2. Penalty for cart position (keep centered)
        position_penalty = x_norm ** 2
        
        # 3. Penalty for velocities (encourage stability)
        velocity_penalty = 0.1 * (np.tanh(x_dot / 2.0) ** 2 + np.tanh(theta_dot / 8.0) ** 2)
        
        # Combine penalties into reward (higher is better)
        reward = 1.0 - (angle_penalty + 0.1 * position_penalty + velocity_penalty)
        
        return reward
    
    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Replace original reward with custom reward
        custom_reward = self.custom_reward_function(observation, action)
        
        return observation, custom_reward, terminated, truncated, info


def get_optimal_ppo_params() -> Dict[str, Any]:
    """
    Get optimal PPO hyperparameters for CartPole-v1 from Stable Baselines3 Zoo.
    These parameters are tuned for optimal performance on CartPole.
    """
    return {
        'learning_rate': 0.0003,
        'n_steps': 128,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None,
        'verbose': 1,
        'seed': SEED,
        'device': 'auto'
    }


def create_environments() -> Tuple[gym.Env, gym.Env]:
    """Create and return both standard and custom reward environments."""
    # Standard environment
    env_standard = gym.make('CartPole-v1')
    env_standard = Monitor(env_standard)
    env_standard.reset(seed=SEED)
    
    # Custom reward environment
    env_custom = gym.make('CartPole-v1')
    env_custom = CustomRewardCartPole(env_custom)
    env_custom = Monitor(env_custom)
    env_custom.reset(seed=SEED)
    
    return env_standard, env_custom


def train_ppo_model(env: gym.Env, model_name: str) -> Tuple[PPO, TrainingCallback]:
    """Train a PPO model on the given environment."""
    print(f"Training PPO on {model_name} environment")
    ppo_params = get_optimal_ppo_params()
    # Create PPO model
    model = PPO(
        policy='MlpPolicy',
        env=env,
        **ppo_params
    )
    # Create callback for tracking training
    callback = TrainingCallback()
    # Train the model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )
    print(f"Training completed for {model_name}")
    return model, callback


def evaluate_model(model: PPO, env: gym.Env, model_name: str, n_eval_episodes: int = 100):
    """Evaluate a trained model."""
    print(f"\nEvaluating {model_name} model...")
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_eval_episodes,
        # Controls whether the policy uses exploration (stochastic) or exploitation (deterministic)
        deterministic=True,
        # Controls whether to render the visual environment during evaluation
        render=False
    )
    
    print(f"{model_name} - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward


def plot_training_curves(callbacks: Dict[str, TrainingCallback], save_path: str):
    """Plot training curves for both models."""
    
    plt.figure(figsize=(12, 5))
    # Plot 1: Episode Returns over Time
    plt.subplot(1, 2, 1)
    for name, callback in callbacks.items():
        if len(callback.episode_returns) > 0:
            plt.plot(callback.timesteps_log, callback.episode_returns, 
                    label=f'{name}', alpha=0.7)
            
            # Add smoothed line
            if len(callback.episode_returns) > 10:
                window = min(50, len(callback.episode_returns) // 4)
                smoothed = pd.Series(callback.episode_returns).rolling(window).mean()
                plt.plot(callback.timesteps_log, smoothed, 
                        label=f'{name} (smoothed)', linewidth=2)
    
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Return')
    plt.title('Training Progress: Episode Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths over Time
    plt.subplot(1, 2, 2)
    for name, callback in callbacks.items():
        if len(callback.episode_lengths) > 0:
            plt.plot(callback.timesteps_log, callback.episode_lengths, 
                    label=f'{name}', alpha=0.7)
            
            # Add smoothed line
            if len(callback.episode_lengths) > 10:
                window = min(50, len(callback.episode_lengths) // 4)
                smoothed = pd.Series(callback.episode_lengths).rolling(window).mean()
                plt.plot(callback.timesteps_log, smoothed, 
                        label=f'{name} (smoothed)', linewidth=2)
    
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Length')
    plt.title('Training Progress: Episode Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to: {save_path}")


def save_results(callbacks: Dict[str, TrainingCallback], eval_results: Dict[str, Tuple[float, float]]):
    """Save training and evaluation results to CSV."""
    
    # Save training data
    training_data = []
    for name, callback in callbacks.items():
        for i in range(len(callback.episode_returns)):
            training_data.append({
                'environment': name,
                'timestep': callback.timesteps_log[i],
                'episode_return': callback.episode_returns[i],
                'episode_length': callback.episode_lengths[i]
            })
    
    training_df = pd.DataFrame(training_data)
    training_path = os.path.join(RUN_DIR, 'training_results.csv')
    training_df.to_csv(training_path, index=False)
    print(f"Training results saved to: {training_path}")
    
    # Save evaluation results
    eval_data = []
    for name, (mean_reward, std_reward) in eval_results.items():
        eval_data.append({
            'environment': name,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })
    
    eval_df = pd.DataFrame(eval_data)
    eval_path = os.path.join(RUN_DIR, 'evaluation_results.csv')
    eval_df.to_csv(eval_path, index=False)
    print(f"Evaluation results saved to: {eval_path}")
    
    return training_df, eval_df


def print_summary(eval_results: Dict[str, Tuple[float, float]], ppo_params: Dict[str, Any]):
    """Print a summary of the experiment."""
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    print(f"Configuration:")
    print(f"  - Total training timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - Random seed: {SEED}")
    print(f"  - Results directory: {RUN_DIR}")
    
    print(f"\nPPO Hyperparameters:")
    for key, value in ppo_params.items():
        if key not in ['verbose', 'device']:
            print(f"  - {key}: {value}")
    
    print(f"\nEvaluation Results (100 episodes):")
    for name, (mean_reward, std_reward) in eval_results.items():
        print(f"  - {name:15s}: {mean_reward:6.2f} ± {std_reward:5.2f}")
    
    # Calculate improvement
    if len(eval_results) == 2:
        results = list(eval_results.values())
        improvement = results[1][0] - results[0][0]  # Custom - Standard
        print(f"\nCustom vs Standard Improvement: {improvement:+.2f} reward units")


def main():
    """Main execution function."""

    # Create environments
    print("Creating environments...")
    env_standard, env_custom = create_environments()
    # Store callbacks and models
    callbacks = {}
    models = {}
    
    # Train standard model
    models['Standard'], callbacks['Standard'] = train_ppo_model(env_standard, 'Standard')
    
    # Train custom reward model
    models['Custom'], callbacks['Custom'] = train_ppo_model(env_custom, 'Custom')
    
    # Evaluate both models
    print(f"\n{'='*50}")
    print("EVALUATION")
    print(f"{'='*50}")
    
    eval_results = {}
    eval_results['Standard'] = evaluate_model(models['Standard'], env_standard, 'Standard')
    eval_results['Custom'] = evaluate_model(models['Custom'], env_custom, 'Custom')
    
    # Plot results
    plot_path = os.path.join(RUN_DIR, 'training_curves.png')
    plot_training_curves(callbacks, plot_path)
    
    # Save results
    training_df, eval_df = save_results(callbacks, eval_results)
    
    # Save models
    models['Standard'].save(os.path.join(RUN_DIR, 'ppo_standard_cartpole'))
    models['Custom'].save(os.path.join(RUN_DIR, 'ppo_custom_cartpole'))
    print("Models saved to:", RUN_DIR)
    
    # Print summary
    ppo_params = get_optimal_ppo_params()
    print_summary(eval_results, ppo_params)
    
    # Close environments
    env_standard.close()
    env_custom.close()
    print(f"\nExperiment completed successfully!")
    print(f"All results saved in: {RUN_DIR}")


if __name__ == "__main__":
    main()
