#!/usr/bin/env python3
"""
InvertedDoublePendulum PPO Training: Standard vs Custom Reward Functions
=======================================================================

This script creates and trains two InvertedDoublePendulum environments using PPO from Stable Baselines3:
1. Standard environment with original InvertedDoublePendulum reward function
2. Custom environment with a reward function that takes action and state as inputs

Both environments use optimized PPO parameters for continuous control tasks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Tuple, Dict, Any
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

# Configuration
SEED = 42
TOTAL_TIMESTEPS = 10_000  # Training timesteps for each environment
STEPS_PER_EPOCH = 400
run_index = 1
RUN_DIR = f"runs_double_pendulum_ppo"
os.makedirs(RUN_DIR, exist_ok=True)
set_random_seed(SEED)


class EpisodicLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.returns = []
        self.timesteps = []
        self.episode_lengths = []
        self.epoch_returns = []  # Mean return per epoch
        self.epoch_lengths = []  # Mean episode length per epoch
        self.epoch_timesteps = []  # Timesteps per epoch
        self.epochs_log = []  # Epoch numbers for plotting
        self.current_epoch_returns = []  # Collect returns during current epoch
        self.current_epoch_lengths = []  # Collect episode lengths during current epoch
        self.episode_returns = []  # All episode returns for plotting
        self._ep_ret = 0.0
        self.total_episodes = 0
        self.current_epoch = 0

    def _on_step(self) -> bool:
        if "episode" in self.locals.get("infos", [{}])[-1]:
            ep_info = self.locals["infos"][-1]["episode"]
            ep_return = ep_info["r"]
            ep_length = ep_info["l"]
            
            self.returns.append(ep_return)
            self.episode_returns.append(ep_return)
            self.episode_lengths.append(ep_length)
            self.timesteps.append(self.num_timesteps)
            self.current_epoch_returns.append(ep_return)
            self.current_epoch_lengths.append(ep_length)
            self.total_episodes += 1
            
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each PPO rollout (epoch)"""
        if len(self.current_epoch_returns) > 0:
            mean_return = np.mean(self.current_epoch_returns)
            mean_length = np.mean(self.current_epoch_lengths)
            self.epoch_returns.append(mean_return)
            self.epoch_lengths.append(mean_length)
            self.epoch_timesteps.append(self.num_timesteps)
            self.epochs_log.append(self.current_epoch)
            self.current_epoch_returns = []  # Reset for next epoch
            self.current_epoch_lengths = []  # Reset for next epoch
            self.current_epoch += 1


class CustomRewardDoublePendulum(gym.Wrapper):
    """
    Custom InvertedDoublePendulum environment with reward function that takes action and state as inputs.
    The custom reward function encourages:
    1. Keeping both poles upright (small angles)
    2. Keeping the cart centered (small position)
    3. Minimizing velocities for stability
    4. Penalizing large control actions
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def custom_reward_function(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        Custom reward function that takes both action and state as inputs.
        
        Args:
            observation: State vector [x, sin(theta1), cos(theta1), sin(theta2), cos(theta2), 
                                     x_dot, theta1_dot, theta2_dot, ...]
            action: Continuous action (force applied to cart)
        
        Returns:
            Custom reward value
        """
        # InvertedDoublePendulum-v5 observation space:
        # [0]: x position of cart
        # [1]: sin(theta1) - angle of first pole
        # [2]: cos(theta1) - angle of first pole  
        # [3]: sin(theta2) - angle of second pole
        # [4]: cos(theta2) - angle of second pole
        # [5]: x_dot - cart velocity
        # [6]: theta1_dot - angular velocity of first pole
        # [7]: theta2_dot - angular velocity of second pole
        # [8-10]: additional state variables
        
        x = observation[0]
        sin_theta1, cos_theta1 = observation[1], observation[2]
        sin_theta2, cos_theta2 = observation[3], observation[4]
        x_dot = observation[5]
        theta1_dot = observation[6] if len(observation) > 6 else 0.0
        theta2_dot = observation[7] if len(observation) > 7 else 0.0
        
        # Calculate actual angles from sin/cos
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        theta2 = np.arctan2(sin_theta2, cos_theta2)
        
        # Normalize state variables (typical ranges for InvertedDoublePendulum)
        x_threshold = 2.4  # Cart position limit
        x_norm = np.clip(x / x_threshold, -1.0, 1.0)
        
        # Angular deviations from upright (0 radians)
        theta1_norm = theta1 / np.pi  # Normalize to [-1, 1]
        theta2_norm = theta2 / np.pi  # Normalize to [-1, 1]
        
        # Velocity normalization
        x_dot_norm = np.tanh(x_dot / 5.0)  # Saturate large velocities
        theta1_dot_norm = np.tanh(theta1_dot / 10.0)  # Saturate large angular velocities
        theta2_dot_norm = np.tanh(theta2_dot / 10.0)
        
        # Action normalization (assuming action is in reasonable range)
        action_norm = np.clip(action[0] / 10.0, -1.0, 1.0) if len(action) > 0 else 0.0
        
        # Custom reward components
        # 1. Primary objective: Keep both poles upright
        angle_penalty1 = theta1_norm ** 2
        angle_penalty2 = theta2_norm ** 2
        
        # 2. Keep cart centered
        position_penalty = x_norm ** 2
        
        # 3. Minimize velocities for stability
        velocity_penalty = 0.1 * (x_dot_norm ** 2 + theta1_dot_norm ** 2 + theta2_dot_norm ** 2)
        
        # 4. Penalize large control actions
        action_penalty = 0.01 * action_norm ** 2
        
        # 5. Bonus for keeping both poles close to upright
        upright_bonus = (1.0 - angle_penalty1) * (1.0 - angle_penalty2)
        
        # Combine into reward (higher is better)
        reward = (
            1.0  # Base survival reward
            + 2.0 * upright_bonus  # Strong bonus for both poles upright
            - 0.5 * (angle_penalty1 + angle_penalty2)  # Angle penalties
            - 0.2 * position_penalty  # Position penalty
            - velocity_penalty  # Velocity penalty
            - action_penalty  # Action penalty
        )
        
        return reward
    
    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Replace original reward with custom reward
        custom_reward = self.custom_reward_function(observation, action)
        
        return observation, custom_reward, terminated, truncated, info


def get_optimal_ppo_params() -> Dict[str, Any]:
    """
    Get optimal PPO hyperparameters for InvertedDoublePendulum continuous control.
    These parameters are tuned for continuous control tasks.
    """
    return {
        'seed': SEED, 
        'verbose': 0,
        'learning_rate': 3e-4,
        'n_steps': 1024,  # Smaller rollout for continuous control
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,  # Small exploration bonus for continuous control
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(
            net_arch=[64, 64],  # Network architecture
            activation_fn=nn.Tanh,  # Use tanh activation for continuous control
        ),
    }


def create_environments() -> Tuple[gym.Env, gym.Env]:
    """Create and return both standard and custom reward environments."""
    # Standard environment
    env_standard = gym.make('InvertedDoublePendulum-v5')
    env_standard = Monitor(env_standard)
    env_standard.reset(seed=SEED)
    
    # Custom reward environment
    env_custom = gym.make('InvertedDoublePendulum-v5')
    env_custom = CustomRewardDoublePendulum(env_custom)
    env_custom = Monitor(env_custom)
    env_custom.reset(seed=SEED)
    
    return env_standard, env_custom


def train_ppo_model(env: gym.Env, model_name: str) -> Tuple[PPO, EpisodicLogger]:
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
    callback = EpisodicLogger()
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
        deterministic=True,
        render=False
    )
    
    print(f"{model_name} - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward


def plot_training_curves(callbacks: Dict[str, EpisodicLogger], save_path: str):
    """Plot training curves for both models - single plot showing epoch vs mean episode length."""
    
    # Single plot: Mean episode length per epoch (PPO rollout) - Fair comparison metric
    plt.figure(figsize=(8, 5))
    for name, callback in callbacks.items():
        if len(callback.epoch_lengths) > 0:
            epochs = np.arange(1, len(callback.epoch_lengths) + 1)
            plt.plot(epochs, callback.epoch_lengths, 
                    marker='o' if name == 'Standard' else 's', 
                    label=name, alpha=0.7, linewidth=2)
    
    plt.xlabel('PPO Epoch (Rollout)')
    plt.ylabel('Mean Episode Length per Epoch')
    plt.title('InvertedDoublePendulum: Standard vs Custom Reward - Episode Length Comparison (PPO)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to: {save_path}")


def save_results(callbacks: Dict[str, EpisodicLogger], eval_results: Dict[str, Tuple[float, float]]):
    """Save training and evaluation results to CSV."""
    
    # Save training data with both returns and episode lengths
    training_data = []
    for name, callback in callbacks.items():
        for i in range(len(callback.epoch_returns)):
            training_data.append({
                'environment': name,
                'epoch': i,
                'timesteps': callback.epoch_timesteps[i],
                'mean_return': callback.epoch_returns[i],
                'mean_episode_length': callback.epoch_lengths[i]
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
    print(f"  - Environment: InvertedDoublePendulum-v5")
    print(f"  - Total training timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - Random seed: {SEED}")
    print(f"  - Results directory: {RUN_DIR}")
    
    print(f"\nPPO Hyperparameters:")
    for key, value in ppo_params.items():
        if key not in ['verbose', 'device', 'policy_kwargs']:
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
    print("Creating InvertedDoublePendulum environments...")
    env_standard, env_custom = create_environments()
    
    # Print environment info
    print(f"Observation space: {env_standard.observation_space}")
    print(f"Action space: {env_standard.action_space}")
    
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
    models['Standard'].save(os.path.join(RUN_DIR, 'ppo_standard_double_pendulum'))
    models['Custom'].save(os.path.join(RUN_DIR, 'ppo_custom_double_pendulum'))
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
