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
from stable_baselines3.common.evaluation import evaluate_policy

# Configuration
SEED = 42
TOTAL_TIMESTEPS = 50_000  # Training timesteps for each environment
STEPS_PER_EPOCH = 4000
run_index = 1
RUN_DIR = f"runs_cartpole2_ppo_{run_index}"
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
            
            # Check if episode failed due to x position exceeding threshold
            # Get the current observation to check the cart position
            obs = self.locals.get("new_obs", None)
            if obs is not None:
                # For CartPole, obs[0] is the cart position x
                x_pos = obs[0] if hasattr(obs, '__len__') and len(obs) > 0 else obs
                if hasattr(x_pos, '__len__'):
                    x_pos = x_pos[0]
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
        
        # Normalize state variables (InvertedPendulum has different scales)
        # Use approximate thresholds for normalization
        x_threshold = 2.0  # Approximate threshold for cart position
        theta_threshold = np.pi  # Full rotation
        
        x_norm = x / x_threshold
        theta_norm = theta / theta_threshold
        
        # Normalize action (action is typically in range [-3, 3])
        action_magnitude = np.abs(action[0]) / 3.0
        
        # Custom reward components
        # 1. Penalty for pole angle (primary objective) - want pole upright
        angle_penalty = theta_norm ** 2
        
        # 2. Penalty for cart position (keep centered)
        position_penalty = x_norm ** 2
        
        # 3. Penalty for velocities (encourage stability)
        velocity_penalty = 0.1 * (np.tanh(x_dot / 2.0) ** 2 + np.tanh(theta_dot / 8.0) ** 2)
        
        # 4. Penalty for large actions (energy efficiency)
        action_penalty = 0.05 * action_magnitude ** 2
        
        # Combine penalties into reward (higher is better)
        reward = 1.0 - 0.5*(theta_norm ** 2 + theta_dot ** 2 + x_dot ** 2 + x_dot*theta_dot) - action_penalty
        
        return reward
    
    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Replace original reward with custom reward
        custom_reward = self.custom_reward_function(observation, action)
        
        return observation, custom_reward, terminated, truncated, info


def get_optimal_ppo_params() -> Dict[str, Any]:
    """
    Get optimal PPO hyperparameters for InvertedPendulum-v4 (continuous action space).
    These parameters are tuned for optimal performance on MuJoCo environments.
    """
    return {
        'seed': SEED, 
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


def create_environments() -> Tuple[gym.Env, gym.Env]:
    """Create and return both standard and custom reward environments."""
    # Standard environment
    env_standard = gym.make('InvertedPendulum-v4')
    env_standard = Monitor(env_standard)
    env_standard.reset(seed=SEED)
    
    # Custom reward environment
    env_custom = gym.make('InvertedPendulum-v4')
    env_custom = CustomRewardCartPole(env_custom)
    env_custom = Monitor(env_custom)
    env_custom.reset(seed=SEED)
    
    return env_standard, env_custom


def train_ppo_model(env: gym.Env, model_name: str) -> Tuple[PPO, EpisodicLogger]:
    """Train a PPO model on the given environment."""
    print(f"Training PPO on {model_name} environment")
    ppo_params = get_optimal_ppo_params()
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
    plt.title('InvertedPendulum (MuJoCo): Standard vs Custom Reward - Episode Length Comparison (PPO)')
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
    models['Standard'].save(os.path.join(RUN_DIR, 'ppo_standard_inverted_pendulum'))
    models['Custom'].save(os.path.join(RUN_DIR, 'ppo_custom_inverted_pendulum'))
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
