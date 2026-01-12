import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch.nn as nn
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Configuration
LOG_DIR = "ppo_reacher_baselinefinal_logs"
VIDEO_DIR = "ppo_reacher_baselinefinal_videos"
TOTAL_TIMESTEPS = 700_000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value
    return func

class DistanceTrackingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_distances = []
        
    def reset(self, **kwargs):
        self.episode_distances = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate Euclidean distance
        # Reacher-v5 obs: [cos(t1), cos(t2), sin(t1), sin(t2), target_x, target_y, av1, av2, tip_x, tip_y, ...]
        if len(obs) >= 10:
            target = obs[4:6]
            tip = obs[8:10]
            dist = np.linalg.norm(target - tip)
            self.episode_distances.append(dist)
            
        if terminated or truncated:
            last_10 = self.episode_distances[-10:] if self.episode_distances else [0.0]
            info["last_10_dist_sum"] = sum(last_10)
            
        return obs, reward, terminated, truncated, info

def make_env():
    # Create the environment
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    
    # Wrap to track distances
    env = DistanceTrackingWrapper(env)
    
    # Wrap with Monitor to track rewards/timesteps
    env = Monitor(env, LOG_DIR, info_keywords=("last_10_dist_sum",))
    
    # Wrap to record videos
    # Record every 200 episodes
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda episode_id: episode_id % 200 == 0,
        name_prefix="ppo-reacher"
    )
    return env

def plot_results(log_folder, title='Reacher-v5 PPO Training Reward'):
    """
    Plot the results using stable_baselines3 helpers or manual pandas loading
    """
    try:
        # Load results from the Monitor CSV file
        results_df = load_results(log_folder)
        
        # Calculate rolling window mean for smoother plot
        window = 50
        if len(results_df) > window:
            rolling_mean = results_df['r'].rolling(window=window).mean()
        else:
            rolling_mean = results_df['r']

        plt.figure(figsize=(10, 6))
        plt.plot(results_df['l'].cumsum(), results_df['r'], alpha=0.3, label='Episode Reward')
        plt.plot(results_df['l'].cumsum(), rolling_mean, color='r', label=f'Rolling Mean ({window} eps)')
        
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(log_folder, "reward_vs_timesteps.png")
        plt.savefig(save_path)
        print(f"Reward plot saved to {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting results: {e}")

def main():
    print("Starting PPO training on Reacher-v5 with Optimized Hyperparameters...")
    
    # 1. Create the environment
    # We wrap it in DummyVecEnv because VecNormalize requires a VecEnv
    env = DummyVecEnv([make_env])
    
    # 2. Add Normalization (Critical for MuJoCo)
    # NormObs=True, NormReward=True
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Initialize PPO model with Optimized Hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        # learning_rate=linear_schedule(2.5e-4),
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        # clip_range=linear_schedule(0.2),
        clip_range=0.2,
        ent_coef=0.0,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto" 
    )
    
    # Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save the final model and the normalization statistics
    model_save_path = os.path.join(LOG_DIR, "ppo_reacher_final")
    model.save(model_save_path)
    env.save(os.path.join(LOG_DIR, "vec_normalize.pkl"))
    print(f"Model saved to {model_save_path}")
    
    # Close environment
    env.close()
    
    # Plot results
    plot_results(LOG_DIR)
    print("Training and plotting completed.")

if __name__ == '__main__':
    main()
