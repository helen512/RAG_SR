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
LOG_DIR = "reacher_custom2potential_final"
VIDEO_DIR = "reacher_custom2videospotential_final"
TOTAL_TIMESTEPS = 700_000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

class ConstraintWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self.prev_potential = 0.0
        self.episode_original_reward = 0.0
        self.episode_kinetic_energy = 0.0
        self.episode_euclidean_distance = 0.0
        self.episode_distances = []
        print(f"Custom Reward Wrapper initialized:")
        print(f"  Training reward = -EuclideanDistance + Potential Based Shaping (Potential = -KineticEnergy)")
        
    def reset(self, **kwargs):
        self.episode_original_reward = 0.0
        self.episode_kinetic_energy = 0.0
        self.episode_euclidean_distance = 0.0
        self.episode_distances = []
        
        obs, info = self.env.reset(**kwargs)
        
        # Calculate initial potential
        # We reuse costom_reward logic to get KE
        _, kinetic_energy, _ = self.costom_reward(obs)
        self.prev_potential = -kinetic_energy
        
        return obs, info

    def costom_reward(self, obs):

        I1 = 3.056e-5
        I2 = 4.422e-5
        ang_vel_0 = obs[6]
        ang_vel_1 = obs[7]
        m1 = 3.56e-2
        m2 = 3.979e-2
        l1 = l2 = 0.1

        kinetic_energy = 1/2 * (1/3 * m1 * l1**2 * ang_vel_0**2 + m2 * l2**2 * ang_vel_0**2 + m2 * l2*l1 * ang_vel_0 * ang_vel_1* np.cos(ang_vel_1) + 1/3 * m2 * l2**2 * ang_vel_1**2)

        target_x, target_y = obs[4], obs[5]
        tip_x, tip_y = obs[8], obs[9]

        # Calculate euclidean distance between target and tip
        euclidean_distance = np.sqrt((target_x - tip_x) ** 2 + (target_y - tip_y) ** 2)
        
        # Custom reward: negative kinetic energy and negative distance (both are penalties)
        # We keep this method for calculating metrics, but the return value 'custom_reward' 
        # might not be used directly as the main reward if we are using original + shaping.
        custom_reward = -kinetic_energy - euclidean_distance
        
        return custom_reward, kinetic_energy, euclidean_distance


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_original_reward += reward
        
        # Calculate metrics and potential
        _, kinetic_energy, euclidean_distance = self.costom_reward(obs)
        self.episode_kinetic_energy += kinetic_energy
        self.episode_euclidean_distance += euclidean_distance
        self.episode_distances.append(euclidean_distance)

        # Potential Based Reward Shaping
        # Potential Phi(s) = -KineticEnergy(s)
        # Shaping F = gamma * Phi(s') - Phi(s)
        current_potential = -kinetic_energy
        shaping = self.gamma * current_potential - self.prev_potential
        self.prev_potential = current_potential

        # Reward = -EuclideanDistance + Shaping
        reward = -euclidean_distance + shaping
        
        # Check theta2 constraint
        # theta2 is the second joint angle.
        # In MuJoCo Reacher, qpos is [theta1, theta2, target_x, target_y]
        # We access the unwrapped environment to get the physical state
        try:
            theta2 = self.unwrapped.data.qpos[1]
        except AttributeError:
            # Fallback for non-mujoco environments or testing
            theta2 = 0.0
        
        # Constraint: -2.55 < theta2 < 2.55
        if theta2 < -2.55 or theta2 > 2.55:
            reward += -200
            terminated = True
            info["violation"] = True
        else:
            info["violation"] = False
            
        if terminated or truncated:
            info["original_reward"] = self.episode_original_reward
            info["episode_kinetic_energy"] = self.episode_kinetic_energy
            info["episode_euclidean_distance"] = self.episode_euclidean_distance

            # Sum of last 10 distances
            last_10 = self.episode_distances[-10:] if self.episode_distances else [0.0]
            info["last_10_dist_sum"] = sum(last_10)

            # Ensure violation is present in info even if it wasn't the trigger in this exact step 
            if "violation" not in info:
                info["violation"] = False
            
        return obs, reward, terminated, truncated, info

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

def make_env():
    # Create the environment
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    
    # Wrap with constraint wrapper
    env = ConstraintWrapper(env)
    
    # Wrap with Monitor to track rewards/timesteps
    # Add info_keywords to log original reward, violation status, and accumulated metrics
    env = Monitor(env, LOG_DIR, info_keywords=("original_reward", "violation", "episode_kinetic_energy", "episode_euclidean_distance", "last_10_dist_sum"))
    
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
    Plot the results using pandas loading
    """
    try:
        # Find monitor file
        monitor_files = [f for f in os.listdir(log_folder) if f.endswith("monitor.csv")]
        if not monitor_files:
            print("No monitor file found.")
            return
            
        # Assuming single env or picking the first one
        # Here we pick the most recent one based on modification time.
        latest_file = max([os.path.join(log_folder, f) for f in monitor_files], key=os.path.getmtime)
        print(f"Plotting results from: {latest_file}")
            
        results_df = pd.read_csv(latest_file, comment='#')
        
        # 'r' is modified reward
        # 'original_reward' is standard reward
        
        # Calculate rolling window mean for smoother plot
        window = 50
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Helper to plot with rolling mean
        def plot_metric(ax, x, y, label, color):
            ax.plot(x, y, alpha=0.3, label=label, color=color)
            if len(y) > window:
                rolling_mean = y.rolling(window=window).mean()
                ax.plot(x, rolling_mean, color=color, linewidth=2, label=f'{label} Mean ({window} eps)')
            else:
                 ax.plot(x, y, color=color, linewidth=2, label=f'{label} Mean')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Value')
            ax.set_title(f'{label} vs Timesteps')
            ax.legend()
            ax.grid(True)

        # 1. Custom Reward
        plot_metric(axes[0, 0], results_df['l'].cumsum(), results_df['r'], 'Custom Reward', 'blue')
        
        # 2. Original Reward
        if 'original_reward' in results_df.columns:
            plot_metric(axes[0, 1], results_df['l'].cumsum(), results_df['original_reward'], 'Original Reward', 'green')
            
        # 3. Accumulated Kinetic Energy
        if 'episode_kinetic_energy' in results_df.columns:
            plot_metric(axes[1, 0], results_df['l'].cumsum(), results_df['episode_kinetic_energy'], 'Accumulated Kinetic Energy', 'red')
            
        # 4. Accumulated Euclidean Distance
        if 'episode_euclidean_distance' in results_df.columns:
            plot_metric(axes[1, 1], results_df['l'].cumsum(), results_df['episode_euclidean_distance'], 'Accumulated Euclidean Distance', 'purple')

        plt.suptitle(title)
        plt.tight_layout()
        
        save_path = os.path.join(log_folder, "reward_vs_timesteps_modified.png")
        plt.savefig(save_path)
        print(f"Reward plot saved to {save_path}")
        plt.close()
        
        # Track violations
        total_episodes = len(results_df)
        if 'violation' in results_df.columns:
            # violation column might be boolean or string 'True'/'False'
            # Convert to boolean if necessary
            if results_df['violation'].dtype == object:
                 violation_count = (results_df['violation'] == 'True').sum()
            else:
                 violation_count = results_df['violation'].sum()
                 
            print(f"Total Episodes: {total_episodes}")
            print(f"Episodes Terminated due to Violation: {violation_count}")
            print(f"Violation Rate: {violation_count/total_episodes:.2%}")
        else:
            print("No violation tracking data found.")
        
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting PPO training on Reacher-v5 with Constraints...")
    
    # 1. Create the environment
    # We wrap it in DummyVecEnv because VecNormalize requires a VecEnv
    env = DummyVecEnv([make_env])
    
    # 2. Add Normalization (Critical for MuJoCo)
    # NormObs=True, NormReward=True
    # Note: NormReward will normalize the *modified* reward.
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
    model_save_path = os.path.join(LOG_DIR, "ppo_reacher_potentialbased_final")
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
