#!/usr/bin/env python3
"""
CartPole with Safe RL Package: PPO vs PPO-Lagrangian vs CPO
===========================================================
Uses the official safety-starter-agents package to compare:
1. Standard PPO
2. PPO-Lagrangian with x-displacement constraints
3. CPO (Constrained Policy Optimization) with x-displacement constraints

Tracks constraint violations (x > ±MAX_X) during training and evaluation.
Features optimized hyperparameters and smooth cost function design.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Import TensorFlow to reset graph between runs
import tensorflow as tf

# Import from safety-starter-agents
sys.path.append('/home/dmy/gymtest/safety-starter-agents')
from safe_rl import ppo, ppo_lagrangian, cpo

sys.path.append('/home/dmy/gymtest/safe-control-gym')
from safe_control_gym.utils.registration import make

# Configuration
SEED = 42
TOTAL_TIMESTEPS = 100_000
STEPS_PER_EPOCH = 4000
MAX_X_DISPLACEMENT = 1.5  # Constraint threshold
RUN_DIR = "runs_safe_rl_comparison_cpo_logbarrier_with_timesteps2"
os.makedirs(RUN_DIR, exist_ok=True)
save_index = 1

# Set random seeds
np.random.seed(SEED)

print("=" * 80)
print("Safe RL Package: PPO vs PPO-Lagrangian vs CPO on CartPole")
print("=" * 80)
print(f"Configuration:")
print(f"  - Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  - X-displacement constraint: ±{MAX_X_DISPLACEMENT}")
print(f"  - Random seed: {SEED}")
print(f"  - Results directory: {RUN_DIR}")
print()


# =======================================
# CONSTRAINT TRACKING WRAPPER
# =======================================

def log_barrier_x(x, x_max, mu=1.0):

    z = (x / x_max)**2
    z = min(z, 1 - 1e-12)   
    return -mu * np.log(1 - z)

class ConstraintViolationCounter:
    """Counter for tracking x-displacement constraint violations"""
    
    def __init__(self, x_threshold: float = MAX_X_DISPLACEMENT):
        self.x_threshold = x_threshold
        self.violation_episodes = 0
        self.total_episodes = 0
        self.violation_timesteps = 0
        self.total_timesteps = 0
        self.violation_history = []
        # Track violations per epoch
        self.violations_per_epoch = []
        self.episodes_per_epoch = []
        self.current_epoch_violations = 0
        self.current_epoch_episodes = 0
        
    def check_violation(self, obs) -> bool:
        """Check if current observation violates x-displacement constraint"""
        x_pos = obs[0] if isinstance(obs, np.ndarray) else obs
        return abs(x_pos) > self.x_threshold
    
    def compute_cost(self, obs, info) -> float:
        """Compute smooth cost signal for constrained RL algorithms"""
        x_pos = abs(obs[0] if isinstance(obs, np.ndarray) else obs)
        current_timestep = info['episode_timestep']
        return log_barrier_x(x_pos, self.x_threshold)/(current_timestep/100)
     
    
    def step(self, obs) -> bool:
        """Record a timestep and return if violated"""
        self.total_timesteps += 1
        violated = self.check_violation(obs)
        if violated:
            self.violation_timesteps += 1
        return violated
    
    def episode_ended(self, had_violation: bool):
        """Record episode completion"""
        self.total_episodes += 1
        self.current_epoch_episodes += 1
        if had_violation:
            self.violation_episodes += 1
            self.current_epoch_violations += 1
        self.violation_history.append(had_violation)
    
    def epoch_ended(self):
        """Record epoch completion and reset current epoch counters"""
        self.violations_per_epoch.append(self.current_epoch_violations)
        self.episodes_per_epoch.append(self.current_epoch_episodes)
        self.current_epoch_violations = 0
        self.current_epoch_episodes = 0
    
    def get_violation_rate(self) -> float:
        """Get current episode violation rate"""
        if self.total_episodes == 0:
            return 0.0
        return self.violation_episodes / self.total_episodes
    
    def get_timestep_violation_rate(self) -> float:
        """Get current timestep violation rate"""
        if self.total_timesteps == 0:
            return 0.0
        return self.violation_timesteps / self.total_timesteps
    
    def reset(self):
        """Reset all counters"""
        self.violation_episodes = 0
        self.total_episodes = 0
        self.violation_timesteps = 0
        self.total_timesteps = 0
        self.violation_history = []
        self.violations_per_epoch = []
        self.episodes_per_epoch = []
        self.current_epoch_violations = 0
        self.current_epoch_episodes = 0
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_episodes': self.total_episodes,
            'violation_episodes': self.violation_episodes,
            'episode_violation_rate': self.get_violation_rate(),
            'total_timesteps': self.total_timesteps,
            'violation_timesteps': self.violation_timesteps,
            'timestep_violation_rate': self.get_timestep_violation_rate()
        }


class ConstrainedCartPoleWrapper(gym.Wrapper):
    """
    CartPole wrapper that tracks x-displacement violations and provides
    cost signals for constrained RL algorithms.
    """
    
    def __init__(self, env, counter: ConstraintViolationCounter, steps_per_epoch: int = STEPS_PER_EPOCH):
        super().__init__(env)
        self.counter = counter 
        self.episode_had_violation = False
        self.steps_per_epoch = steps_per_epoch
        self.epoch_timesteps = 0  # Track timesteps within current epoch
        self.episode_timestep = 0  # Track timesteps within current episode
        
    def reset(self, **kwargs):
        # Record previous episode
        if self.counter.total_timesteps > 0:
            self.counter.episode_ended(self.episode_had_violation)
        
        self.episode_had_violation = False
        self.episode_timestep = 0  # Reset episode timestep counter
        return self.env.reset(**kwargs)
        
    def step(self, action):
        # Increment episode timestep
        self.episode_timestep += 1
        
        # Convert action from array to scalar if needed (safe_rl returns arrays)
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.size == 1 else int(action[0])
        
        obs, reward, done, info = self.env.step(action)
        
        # Add episode timestep to info dictionary
        info['episode_timestep'] = self.episode_timestep
        
        # Check for violation
        violated = self.counter.step(obs)
        if violated:
            self.episode_had_violation = True
        
        # Track epoch boundaries
        self.epoch_timesteps += 1
        if self.epoch_timesteps % self.steps_per_epoch == 0:
            # Epoch boundary reached
            self.counter.epoch_ended()
            print(f"Epoch ended at timestep {self.counter.total_timesteps} "
                  f"({self.counter.current_epoch_violations} violations this epoch)")
        
        # Add cost information for PPO-Lagrangian
        cost = self.counter.compute_cost(obs, info)
        info['cost'] = cost
        
        return obs, reward, done, info


# =======================================
# TRAINING FUNCTIONS
# =======================================

def train_ppo(counter: ConstraintViolationCounter):
    """Train standard PPO"""
    print("\n" + "=" * 50)
    print("Training Standard PPO")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = 4000
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    # Create a shared counter that persists across environment resets
    def env_fn():
        env = gym.make('CartPole-v1')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)
    
    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'ppo_{save_index}'),
        'exp_name': f'ppo_cartpole'
    }
    
    start_time = time.time()
    
    # Train PPO with optimized hyperparameters
    # Note: PPO in safe_rl uses different parameter names than other implementations
    ppo(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        seed=SEED,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=0.99,                    # Standard discount factor
        lam=0.95,                      # Improved from 0.97 for consistency with PPO-Lag
        target_kl=0.01,                # Good conservative value
        vf_lr=3e-4,                    # Improved from 1e-3 for better value function learning
        vf_iters=80,                   # Keep same
        logger_kwargs=logger_kwargs,
        save_freq=10
    )
    
    training_time = time.time() - start_time
    
    print(f"\nPPO Training Complete!")
    print(f"  Time: {training_time:.1f} seconds")
    print(f"  Training violation summary:")
    summary = counter.get_summary()
    print(f"    - Total episodes: {summary['total_episodes']}")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}")
    print(f"    - Episode violation rate: {summary['episode_violation_rate']:.3f}")
    print(f"    - Timesteps with violations: {summary['violation_timesteps']}/{summary['total_timesteps']}")
    print(f"    - Timestep violation rate: {summary['timestep_violation_rate']:.4f}")
    
    return summary


def train_ppo_lagrangian(counter: ConstraintViolationCounter):
    """Train PPO-Lagrangian"""
    print("\n" + "=" * 50)
    print("Training PPO-Lagrangian")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = 4000
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    def env_fn():
        env = gym.make('CartPole-v1')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)
    
    # Cost limit: tighter constraint for better safety  
    cost_lim = 2.0  # Reduced from 5.0 for stricter constraint adherence
    
    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'ppo_lagrangian_{save_index}'),
        'exp_name': f'ppo_lagrangian_cartpole'
    }
    
    start_time = time.time()
    
    # Train PPO-Lagrangian with optimized hyperparameters
    ppo_lagrangian(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),  # Keep same for CartPole
        seed=SEED,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=0.99,                    # Standard discount factor
        lam=0.95,                      # Improved from 0.9 for better advantage estimation
        cost_gamma=0.99,               # Keep same for cost discount
        cost_lam=0.95,                 # Match lam for consistency
        target_kl=0.01,                # Good conservative value
        cost_lim=cost_lim,             # Tighter constraint
        penalty_init=0.85,            # Much lower init (was 1) for gradual ramping
        penalty_lr=0.035,              # Improved from 5e-2 (0.05) for better responsiveness
        vf_lr=3e-4,                    # Improved from 1e-3 for better value function learning
        vf_iters=80,                   # Keep same             # Add explicit policy learning rate
        logger_kwargs=logger_kwargs,
        save_freq=10
    )
    
    training_time = time.time() - start_time
    
    print(f"\nPPO-Lagrangian Training Complete!")
    print(f"  Time: {training_time:.1f} seconds")
    print(f"  Training violation summary:")
    summary = counter.get_summary()
    print(f"    - Total episodes: {summary['total_episodes']}")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}")
    print(f"    - Episode violation rate: {summary['episode_violation_rate']:.3f}")
    print(f"    - Timesteps with violations: {summary['violation_timesteps']}/{summary['total_timesteps']}")
    print(f"    - Timestep violation rate: {summary['timestep_violation_rate']:.4f}")
    
    return summary


def train_cpo(counter: ConstraintViolationCounter):
    """Train Constrained Policy Optimization (CPO)"""
    print("\n" + "=" * 50)
    print("Training CPO (Constrained Policy Optimization)")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    def env_fn():
        env = gym.make('CartPole-v1')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)
    
    # Cost limit: stricter for CPO to enforce better constraint satisfaction
    cost_lim = 2  # match ppo_lagrangian
    
    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'cpo_{save_index}'),
        'exp_name': f'cpo_cartpole'
    }
    
    start_time = time.time()
    
    # Train CPO with optimized hyperparameters
    # Note: CPO in safe_rl uses same interface as PPO/PPO-Lagrangian
    cpo(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),  # Keep same for CartPole
        seed=SEED,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=0.99,                    # Standard discount factor
        lam=0.95,                      # GAE lambda parameter
        cost_gamma=0.99,               # Cost discount factor
        cost_lam=0.95,                 # Cost GAE lambda parameter
        target_kl=0.005,               # More conservative KL for better constraint satisfaction
        cost_lim=cost_lim,             # Constraint limit
        vf_lr=3e-4,                    # Value function learning rate
        vf_iters=80,                   # Value function training iterations
        logger_kwargs=logger_kwargs,
        save_freq=10
    )
    
    training_time = time.time() - start_time
    
    print(f"\nCPO Training Complete!")
    print(f"  Time: {training_time:.1f} seconds")
    print(f"  Training violation summary:")
    summary = counter.get_summary()
    print(f"    - Total episodes: {summary['total_episodes']}")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}")
    print(f"    - Episode violation rate: {summary['episode_violation_rate']:.3f}")
    print(f"    - Timesteps with violations: {summary['violation_timesteps']}/{summary['total_timesteps']}")
    print(f"    - Timestep violation rate: {summary['timestep_violation_rate']:.4f}")
    
    return summary


# =======================================
# EVALUATION FUNCTION
# =======================================

def evaluate_trained_policy(policy_path: str, n_episodes: int = 100) -> Dict:
    """
    Evaluate a trained policy and count constraint violations.
    
    Note: This is a placeholder since loading policies from safe_rl requires
    their specific format. In practice, you'd use their test_policy.py script.
    """
    print(f"\nEvaluating policy from: {policy_path}")
    print(f"  Running {n_episodes} episodes...")
    
    counter = ConstraintViolationCounter()
    env = ConstrainedCartPoleWrapper(gym.make('CartPole-v1'), counter, steps_per_epoch=4000)
    
    episode_returns = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        ep_ret = 0
        ep_len = 0
        done = False
        
        while not done:
            # For this demo, use random policy since loading trained models
            # requires the safe_rl specific loading mechanism
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            
            if ep_len >= 1000:  # CartPole max steps
                break
        
        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)
    
    # Record final episode
    counter.episode_ended(env.episode_had_violation)
    
    summary = counter.get_summary()
    summary['mean_return'] = np.mean(episode_returns)
    summary['std_return'] = np.std(episode_returns)
    summary['mean_length'] = np.mean(episode_lengths)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Return: {summary['mean_return']:.2f} ± {summary['std_return']:.2f}")
    print(f"  Mean Length: {summary['mean_length']:.1f}")
    print(f"  Violation Summary:")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}/{summary['total_episodes']}")
    print(f"    - Episode violation rate: {summary['episode_violation_rate']:.3f}")
    print(f"    - Timestep violation rate: {summary['timestep_violation_rate']:.4f}")
    
    return summary


# =======================================
# VISUALIZATION
# =======================================

def extract_training_data(ppo_dir: str, ppo_lag_dir: str, cpo_dir: str,
                         ppo_counter: ConstraintViolationCounter,
                         ppo_lag_counter: ConstraintViolationCounter,
                         cpo_counter: ConstraintViolationCounter) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract and combine training data from safe_rl logs and our violation counters"""
    
    # Load progress files from safe_rl logs
    ppo_progress = pd.read_csv(os.path.join(ppo_dir, 'progress.txt'), sep='\t')
    ppo_lag_progress = pd.read_csv(os.path.join(ppo_lag_dir, 'progress.txt'), sep='\t')
    cpo_progress = pd.read_csv(os.path.join(cpo_dir, 'progress.txt'), sep='\t')
    
    # Use actual per-epoch violation data from our counters
    ppo_violations_per_epoch = ppo_counter.violations_per_epoch
    ppo_lag_violations_per_epoch = ppo_lag_counter.violations_per_epoch
    cpo_violations_per_epoch = cpo_counter.violations_per_epoch
    
    # Ensure we have violation data for all epochs (pad with zeros if needed)
    num_epochs = len(ppo_progress)
    if len(ppo_violations_per_epoch) < num_epochs:
        ppo_violations_per_epoch.extend([0] * (num_epochs - len(ppo_violations_per_epoch)))
    if len(ppo_lag_violations_per_epoch) < num_epochs:
        ppo_lag_violations_per_epoch.extend([0] * (num_epochs - len(ppo_lag_violations_per_epoch)))
    if len(cpo_violations_per_epoch) < num_epochs:
        cpo_violations_per_epoch.extend([0] * (num_epochs - len(cpo_violations_per_epoch)))
    
    # Add actual violation data per epoch
    ppo_progress['ViolationsPerEpoch'] = ppo_violations_per_epoch[:num_epochs]
    ppo_lag_progress['ViolationsPerEpoch'] = ppo_lag_violations_per_epoch[:len(ppo_lag_progress)]
    cpo_progress['ViolationsPerEpoch'] = cpo_violations_per_epoch[:len(cpo_progress)]
    
    # Also add the violation rate
    ppo_progress['ViolationRate'] = ppo_counter.get_violation_rate()
    ppo_lag_progress['ViolationRate'] = ppo_lag_counter.get_violation_rate()
    cpo_progress['ViolationRate'] = cpo_counter.get_violation_rate()
    
    return ppo_progress, ppo_lag_progress, cpo_progress


def save_training_data_csv(ppo_progress: pd.DataFrame, ppo_lag_progress: pd.DataFrame, 
                          cpo_progress: pd.DataFrame, save_dir: str):
    """Save training data to CSV files"""
    print("\nSaving training data to CSV...")
    
    # Save PPO data
    ppo_csv_path = os.path.join(save_dir, 'ppo_training_data.csv')
    ppo_progress.to_csv(ppo_csv_path, index=False)
    print(f"  PPO data saved to: {ppo_csv_path}")
    
    # Save PPO-Lagrangian data
    ppo_lag_csv_path = os.path.join(save_dir, 'ppo_lagrangian_training_data.csv')
    ppo_lag_progress.to_csv(ppo_lag_csv_path, index=False)
    print(f"  PPO-Lagrangian data saved to: {ppo_lag_csv_path}")
    
    # Save CPO data
    cpo_csv_path = os.path.join(save_dir, 'cpo_training_data.csv')
    cpo_progress.to_csv(cpo_csv_path, index=False)
    print(f"  CPO data saved to: {cpo_csv_path}")
    
    # Save summary comparison
    summary_data = {
        'Algorithm': ['PPO', 'PPO-Lagrangian', 'CPO'],
        'Final_Avg_Return': [
            ppo_progress['AverageEpRet'].iloc[-1],
            ppo_lag_progress['AverageEpRet'].iloc[-1],
            cpo_progress['AverageEpRet'].iloc[-1]
        ],
        'Final_Avg_Length': [
            ppo_progress['EpLen'].iloc[-1],
            ppo_lag_progress['EpLen'].iloc[-1],
            cpo_progress['EpLen'].iloc[-1]
        ],
        'Final_Violation_Rate': [
            ppo_progress['ViolationRate'].iloc[-1],
            ppo_lag_progress['ViolationRate'].iloc[-1],
            cpo_progress['ViolationRate'].iloc[-1]
        ],
        'Total_Cost': [
            ppo_progress['CumulativeCost'].iloc[-1] if 'CumulativeCost' in ppo_progress.columns else 0,
            ppo_lag_progress['CumulativeCost'].iloc[-1] if 'CumulativeCost' in ppo_lag_progress.columns else 0,
            cpo_progress['CumulativeCost'].iloc[-1] if 'CumulativeCost' in cpo_progress.columns else 0
        ]
    }
    
    # Add penalty information for algorithms that use it
    lambda_values = [0, 0, 0]  # Default values
    if 'Penalty' in ppo_lag_progress.columns:
        lambda_values[1] = ppo_lag_progress['Penalty'].iloc[-1]
    summary_data['Final_Lambda'] = lambda_values
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(save_dir, 'training_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Summary saved to: {summary_csv_path}")


def plot_training_comparison(ppo_progress: pd.DataFrame, ppo_lag_progress: pd.DataFrame, 
                            cpo_progress: pd.DataFrame,
                            ppo_counter: ConstraintViolationCounter,
                            ppo_lag_counter: ConstraintViolationCounter,
                            cpo_counter: ConstraintViolationCounter,
                            save_path: str):
    """Plot comprehensive training comparison for PPO, PPO-Lagrangian, and CPO"""
    print("\nGenerating comparison plots...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Episode Returns vs Epoch
        axes[0,0].plot(ppo_progress['Epoch'], ppo_progress['AverageEpRet'], 
                       label='PPO', linewidth=2, alpha=0.8, marker='o', markersize=4, color='blue')
        axes[0,0].plot(ppo_lag_progress['Epoch'], ppo_lag_progress['AverageEpRet'], 
                       label='PPO-Lagrangian', linewidth=2, alpha=0.8, marker='s', markersize=4, color='green')
        axes[0,0].plot(cpo_progress['Epoch'], cpo_progress['AverageEpRet'], 
                       label='CPO', linewidth=2, alpha=0.8, marker='^', markersize=4, color='red')
        axes[0,0].set_xlabel('Epoch', fontsize=11)
        axes[0,0].set_ylabel('Average Episode Return', fontsize=11)
        axes[0,0].set_title('Return vs Epoch', fontsize=12, fontweight='bold')
        axes[0,0].legend(fontsize=10)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Actual Violations per Epoch (using real violation counter data)
        axes[0,1].plot(ppo_progress['Epoch'], ppo_progress['ViolationsPerEpoch'], 
                       label='PPO', linewidth=2, alpha=0.8, marker='o', markersize=4, color='blue')
        axes[0,1].plot(ppo_lag_progress['Epoch'], ppo_lag_progress['ViolationsPerEpoch'], 
                       label='PPO-Lagrangian', linewidth=2, alpha=0.8, marker='s', markersize=4, color='green')
        axes[0,1].plot(cpo_progress['Epoch'], cpo_progress['ViolationsPerEpoch'], 
                       label='CPO', linewidth=2, alpha=0.8, marker='^', markersize=4, color='red')
        axes[0,1].set_xlabel('Epoch', fontsize=11)
        axes[0,1].set_ylabel('Violations per Epoch', fontsize=11)
        axes[0,1].set_title('Violations per Epoch', fontsize=12, fontweight='bold')
        axes[0,1].legend(fontsize=10)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Lambda Value vs Epoch (PPO-Lagrangian only)
        if 'Penalty' in ppo_lag_progress.columns:
            axes[1,0].plot(ppo_lag_progress['Epoch'], ppo_lag_progress['Penalty'], 
                           label='PPO-Lagrangian Lambda', color='green', linewidth=2.5, 
                           alpha=0.8, marker='s', markersize=4)
            axes[1,0].set_xlabel('Epoch', fontsize=11)
            axes[1,0].set_ylabel('Lambda Value', fontsize=11)
            axes[1,0].set_title('Lambda Value vs Epoch (PPO-Lagrangian)', fontsize=12, fontweight='bold')
            axes[1,0].legend(fontsize=10)
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'No Lambda data available', 
                           ha='center', va='center', fontsize=12)
            axes[1,0].set_title('Lambda Value vs Epoch', fontsize=12, fontweight='bold')
        
        # 4. Cumulative Cost vs Epoch (if available)
        cost_plotted = False
        if 'CumulativeCost' in ppo_progress.columns:
            axes[1,1].plot(ppo_progress['Epoch'], ppo_progress['CumulativeCost'], 
                           label='PPO', linewidth=2, alpha=0.8, marker='o', markersize=4, color='blue')
            cost_plotted = True
        if 'CumulativeCost' in ppo_lag_progress.columns:
            axes[1,1].plot(ppo_lag_progress['Epoch'], ppo_lag_progress['CumulativeCost'], 
                           label='PPO-Lagrangian', linewidth=2, alpha=0.8, marker='s', markersize=4, color='green')
            cost_plotted = True
        if 'CumulativeCost' in cpo_progress.columns:
            axes[1,1].plot(cpo_progress['Epoch'], cpo_progress['CumulativeCost'], 
                           label='CPO', linewidth=2, alpha=0.8, marker='^', markersize=4, color='red')
            cost_plotted = True
        
        if cost_plotted:
            axes[1,1].set_xlabel('Epoch', fontsize=11)
            axes[1,1].set_ylabel('Cumulative Cost', fontsize=11)
            axes[1,1].set_title('Cumulative Cost vs Epoch', fontsize=12, fontweight='bold')
            axes[1,1].legend(fontsize=10)
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'No cost data available', 
                           ha='center', va='center', fontsize=12)
            axes[1,1].set_title('Cumulative Cost vs Epoch', fontsize=12, fontweight='bold')
        
        # Add overall title
        fig.suptitle('PPO vs PPO-Lagrangian vs CPO: Safe RL Comparison', fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plots saved to: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()


# =======================================
# MAIN EXECUTION
# =======================================

def main():
    """Main execution function"""
    
    training_results = {}
    
    # Train PPO
    ppo_counter = ConstraintViolationCounter()
    training_results['PPO'] = train_ppo(ppo_counter)
    
    # Reset TensorFlow graph between training runs (required for TF 1.x)
    tf.reset_default_graph()
    print("\n[TensorFlow graph reset]\n")
    
    # Train PPO-Lagrangian
    ppo_lag_counter = ConstraintViolationCounter()
    training_results['PPO-Lagrangian'] = train_ppo_lagrangian(ppo_lag_counter)
    
    # Reset TensorFlow graph between training runs (required for TF 1.x)
    tf.reset_default_graph()
    print("\n[TensorFlow graph reset]\n")
    
    # Train CPO
    cpo_counter = ConstraintViolationCounter()
    training_results['CPO'] = train_cpo(cpo_counter)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - Constraint Violation Comparison")
    print("=" * 80)
    
    for algo, results in training_results.items():
        print(f"\n{algo}:")
        print(f"  Episodes: {results['total_episodes']:,}")
        print(f"  Episodes with x-violations: {results['violation_episodes']:,}")
        print(f"  Episode violation rate: {results['episode_violation_rate']:.3f}")
        print(f"  Timesteps with violations: {results['violation_timesteps']:,}/{results['total_timesteps']:,}")
        print(f"  Timestep violation rate: {results['timestep_violation_rate']:.4f}")
    
    # Compare violation rates
    ppo_viol_rate = training_results['PPO']['episode_violation_rate']
    ppo_lag_viol_rate = training_results['PPO-Lagrangian']['episode_violation_rate']
    cpo_viol_rate = training_results['CPO']['episode_violation_rate']
    
    print(f"\n" + "=" * 50)
    print("VIOLATION RATE COMPARISON:")
    print(f"  PPO:           {ppo_viol_rate:.3f}")
    print(f"  PPO-Lagrangian: {ppo_lag_viol_rate:.3f}")
    print(f"  CPO:           {cpo_viol_rate:.3f}")
    
    if ppo_viol_rate > 0:
        ppo_lag_reduction = (ppo_viol_rate - ppo_lag_viol_rate) / ppo_viol_rate * 100
        cpo_reduction = (ppo_viol_rate - cpo_viol_rate) / ppo_viol_rate * 100
        print(f"\nReduction vs PPO:")
        print(f"  PPO-Lagrangian: {ppo_lag_reduction:+.1f}%")
        print(f"  CPO:           {cpo_reduction:+.1f}%")
    
    # Extract training data from logs
    print("\nProcessing training data...")
    ppo_progress, ppo_lag_progress, cpo_progress = extract_training_data(
        os.path.join(RUN_DIR, 'ppo'),
        os.path.join(RUN_DIR, 'ppo_lagrangian'),
        os.path.join(RUN_DIR, 'cpo'),
        ppo_counter,
        ppo_lag_counter,
        cpo_counter
    )
    
    # Save to CSV
    save_training_data_csv(ppo_progress, ppo_lag_progress, cpo_progress, RUN_DIR)
    
    # Generate comprehensive plots
    plot_training_comparison(
        ppo_progress,
        ppo_lag_progress,
        cpo_progress,
        ppo_counter,
        ppo_lag_counter,
        cpo_counter,
        os.path.join(RUN_DIR, f'comparison_{save_index}.png')
    )
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"Results saved in: {RUN_DIR}")
    print(f"  - PPO logs: {os.path.join(RUN_DIR, 'ppo')}")
    print(f"  - PPO-Lagrangian logs: {os.path.join(RUN_DIR, 'ppo_lagrangian')}")
    print(f"  - CPO logs: {os.path.join(RUN_DIR, 'cpo')}")
    print(f"  - Comparison plot: {os.path.join(RUN_DIR, 'comparison.png')}")
    print(f"  - PPO training CSV: {os.path.join(RUN_DIR, 'ppo_training_data.csv')}")
    print(f"  - PPO-Lagrangian training CSV: {os.path.join(RUN_DIR, 'ppo_lagrangian_training_data.csv')}")
    print(f"  - CPO training CSV: {os.path.join(RUN_DIR, 'cpo_training_data.csv')}")
    print(f"  - Summary CSV: {os.path.join(RUN_DIR, 'training_summary.csv')}")
    print("\nTo evaluate trained policies, use:")
    print(f"  cd {os.path.join(RUN_DIR, 'ppo')} && python ../../safety-starter-agents/scripts/test_policy.py")
    print(f"  cd {os.path.join(RUN_DIR, 'ppo_lagrangian')} && python ../../safety-starter-agents/scripts/test_policy.py")
    print(f"  cd {os.path.join(RUN_DIR, 'cpo')} && python ../../safety-starter-agents/scripts/test_policy.py")
    

if __name__ == "__main__":
    main()

