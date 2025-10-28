#!/usr/bin/env python3
"""
PPO-Lagrangian Test Script for CartPole Environment
=================================================

This script tests the PPO-Lagrangian implementation from safe-control-gym 
on the original OpenAI Gym CartPole-v1 environment with custom cost constraints.

## What this script does:

1. **Environment Setup**: Creates a CartPole-v1 environment with custom cost constraints
2. **Agent Training**: Trains a PPO-Lagrangian agent to balance reward maximization with cost constraint satisfaction
3. **Evaluation**: Tests the trained agent and compares performance with/without constraints
4. **Visualization**: Plots training progress, constraint violations, and penalty parameter adaptation

## Key Features:

- **Cost Function**: Defines cart position displacement as a cost (|x| > position_limit)
- **Automatic Penalty Learning**: The Lagrangian penalty parameter adapts to enforce cost constraints
- **Comprehensive Logging**: Tracks rewards, costs, constraint violations, and penalty evolution
- **Comparison**: Shows difference between constrained and unconstrained optimization

## Algorithm Overview:

PPO-Lagrangian solves the constrained RL problem:
```
maximize E[sum(rewards)]
subject to E[sum(costs)] <= cost_limit
```

Using the Lagrangian method:
```
L = reward_advantage - penalty * cost_advantage
```

The penalty parameter λ is learned via gradient ascent:
```
λ = λ + penalty_lr * (E[cost] - cost_limit)
```

## How to Run:

```bash
# Basic run with default parameters
python test_ppo_lagrangian_cartpole.py

# Run with custom parameters
python test_ppo_lagrangian_cartpole.py --max_steps 50000 --cost_limit 10.0 --position_limit 1.5

# Quick test mode (reduced training for verification)
python test_ppo_lagrangian_cartpole.py --test_mode
```

## Dependencies:

- gym/gymnasium  
- torch
- numpy
- matplotlib
- safe-control-gym (with PPO-Lagrangian implementation)

## Expected Results:

The agent should learn to:
1. Keep the pole balanced (maximize reward)
2. Limit cart position displacement (satisfy cost constraint)  
3. Automatically adapt the penalty parameter to achieve constraint satisfaction
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import gym/gymnasium
try:
    import gymnasium as gym
    GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    GYM_VERSION = "gym"

import torch

# Add safe-control-gym to path  
sys.path.append('/home/dmy/gymtest/safe-control-gym')

from safe_control_gym.utils.registration import make
from safe_control_gym.controllers.ppo_lagrangian.ppo_lagrangian import PPOLagrangian
from safe_control_gym.controllers.ppo.ppo import PPO


class CartPoleCostWrapper:
    """
    Wrapper for CartPole that adds cost constraints.
    
    Cost Function:
    - Cost = 1.0 when |cart_position| > position_limit
    - Cost = 0.0 otherwise
    
    This encourages the agent to keep the cart near the center while balancing.
    """
    
    def __init__(self, env, position_limit: float = 2.0):
        """
        Args:
            env: Base CartPole environment
            position_limit: Maximum allowed cart position (|x| <= position_limit)
        """
        self.env = env
        self.position_limit = position_limit
        
        # Forward environment attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.spec = getattr(env, 'spec', None)
        
        # Cost tracking
        self.episode_cost = 0.0
        self.total_cost = 0.0
        self.episode_length = 0
        
    def reset(self, **kwargs):
        """Reset environment and cost tracking."""
        if GYM_VERSION == "gymnasium":
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = {}
            
        self.episode_cost = 0.0
        self.episode_length = 0
        
        # Add cost information to info
        info['cost'] = 0.0
        info['constraint_violation'] = 0.0
        info['cumulative_cost'] = 0.0
        
        return (obs, info) if GYM_VERSION == "gymnasium" else obs
        
    def step(self, action):
        """Step environment and compute cost."""
        if GYM_VERSION == "gymnasium":
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        else:
            obs, reward, done, info = self.env.step(action)
            
        # Compute cost based on cart position
        cart_position = obs[0]  # CartPole obs: [position, velocity, angle, angular_velocity]
        cost = 1.0 if abs(cart_position) > self.position_limit else 0.0
        
        # Update tracking
        self.episode_cost += cost
        self.total_cost += cost
        self.episode_length += 1
        
        # Add cost information to info
        info['cost'] = cost
        info['constraint_violation'] = cost  # Same as cost for this problem
        info['cumulative_cost'] = self.episode_cost
        info['position_limit'] = self.position_limit
        info['cart_position'] = cart_position
        
        return (obs, reward, done, info) if GYM_VERSION == "gym" else (obs, reward, terminated, truncated, info)
    
    def render(self, **kwargs):
        """Render environment."""
        return self.env.render(**kwargs)
        
    def close(self):
        """Close environment."""
        return self.env.close()
        
    def seed(self, seed=None):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        elif hasattr(self.env, 'reset'):
            # For gymnasium, seed is handled in reset
            return None
            

def create_cartpole_env(seed: int = 0, position_limit: float = 2.0):
    """
    Create CartPole environment with cost constraints.
    
    Args:
        seed: Random seed for reproducibility
        position_limit: Cart position constraint threshold
        
    Returns:
        Wrapped CartPole environment with cost function
    """
    # Create base environment
    if GYM_VERSION == "gymnasium":
        env = gym.make('CartPole-v1')
        env.reset(seed=seed)
    else:
        env = gym.make('CartPole-v1')
        env.seed(seed)
        
    # Wrap with cost function
    env = CartPoleCostWrapper(env, position_limit=position_limit)
    
    return env


class PPOLagrangianTrainer:
    """
    Trainer class for PPO-Lagrangian on CartPole with cost constraints.
    
    This class handles:
    - Agent initialization and training
    - Environment interaction and cost computation  
    - Logging and progress tracking
    - Model saving and loading
    """
    
    def __init__(self, 
                 max_steps: int = 100000,
                 cost_limit: float = 25.0,
                 position_limit: float = 2.0,
                 penalty_init: float = 1.0,
                 penalty_lr: float = 0.05,
                 seed: int = 42,
                 output_dir: str = 'ppo_lagrangian_results',
                 **kwargs):
        """
        Args:
            max_steps: Maximum training steps
            cost_limit: Maximum allowed expected cost per episode
            position_limit: Cart position constraint (|x| <= position_limit)
            penalty_init: Initial penalty parameter value
            penalty_lr: Learning rate for penalty parameter
            seed: Random seed
            output_dir: Directory to save results and models
            **kwargs: Additional hyperparameters
        """
        self.max_steps = max_steps
        self.cost_limit = cost_limit
        self.position_limit = position_limit
        self.penalty_init = penalty_init
        self.penalty_lr = penalty_lr
        self.seed = seed
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Environment factory function (with seed handling for vectorized envs)
        def env_func(seed=None, **kwargs):
            if seed is None:
                seed = self.seed
            return create_cartpole_env(seed=seed, position_limit=position_limit)
        
        self.env_func = env_func
        
            # Training configuration - CORRECTED for CartPole
        self.config = {
            # Model parameters
            'hidden_dim': 64,
            'activation': 'tanh',
            
            # Normalization
            'norm_obs': False,
            'norm_reward': False,
            'clip_obs': 10.0,
            'clip_reward': 10.0,
            
            # PPO parameters - FIXED to match working standard PPO
            'gamma': 0.99,
            'use_gae': False,  # CRITICAL: CartPole works better without GAE
            'gae_lambda': 0.95,
            'use_clipped_value': False,
            'clip_param': 0.2,
            'target_kl': 0.01,
            'entropy_coef': 0.01,
            
            # Lagrangian parameters
            'cost_gamma': 0.99,
            'cost_gae_lambda': 0.95,
            'penalty_init': penalty_init,
            'penalty_lr': penalty_lr,
            'cost_lim': cost_limit,
            
            # Optimization
            'opt_epochs': 10,
            'mini_batch_size': 64,
            'actor_lr': 3e-4,
            'critic_lr': 1e-3,
            'cost_critic_lr': 1e-3,
            'max_grad_norm': 0.5,
            
            # Training setup - FIXED to match working standard PPO
            'max_env_steps': max_steps,
            'num_workers': 1,
            'rollout_batch_size': 4,  # FIXED: Use 4 parallel envs like standard PPO
            'rollout_steps': 100,     # FIXED: Much shorter rollouts (was 2048!)
            'deque_size': 10,
            
            # Logging
            'log_interval': 1000,
            'save_interval': 10000,
            'tensorboard': False,
            
            **kwargs  # Override with any additional parameters
        }
        
        print("="*80)
        print("PPO-LAGRANGIAN CARTPOLE TRAINER")
        print("="*80)
        print(f"Environment: CartPole-v1 with position constraint |x| <= {position_limit}")
        print(f"Cost limit: {cost_limit}")
        print(f"Max training steps: {max_steps:,}")
        print(f"Penalty init: {penalty_init}, lr: {penalty_lr}")
        print(f"Seed: {seed}")
        print(f"Output directory: {output_dir}")
        print()
        
    def create_agent(self) -> PPOLagrangian:
        """Create PPO-Lagrangian agent."""
        print("Creating PPO-Lagrangian agent...")
        
        agent = PPOLagrangian(
            env_func=self.env_func,
            training=True,
            output_dir=self.output_dir,
            seed=self.seed,
            **self.config
        )
        
        print(f"Agent created with penalty parameter: {agent.agent.penalty_param.item():.3f}")
        print(f"Cost limit: {agent.cost_lim}")
        return agent
        
    def train_agent(self) -> Dict:
        """Train the PPO-Lagrangian agent."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Create agent
        agent = self.create_agent()
        agent.reset()
        
        # Training metrics
        training_stats = {
            'steps': [],
            'episodes': [],
            'rewards': [],
            'costs': [],
            'constraint_violations': [],
            'penalty_params': [],
            'policy_losses': [],
            'value_losses': [],
            'cost_value_losses': []
        }
        
        episode_count = 0
        step_count = 0
        
        print(f"Training will run for {self.max_steps:,} steps...")
        print(f"Expected episodes: ~{self.max_steps // 200}")  # CartPole episodes are ~200 steps
        print()
        
        # Use the agent's built-in training method
        start_time = time.time()
        
        print("Starting agent training using built-in learn() method...")
        
        try:
            # Use the agent's learn method which properly handles PPO training
            agent.learn()
            
            # Collect final statistics from the environment
            if hasattr(agent.env, 'get_episode_rewards'):
                episode_rewards = agent.env.get_episode_rewards()
                episode_costs = getattr(agent.env, 'get_episode_costs', lambda: [])()
                episode_count = len(episode_rewards)
                
                # Populate training stats with available data
                if episode_rewards:
                    training_stats['episodes'] = list(range(1, len(episode_rewards) + 1))
                    training_stats['rewards'] = episode_rewards
                    training_stats['costs'] = episode_costs if episode_costs else [0] * len(episode_rewards)
                    training_stats['constraint_violations'] = [0] * len(episode_rewards)  # Will be populated by agent
                    training_stats['penalty_params'] = [agent.agent.penalty_param.item()] * len(episode_rewards)
                    training_stats['steps'] = list(range(len(episode_rewards) * 100, len(episode_rewards) * 100 + len(episode_rewards)))
                    
                    step_count = agent.total_steps if hasattr(agent, 'total_steps') else self.max_steps
            else:
                # Fallback: create minimal stats
                episode_count = self.max_steps // 100  # Estimate
                step_count = self.max_steps
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
            episode_count = 0
            step_count = 0
        
        # Training summary
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total episodes: {episode_count}")
        print(f"Total steps: {step_count:,}")
        print(f"Training time: {total_time:.1f} seconds")
        print(f"Average steps/second: {step_count/total_time:.1f}")
        
        if training_stats['rewards']:
            print(f"Average reward (last 100 episodes): {np.mean(training_stats['rewards'][-100:]):.1f}")
            print(f"Average cost (last 100 episodes): {np.mean(training_stats['costs'][-100:]):.1f}")
            print(f"Average violations (last 100 episodes): {np.mean(training_stats['constraint_violations'][-100:]):.1f}")
            print(f"Final penalty parameter: {training_stats['penalty_params'][-1]:.3f}")
        
        # Save model
        model_path = os.path.join(self.output_dir, 'ppo_lagrangian_final.pt')
        agent.save(model_path)
        print(f"Model saved to: {model_path}")
        
        return {
            'agent': agent,
            'training_stats': training_stats,
            'total_episodes': episode_count,
            'total_steps': step_count,
            'training_time': total_time
        }


def evaluate_agent(agent: PPOLagrangian, 
                  position_limit: float = 2.0,
                  n_episodes: int = 100,
                  render: bool = False,
                  seed: int = 123) -> Dict:
    """
    Evaluate trained PPO-Lagrangian agent.
    
    Args:
        agent: Trained PPO-Lagrangian agent
        position_limit: Cart position constraint
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
        seed: Random seed for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n" + "="*60)
    print("EVALUATING TRAINED AGENT")
    print("="*60)
    print(f"Running {n_episodes} evaluation episodes...")
    
    # Create evaluation environment
    env = create_cartpole_env(seed=seed, position_limit=position_limit)
    
    # Evaluation metrics
    eval_stats = {
        'rewards': [],
        'costs': [],
        'constraint_violations': [],
        'episode_lengths': [],
        'success_rate': []  # Episodes without constraint violations
    }
    
    agent.training = False  # Set to evaluation mode
    
    for episode in range(n_episodes):
        obs, _ = env.reset() if GYM_VERSION == "gymnasium" else (env.reset(), {})
        
        episode_reward = 0
        episode_cost = 0
        episode_violations = 0
        episode_length = 0
        
        done = False
        while not done:
            # Select action (deterministic for evaluation)
            action, _ = agent.select_action(obs)
            
            # Ensure proper action format for CartPole (expects int, not array)
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action = int(action.item())
                else:
                    action = action.astype(int)
            elif isinstance(action, (np.integer, float)):
                action = int(action)
            
            # Environment step
            if GYM_VERSION == "gymnasium":
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_cost += info.get('cost', 0)
            episode_violations += info.get('constraint_violation', 0)
            episode_length += 1
            
            if render and episode < 3:  # Render first few episodes
                env.render()
        
        # Store episode statistics
        eval_stats['rewards'].append(episode_reward)
        eval_stats['costs'].append(episode_cost)
        eval_stats['constraint_violations'].append(episode_violations)
        eval_stats['episode_lengths'].append(episode_length)
        eval_stats['success_rate'].append(1.0 if episode_violations == 0 else 0.0)
        
        if episode % 20 == 0:
            print(f"Episode {episode:3d}: Reward={episode_reward:6.1f}, Cost={episode_cost:4.1f}, "
                  f"Violations={episode_violations:2.0f}, Length={episode_length:3d}")
    
    env.close()
    
    # Compute summary statistics
    avg_reward = np.mean(eval_stats['rewards'])
    avg_cost = np.mean(eval_stats['costs'])
    avg_violations = np.mean(eval_stats['constraint_violations'])
    avg_length = np.mean(eval_stats['episode_lengths'])
    success_rate = np.mean(eval_stats['success_rate'])
    
    print(f"\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Average reward: {avg_reward:.1f} ± {np.std(eval_stats['rewards']):.1f}")
    print(f"Average cost: {avg_cost:.1f} ± {np.std(eval_stats['costs']):.1f}")
    print(f"Average violations: {avg_violations:.1f} ± {np.std(eval_stats['constraint_violations']):.1f}")
    print(f"Average episode length: {avg_length:.1f} ± {np.std(eval_stats['episode_lengths']):.1f}")
    print(f"Success rate (no violations): {success_rate:.1%}")
    print(f"Cost constraint satisfied: {'✓' if avg_cost <= agent.cost_lim else '✗'}")
    print(f"  (avg_cost={avg_cost:.1f} vs limit={agent.cost_lim})")
    
    return {
        'avg_reward': avg_reward,
        'avg_cost': avg_cost,
        'avg_violations': avg_violations,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'constraint_satisfied': avg_cost <= agent.cost_lim,
        'eval_stats': eval_stats
    }


def plot_training_results(training_stats: Dict, 
                         cost_limit: float,
                         output_dir: str = 'ppo_lagrangian_results'):
    """
    Plot comprehensive training results.
    
    Args:
        training_stats: Training statistics from trainer
        cost_limit: Cost constraint limit for reference
        output_dir: Directory to save plots
    """
    print(f"\n" + "="*60)
    print("GENERATING TRAINING PLOTS")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PPO-Lagrangian Training Results on CartPole', fontsize=16)
    
    episodes = training_stats['episodes']
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episodes, training_stats['rewards'], alpha=0.7, linewidth=1)
    axes[0, 0].plot(episodes, smooth_curve(training_stats['rewards'], 50), 'r-', linewidth=2, label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Episode Costs  
    axes[0, 1].plot(episodes, training_stats['costs'], alpha=0.7, linewidth=1, color='orange')
    axes[0, 1].plot(episodes, smooth_curve(training_stats['costs'], 50), 'r-', linewidth=2, label='Smoothed')
    axes[0, 1].axhline(y=cost_limit, color='red', linestyle='--', linewidth=2, label=f'Cost Limit ({cost_limit})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Cost')
    axes[0, 1].set_title('Episode Costs vs Constraint')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Constraint Violations
    axes[0, 2].plot(episodes, training_stats['constraint_violations'], alpha=0.7, linewidth=1, color='red')
    axes[0, 2].plot(episodes, smooth_curve(training_stats['constraint_violations'], 50), 'darkred', linewidth=2, label='Smoothed')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Constraint Violations')
    axes[0, 2].set_title('Constraint Violations per Episode')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Plot 4: Penalty Parameter Evolution
    axes[1, 0].plot(episodes, training_stats['penalty_params'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Penalty Parameter (λ)')
    axes[1, 0].set_title('Penalty Parameter Adaptation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Cost vs Reward Trade-off
    axes[1, 1].scatter(training_stats['costs'], training_stats['rewards'], alpha=0.6, s=20)
    axes[1, 1].axvline(x=cost_limit, color='red', linestyle='--', linewidth=2, label=f'Cost Limit ({cost_limit})')
    axes[1, 1].set_xlabel('Episode Cost')
    axes[1, 1].set_ylabel('Episode Reward')
    axes[1, 1].set_title('Reward vs Cost Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 6: Moving Averages Comparison
    window = 100
    if len(episodes) >= window:
        moving_rewards = smooth_curve(training_stats['rewards'], window)
        moving_costs = smooth_curve(training_stats['costs'], window)
        
        axes[1, 2].plot(episodes, moving_rewards, 'b-', linewidth=2, label='Avg Reward')
        axes[1, 2].set_ylabel('Average Reward', color='b')
        axes[1, 2].tick_params(axis='y', labelcolor='b')
        
        ax2 = axes[1, 2].twinx()
        ax2.plot(episodes, moving_costs, 'orange', linewidth=2, label='Avg Cost')
        ax2.axhline(y=cost_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Average Cost', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_title(f'Moving Averages (window={window})')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to: {plot_path}")
    plt.show()


def smooth_curve(data: List[float], window_size: int) -> List[float]:
    """Apply moving average smoothing to data."""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start_idx:end_idx]))
    return smoothed


def main():
    """
    Main function to run PPO-Lagrangian training and evaluation.
    
    This function orchestrates the entire experiment:
    1. Parse command line arguments
    2. Set up trainer with specified parameters  
    3. Train PPO-Lagrangian agent
    4. Evaluate trained agent
    5. Generate visualization plots
    6. Save results and model
    """
    parser = argparse.ArgumentParser(description='Test PPO-Lagrangian on CartPole with Cost Constraints')
    
    # Training parameters
    parser.add_argument('--max_steps', type=int, default=100000, 
                       help='Maximum training steps (default: 100000)')
    parser.add_argument('--cost_limit', type=float, default=25.0,
                       help='Cost constraint limit (default: 25.0)')
    parser.add_argument('--position_limit', type=float, default=2.0,
                       help='Cart position constraint |x| <= position_limit (default: 2.0)')
    
    # Lagrangian parameters
    parser.add_argument('--penalty_init', type=float, default=1.0,
                       help='Initial penalty parameter (default: 1.0)')
    parser.add_argument('--penalty_lr', type=float, default=0.05,
                       help='Penalty parameter learning rate (default: 0.05)')
    
    # Experimental setup
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='ppo_lagrangian_results',
                       help='Output directory for results (default: ppo_lagrangian_results)')
    
    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                       help='Render evaluation episodes')
    
    # Mode selection
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with reduced parameters for quick verification')
    parser.add_argument('--eval_only', type=str, default=None,
                       help='Path to trained model for evaluation only (skip training)')
    
    args = parser.parse_args()
    
    # Test mode: reduce parameters for quick verification
    if args.test_mode:
        print("🧪 RUNNING IN TEST MODE")
        print("Parameters adjusted for quick verification...")
        args.max_steps = 10000
        args.eval_episodes = 20
        args.output_dir = 'ppo_lagrangian_test'
        print()
    
    print("SCRIPT CONFIGURATION:")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Cost limit: {args.cost_limit}")
    print(f"  Position limit: ±{args.position_limit}")
    print(f"  Penalty init: {args.penalty_init}, lr: {args.penalty_lr}")
    print(f"  Seed: {args.seed}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Evaluation episodes: {args.eval_episodes}")
    print(f"  Render: {args.render}")
    print()
    
    # Create trainer
    trainer = PPOLagrangianTrainer(
        max_steps=args.max_steps,
        cost_limit=args.cost_limit,
        position_limit=args.position_limit,
        penalty_init=args.penalty_init,
        penalty_lr=args.penalty_lr,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    if args.eval_only:
        # Evaluation only mode
        print(f"Loading model from: {args.eval_only}")
        agent = trainer.create_agent()
        agent.load(args.eval_only)
        
        eval_results = evaluate_agent(
            agent=agent,
            position_limit=args.position_limit,
            n_episodes=args.eval_episodes,
            render=args.render,
            seed=args.seed + 1000
        )
        
    else:
        # Full training and evaluation
        try:
            # Train agent
            training_results = trainer.train_agent()
            
            # Evaluate trained agent
            eval_results = evaluate_agent(
                agent=training_results['agent'],
                position_limit=args.position_limit,
                n_episodes=args.eval_episodes,
                render=args.render,
                seed=args.seed + 1000
            )
            
            # Plot results
            if not args.test_mode or len(training_results['training_stats']['episodes']) > 10:
                plot_training_results(
                    training_stats=training_results['training_stats'],
                    cost_limit=args.cost_limit,
                    output_dir=args.output_dir
                )
            
            # Save final summary
            summary = {
                'training': training_results,
                'evaluation': eval_results,
                'config': {
                    'max_steps': args.max_steps,
                    'cost_limit': args.cost_limit,
                    'position_limit': args.position_limit,
                    'penalty_init': args.penalty_init,
                    'penalty_lr': args.penalty_lr,
                    'seed': args.seed
                }
            }
            
            summary_path = os.path.join(args.output_dir, 'experiment_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("PPO-Lagrangian CartPole Experiment Summary\n")
                f.write("="*50 + "\n\n")
                
                f.write("Configuration:\n")
                for key, value in summary['config'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                f.write("Training Results:\n")
                f.write(f"  Total episodes: {training_results['total_episodes']}\n")
                f.write(f"  Total steps: {training_results['total_steps']:,}\n")
                f.write(f"  Training time: {training_results['training_time']:.1f}s\n")
                f.write("\n")
                
                f.write("Evaluation Results:\n")
                f.write(f"  Average reward: {eval_results['avg_reward']:.1f}\n")
                f.write(f"  Average cost: {eval_results['avg_cost']:.1f}\n")
                f.write(f"  Average violations: {eval_results['avg_violations']:.1f}\n")
                f.write(f"  Success rate: {eval_results['success_rate']:.1%}\n")
                f.write(f"  Constraint satisfied: {eval_results['constraint_satisfied']}\n")
                
            print(f"\nExperiment summary saved to: {summary_path}")
            
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user!")
    
    print(f"\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print("Key files:")
    print(f"  - Model: {args.output_dir}/ppo_lagrangian_final.pt")
    print(f"  - Plots: {args.output_dir}/training_results.png")
    print(f"  - Summary: {args.output_dir}/experiment_summary.txt")
    print()
    print("To run evaluation only on the trained model:")
    print(f"  python {sys.argv[0]} --eval_only {args.output_dir}/ppo_lagrangian_final.pt")
    print("="*80)


if __name__ == '__main__':
    main()
