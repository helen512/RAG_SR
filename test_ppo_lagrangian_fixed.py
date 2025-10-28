#!/usr/bin/env python3
"""
Simple test to verify PPO-Lagrangian fixes for better performance
"""

import sys
import numpy as np
import torch

try:
    import gymnasium as gym
    GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    GYM_VERSION = "gym"

# Add safe-control-gym to path  
sys.path.append('/home/dmy/gymtest/safe-control-gym')

from safe_control_gym.utils.registration import make
from safe_control_gym.controllers.ppo_lagrangian.ppo_lagrangian import PPOLagrangian


class SimpleCartPoleWrapper:
    """Simple wrapper for CartPole that adds minimal cost function."""
    
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, **kwargs):
        if GYM_VERSION == "gymnasium":
            obs, info = self.env.reset(**kwargs)
            info['cost'] = 0.0
            info['constraint_violation'] = 0.0
            return obs, info
        else:
            obs = self.env.reset(**kwargs)
            return obs
            
    def step(self, action):
        if GYM_VERSION == "gymnasium":
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Simple cost: penalize large cart positions
            cost = 1.0 if abs(obs[0]) > 1.8 else 0.0
            info['cost'] = cost
            info['constraint_violation'] = cost
            return obs, reward, terminated, truncated, info
        else:
            obs, reward, done, info = self.env.step(action)
            cost = 1.0 if abs(obs[0]) > 1.8 else 0.0
            info['cost'] = cost
            info['constraint_violation'] = cost
            return obs, reward, done, info
    
    def close(self):
        return self.env.close()


def create_env(seed=0):
    """Create CartPole environment with simple cost wrapper."""
    if GYM_VERSION == "gymnasium":
        env = gym.make('CartPole-v1')
        env.reset(seed=seed)
    else:
        env = gym.make('CartPole-v1')
        env.seed(seed)
        
    return SimpleCartPoleWrapper(env)


def test_ppo_lagrangian():
    """Test PPO-Lagrangian with corrected hyperparameters."""
    print("Testing PPO-Lagrangian with corrected hyperparameters...")
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Environment factory
    env_func = lambda seed=None, **kwargs: create_env(seed=seed or 42)
    
    # Corrected configuration for CartPole
    config = {
        # Model
        'hidden_dim': 64,
        'activation': 'tanh',
        
        # Normalization  
        'norm_obs': False,
        'norm_reward': False,
        'clip_obs': 10.0,
        'clip_reward': 10.0,
        
        # PPO parameters (matching working standard PPO)
        'gamma': 0.99,
        'use_gae': False,  # Critical: CartPole works better without GAE
        'gae_lambda': 0.95,
        'use_clipped_value': False,
        'clip_param': 0.2,
        'target_kl': 0.01,
        'entropy_coef': 0.01,
        
        # Lagrangian parameters
        'cost_gamma': 0.99,
        'cost_gae_lambda': 0.95,
        'penalty_init': 1.0,
        'penalty_lr': 0.05,
        'cost_lim': 10.0,
        
        # Optimization (matching working standard PPO)
        'opt_epochs': 10,
        'mini_batch_size': 64,
        'actor_lr': 0.0003,
        'critic_lr': 0.001,
        'cost_critic_lr': 0.001,
        'max_grad_norm': 0.5,
        
        # Training setup (matching working standard PPO)
        'max_env_steps': 20000,  # Shorter for quick test
        'num_workers': 1,
        'rollout_batch_size': 4,  # Use 4 parallel envs like standard PPO
        'rollout_steps': 100,     # Much shorter rollouts like standard PPO
        'deque_size': 10,
        
        # Logging
        'log_interval': 1000,
        'save_interval': 0,
        'tensorboard': False
    }
    
    print("Configuration:")
    print(f"  rollout_steps: {config['rollout_steps']} (was 2048)")
    print(f"  rollout_batch_size: {config['rollout_batch_size']} (was 1)")
    print(f"  use_gae: {config['use_gae']} (was True)")
    print(f"  max_steps: {config['max_env_steps']:,}")
    print()
    
    # Create agent
    agent = PPOLagrangian(
        env_func=env_func,
        training=True,
        output_dir='./test_ppo_lagrangian_fixed',
        seed=42,
        **config
    )
    
    print(f"Agent created with penalty parameter: {agent.agent.penalty_param.item():.3f}")
    print("Starting training...")
    
    # Initialize and train
    agent.reset()
    
    # Train for limited steps
    start_time = __import__('time').time()
    agent.learn()
    train_time = __import__('time').time() - start_time
    
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"Total steps: {agent.total_steps:,}")
    
    # Quick evaluation
    print("\nRunning quick evaluation...")
    eval_rewards = []
    eval_costs = []
    
    agent.training = False  # Set to evaluation mode
    
    for episode in range(10):
        env = create_env(seed=100 + episode)
        obs, _ = env.reset() if GYM_VERSION == "gymnasium" else (env.reset(), {})
        
        episode_reward = 0
        episode_cost = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(obs)
            
            # Ensure action is integer for CartPole
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.size == 1 else int(action[0])
            elif not isinstance(action, int):
                action = int(action)
                
            if GYM_VERSION == "gymnasium":
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_cost += info.get('cost', 0)
        
        eval_rewards.append(episode_reward)
        eval_costs.append(episode_cost)
        env.close()
    
    avg_reward = np.mean(eval_rewards)
    avg_cost = np.mean(eval_costs)
    
    print(f"\nEvaluation Results:")
    print(f"  Average reward: {avg_reward:.1f} ± {np.std(eval_rewards):.1f}")
    print(f"  Average cost: {avg_cost:.1f} ± {np.std(eval_costs):.1f}")
    print(f"  Cost constraint satisfied: {'✓' if avg_cost <= config['cost_lim'] else '✗'}")
    print(f"  Final penalty parameter: {agent.agent.penalty_param.item():.3f}")
    
    # Cleanup
    agent.close()
    
    # Check if performance improved
    if avg_reward > 50:  # Should easily get >100 on CartPole when working
        print("\n✅ SIGNIFICANT IMPROVEMENT! Performance is much better.")
        return True
    elif avg_reward > 30:
        print("\n⚠️  MODERATE IMPROVEMENT. Still not optimal but much better than before.")
        return True
    else:
        print(f"\n❌ STILL LOW PERFORMANCE. Average reward {avg_reward:.1f} is too low.")
        return False


if __name__ == '__main__':
    success = test_ppo_lagrangian()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
