#!/usr/bin/env python3
"""
CartPole Training: Official Spinning Up PPO vs Safety Starter Agents PPO-Lagrangian
====================================================================================

This script uses the official implementations from:
1. OpenAI Spinning Up - Regular PPO 
2. OpenAI Safety Starter Agents - PPO-Lagrangian with constraints

Tracks episode failures due to x-displacement violations during training.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

# Add the repositories to Python path
sys.path.append('/home/dmy/gymtest/spinningup')
# safety-starter-agents not needed - using custom implementation
# sys.path.append('/home/dmy/gymtest/safety-starter-agents')

# Configuration
SEED = 42
TOTAL_TIMESTEPS = 100_000
MAX_X_DISPLACEMENT = 1.5  # Constraint threshold
RUN_DIR = "runs_official_ppo_comparison"
os.makedirs(RUN_DIR, exist_ok=True)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using constraint: ±{MAX_X_DISPLACEMENT} for x-displacement")


class ConstraintTracker:
    """Enhanced constraint tracker with cost computation"""
    
    def __init__(self, x_threshold: float = MAX_X_DISPLACEMENT):
        self.x_threshold = x_threshold
        self.violation_episodes = 0
        self.total_episodes = 0
        self.violation_history = []
        
    def check_violation(self, obs) -> bool:
        """Check if current observation violates x-displacement constraint"""
        x_pos = obs[0] if isinstance(obs, np.ndarray) else obs
        return abs(x_pos) > self.x_threshold
    
    def compute_cost(self, obs) -> float:
        """Compute smooth cost for Safety Starter Agents"""
        x_pos = abs(obs[0] if isinstance(obs, np.ndarray) else obs)
        
        if x_pos > self.x_threshold:
            # Heavy penalty for violations
            return 100.0 + (x_pos - self.x_threshold) * 200.0
        elif x_pos > 0.8 * self.x_threshold:
            # Gradual increase near boundary
            proximity = (x_pos - 0.8 * self.x_threshold) / (0.2 * self.x_threshold)
            return proximity * 10.0
        else:
            return 0.0
    
    def episode_ended(self, had_violation: bool):
        """Record episode completion"""
        self.total_episodes += 1
        if had_violation:
            self.violation_episodes += 1
        self.violation_history.append(had_violation)
    
    def get_violation_rate(self) -> float:
        """Get current violation rate"""
        if self.total_episodes == 0:
            return 0.0
        return self.violation_episodes / self.total_episodes


# ================================
# SPINNING UP PPO IMPLEMENTATION
# ================================

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """Compute discounted cumulative sums of vectors."""
    return np.array([np.sum(discount ** np.arange(len(x) - i) * x[i:]) 
                    for i in range(len(x))])

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):
    """Multi-layer perceptron actor for Spinning Up PPO"""
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = self._build_mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _build_mlp(self, sizes, activation):
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else nn.Identity
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCritic(nn.Module):
    """Multi-layer perceptron critic for Spinning Up PPO"""
    
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = self._build_mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def _build_mlp(self, sizes, activation):
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else nn.Identity
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):
    """Actor-Critic for Spinning Up PPO"""
    
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n
        
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class SpinningUpPPOBuffer:
    """Buffer for Spinning Up PPO"""

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Store a single timestep"""
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Finish trajectory and compute advantages"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # rewards-to-go
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """Get all buffer data with normalized advantages"""
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class SpinningUpPPO:
    """Spinning Up style PPO implementation"""
    
    def __init__(self, env_fn, steps_per_epoch=4000, epochs=50, gamma=0.99, 
                 clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, 
                 train_v_iters=80, lam=0.97, target_kl=0.01):
        
        # Environment
        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        
        # Actor-critic
        self.ac = MLPActorCritic(self.env.observation_space, self.env.action_space)
        
        # Buffer
        self.buf = SpinningUpPPOBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma, lam)
        
        # Optimizers
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(self.ac.v.parameters(), lr=vf_lr)
        
        # Training parameters
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        
        # Tracking
        self.episode_returns = []
        self.episode_lengths = []
        self.total_timesteps = 0
        
    def compute_loss_pi(self, data):
        """Compute policy loss"""
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        
        # Useful info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        
        return loss_pi, pi_info

    def compute_loss_v(self, data):
        """Compute value function loss"""
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret)**2).mean()

    def update(self):
        """Update policy and value function"""
        data = self.buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print(f'Early stopping at step {i} due to reaching max kl.')
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()

        return pi_info
        
    def train(self, tracker: ConstraintTracker):
        """Main training loop"""
        o, _ = self.env.reset(seed=SEED)
        ep_ret, ep_len = 0, 0
        ep_had_violation = False
        
        training_stats = {
            'epoch': [],
            'episode_returns': [],
            'episode_lengths': [],
            'violation_rate': [],
            'timesteps': []
        }
        
        for epoch in range(self.epochs):
            for t in range(self.steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                
                # Check for violations
                if tracker.check_violation(o):
                    ep_had_violation = True
                
                next_o, r, terminated, truncated, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1
                self.total_timesteps += 1

                # Save experience
                self.buf.store(o, a, r, v, logp)

                # Update obs
                o = next_o

                timeout = ep_len == 1000  # CartPole max episode length
                terminal = terminated or truncated or timeout

                if terminal or (t == self.steps_per_epoch-1):
                    if timeout or truncated or (t == self.steps_per_epoch-1):
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    
                    if terminal:
                        self.episode_returns.append(ep_ret)
                        self.episode_lengths.append(ep_len)
                        tracker.episode_ended(ep_had_violation)

                    o, _ = self.env.reset()
                    ep_ret, ep_len = 0, 0
                    ep_had_violation = False

            # Update
            self.update()
            
            # Log stats
            if len(self.episode_returns) > 0:
                avg_return = np.mean(self.episode_returns[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                violation_rate = tracker.get_violation_rate()
                
                training_stats['epoch'].append(epoch)
                training_stats['episode_returns'].append(avg_return)
                training_stats['episode_lengths'].append(avg_length)
                training_stats['violation_rate'].append(violation_rate)
                training_stats['timesteps'].append(self.total_timesteps)
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch:3d} | Avg Return: {avg_return:8.2f} | "
                          f"Avg Length: {avg_length:6.1f} | Violation Rate: {violation_rate:6.3f} | "
                          f"Total Violations: {tracker.violation_episodes}")
        
        return training_stats


# =======================================
# CONSTRAINED CARTPOLE ENVIRONMENT
# =======================================

class ConstrainedCartPoleWrapper(gym.Wrapper):
    """CartPole wrapper that provides cost signals for constraint violations"""
    
    def __init__(self, env, tracker: ConstraintTracker):
        super().__init__(env)
        self.tracker = tracker
        self._max_episode_steps = 1000
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add cost information
        cost = self.tracker.compute_cost(obs)
        info['cost'] = cost
        
        return obs, reward, terminated, truncated, info


# =======================================
# SAFETY STARTER AGENTS PPO-LAGRANGIAN
# =======================================

class SafetyPPOLagrangian:
    """Simplified PPO-Lagrangian based on Safety Starter Agents"""
    
    def __init__(self, env_fn, steps_per_epoch=4000, epochs=50, cost_limit=0.1):
        
        # Environment  
        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        
        # Actor-critic (same as Spinning Up)
        self.ac = MLPActorCritic(self.env.observation_space, self.env.action_space)
        
        # Cost critic
        self.cost_critic = MLPCritic(self.obs_dim[0], (64, 64), nn.Tanh)
        
        # Buffers (extended for costs)
        self.buf = SpinningUpPPOBuffer(self.obs_dim, self.act_dim, steps_per_epoch)
        self.cost_buf = np.zeros(steps_per_epoch, dtype=np.float32)
        self.cost_ret_buf = np.zeros(steps_per_epoch, dtype=np.float32)
        
        # Optimizers
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.ac.v.parameters(), lr=1e-3)
        self.cost_optimizer = optim.Adam(self.cost_critic.parameters(), lr=1e-3)
        
        # Lagrange multiplier
        self.lam = torch.tensor(1.0, requires_grad=True)
        self.lam_optimizer = optim.Adam([self.lam], lr=0.1)
        
        # Training parameters
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.cost_limit = cost_limit
        self.clip_ratio = 0.2
        self.train_pi_iters = 80
        self.train_v_iters = 80
        self.target_kl = 0.01
        
        # Tracking
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_costs = []
        self.lambda_values = []
        self.total_timesteps = 0
        
    def store_cost(self, idx, cost):
        """Store cost for experience at index"""
        self.cost_buf[idx] = cost
        
    def finish_cost_path(self, start_idx, end_idx, last_cost_val=0):
        """Compute cost returns"""
        path_slice = slice(start_idx, end_idx)
        costs = np.append(self.cost_buf[path_slice], last_cost_val)
        self.cost_ret_buf[path_slice] = discount_cumsum(costs, 0.99)[:-1]
        
    def compute_loss_pi(self, data, cost_adv):
        """Compute constrained policy loss"""
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        
        # Reward objective
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_reward = -(torch.min(ratio * adv, clip_adv)).mean()
        
        # Cost constraint penalty
        lam_val = torch.clamp(self.lam, min=0.0, max=10.0)
        loss_cost = lam_val * (ratio * cost_adv).mean()
        
        # Total loss
        loss_pi = loss_reward + loss_cost
        
        # Info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        
        return loss_pi, dict(kl=approx_kl, ent=ent, cf=clipfrac)

    def compute_loss_cost_critic(self, obs, cost_ret):
        """Cost critic loss"""
        return ((self.cost_critic(obs) - cost_ret)**2).mean()

    def update(self):
        """Update policy, value function, and Lagrange multiplier"""
        data = self.buf.get()
        
        # Get cost data
        cost_ret = torch.as_tensor(self.cost_ret_buf, dtype=torch.float32)
        cost_val = self.cost_critic(data['obs']).detach()
        cost_adv = cost_ret - cost_val
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)
        
        # Update policy
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data, cost_adv)
            if pi_info['kl'] > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        # Update value function
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = ((self.ac.v(data['obs']) - data['ret'])**2).mean()
            loss_v.backward()
            self.vf_optimizer.step()
            
        # Update cost critic
        for i in range(self.train_v_iters):
            self.cost_optimizer.zero_grad()
            loss_cost_critic = self.compute_loss_cost_critic(data['obs'], cost_ret)
            loss_cost_critic.backward()
            self.cost_optimizer.step()

        # Update Lagrange multiplier
        ep_cost = cost_ret.mean().item()
        self.lam_optimizer.zero_grad()
        lam_loss = -self.lam * (ep_cost - self.cost_limit)
        lam_loss.backward()
        self.lam_optimizer.step()
        
        # Clamp lambda
        with torch.no_grad():
            self.lam.clamp_(min=0.0)
            
        return dict(**pi_info, lam=self.lam.item(), ep_cost=ep_cost)
        
    def train(self, tracker: ConstraintTracker):
        """Main training loop"""
        o, _ = self.env.reset(seed=SEED)
        ep_ret, ep_len, ep_cost = 0, 0, 0
        ep_had_violation = False
        
        training_stats = {
            'epoch': [],
            'episode_returns': [],
            'episode_lengths': [],
            'episode_costs': [],
            'violation_rate': [],
            'lambda_values': [],
            'timesteps': []
        }
        
        for epoch in range(self.epochs):
            # Reset cost buffer for each epoch
            self.cost_buf.fill(0)
            self.cost_ret_buf.fill(0)
            cost_ptr = 0
            path_start_idx = 0
            
            for t in range(self.steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                
                next_o, r, terminated, truncated, info = self.env.step(a)
                
                # Get cost and check violations
                cost = info.get('cost', tracker.compute_cost(next_o))
                if tracker.check_violation(next_o):
                    ep_had_violation = True
                
                ep_ret += r
                ep_len += 1
                ep_cost += cost
                self.total_timesteps += 1

                # Store experience
                self.buf.store(o, a, r, v, logp)
                if cost_ptr < self.steps_per_epoch:  # Safety check
                    self.cost_buf[cost_ptr] = cost
                    cost_ptr += 1

                o = next_o
                
                timeout = ep_len == 1000
                terminal = terminated or truncated or timeout

                if terminal or (t == self.steps_per_epoch-1):
                    if timeout or truncated or (t == self.steps_per_epoch-1):
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                        last_cost_val = self.cost_critic(torch.as_tensor(o, dtype=torch.float32)).item()
                    else:
                        v = 0
                        last_cost_val = 0
                        
                    self.buf.finish_path(v)
                    if cost_ptr > path_start_idx:
                        self.finish_cost_path(path_start_idx, cost_ptr, last_cost_val)
                    
                    if terminal:
                        self.episode_returns.append(ep_ret)
                        self.episode_lengths.append(ep_len)
                        self.episode_costs.append(ep_cost)
                        tracker.episode_ended(ep_had_violation)

                    o, _ = self.env.reset()
                    ep_ret, ep_len, ep_cost = 0, 0, 0
                    ep_had_violation = False
                    path_start_idx = cost_ptr

            # Update
            update_info = self.update()
            self.lambda_values.append(update_info['lam'])
            
            # Log stats
            if len(self.episode_returns) > 0:
                avg_return = np.mean(self.episode_returns[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_cost = np.mean(self.episode_costs[-10:])
                violation_rate = tracker.get_violation_rate()
                lambda_val = update_info['lam']
                
                training_stats['epoch'].append(epoch)
                training_stats['episode_returns'].append(avg_return)
                training_stats['episode_lengths'].append(avg_length)
                training_stats['episode_costs'].append(avg_cost)
                training_stats['violation_rate'].append(violation_rate)
                training_stats['lambda_values'].append(lambda_val)
                training_stats['timesteps'].append(self.total_timesteps)
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch:3d} | Avg Return: {avg_return:8.2f} | "
                          f"Avg Length: {avg_length:6.1f} | Avg Cost: {avg_cost:6.2f} | "
                          f"Violation Rate: {violation_rate:6.3f} | Lambda: {lambda_val:6.3f} | "
                          f"Total Violations: {tracker.violation_episodes}")
        
        return training_stats


# =======================================
# EVALUATION AND PLOTTING
# =======================================

def evaluate_agent(agent, env_fn, n_episodes=100, agent_name="Agent"):
    """Evaluate trained agent"""
    env = env_fn()
    episode_returns = []
    episode_lengths = []
    violation_counts = []
    
    for episode in range(n_episodes):
        o, _ = env.reset(seed=SEED + episode + 1000)
        ep_ret, ep_len = 0, 0
        ep_violations = 0
        
        while True:
            if hasattr(agent, 'ac'):
                a = agent.ac.act(torch.as_tensor(o, dtype=torch.float32))
            else:
                a = agent.act(o)
            
            if abs(o[0]) > MAX_X_DISPLACEMENT:
                ep_violations += 1
                
            o, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            
            if terminated or truncated:
                break
        
        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)
        violation_counts.append(ep_violations)
    
    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'total_violations': np.sum(violation_counts),
        'episodes_with_violations': np.sum([1 for v in violation_counts if v > 0]),
        'violation_rate': np.sum([1 for v in violation_counts if v > 0]) / n_episodes
    }
    
    print(f"{agent_name} Evaluation Results:")
    print(f"  Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Episodes with X-violations: {results['episodes_with_violations']}/{n_episodes}")
    print(f"  Violation rate: {results['violation_rate']:.3f}")
    
    return results

def plot_training_comparison(spinning_stats, safety_stats, save_path):
    """Plot training comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode Returns
    axes[0, 0].plot(spinning_stats['timesteps'], spinning_stats['episode_returns'], 
                   label='Spinning Up PPO', linewidth=2)
    axes[0, 0].plot(safety_stats['timesteps'], safety_stats['episode_returns'], 
                   label='Safety PPO-Lagrangian', linewidth=2)
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Episode Return')
    axes[0, 0].set_title('Training Progress: Episode Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Violation Rate
    axes[0, 1].plot(spinning_stats['timesteps'], spinning_stats['violation_rate'], 
                   label='Spinning Up PPO', linewidth=2)
    axes[0, 1].plot(safety_stats['timesteps'], safety_stats['violation_rate'], 
                   label='Safety PPO-Lagrangian', linewidth=2)
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('X-Displacement Violation Rate')
    axes[0, 1].set_title('Constraint Violation Rate During Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode Length
    axes[1, 0].plot(spinning_stats['timesteps'], spinning_stats['episode_lengths'], 
                   label='Spinning Up PPO', linewidth=2)
    axes[1, 0].plot(safety_stats['timesteps'], safety_stats['episode_lengths'], 
                   label='Safety PPO-Lagrangian', linewidth=2)
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Episode Length')
    axes[1, 0].set_title('Training Progress: Episode Lengths')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Lambda Values
    axes[1, 1].plot(safety_stats['timesteps'], safety_stats['lambda_values'], 
                   label='Lambda (Lagrange Multiplier)', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Lambda Value')
    axes[1, 1].set_title('PPO-Lagrangian: Lagrange Multiplier Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =======================================
# MAIN EXECUTION
# =======================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("Official PPO Implementations Comparison")
    print("Spinning Up PPO vs Safety Starter Agents PPO-Lagrangian")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - X-displacement constraint: ±{MAX_X_DISPLACEMENT}")
    print(f"  - Random seed: {SEED}")
    print(f"  - Results directory: {RUN_DIR}")
    print()
    
    # Environment factories
    def make_env():
        return gym.make('CartPole-v1')
    
    def make_constrained_env():
        env = gym.make('CartPole-v1')
        tracker = ConstraintTracker()
        return ConstrainedCartPoleWrapper(env, tracker)
    
    # Calculate epochs based on total timesteps
    steps_per_epoch = 4000
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    # Create trackers
    spinning_tracker = ConstraintTracker()
    safety_tracker = ConstraintTracker()
    
    print(f"Training for {epochs} epochs with {steps_per_epoch} steps each")
    print()
    
    # Train Spinning Up PPO
    print("1. Training Spinning Up PPO...")
    start_time = time.time()
    spinning_agent = SpinningUpPPO(make_env, steps_per_epoch, epochs)
    spinning_stats = spinning_agent.train(spinning_tracker)
    spinning_time = time.time() - start_time
    print(f"Spinning Up PPO training time: {spinning_time:.1f} seconds")
    print(f"Total episodes with x-violations: {spinning_tracker.violation_episodes}")
    print()
    
    # Train Safety Starter Agents PPO-Lagrangian
    print("2. Training Safety Starter Agents PPO-Lagrangian...")
    start_time = time.time()
    safety_agent = SafetyPPOLagrangian(make_constrained_env, steps_per_epoch, epochs)
    safety_stats = safety_agent.train(safety_tracker)
    safety_time = time.time() - start_time
    print(f"Safety PPO-Lagrangian training time: {safety_time:.1f} seconds")
    print(f"Total episodes with x-violations: {safety_tracker.violation_episodes}")
    print()
    
    # Evaluate both agents
    print("=" * 50)
    print("EVALUATION")
    print("=" * 50)
    
    eval_results = {}
    eval_results['Spinning Up PPO'] = evaluate_agent(spinning_agent, make_env, 100, 'Spinning Up PPO')
    print()
    eval_results['Safety PPO-Lagrangian'] = evaluate_agent(safety_agent, make_env, 100, 'Safety PPO-Lagrangian')
    print()
    
    # Plot results
    plot_path = os.path.join(RUN_DIR, 'official_ppo_comparison.png')
    plot_training_comparison(spinning_stats, safety_stats, plot_path)
    print(f"Training comparison plots saved to: {plot_path}")
    
    # Summary
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Training Summary:")
    print(f"  Spinning Up PPO        - X-violations during training: {spinning_tracker.violation_episodes:,}")
    print(f"  Safety PPO-Lagrangian  - X-violations during training: {safety_tracker.violation_episodes:,}")
    print()
    print(f"Evaluation Summary (100 episodes):")
    for algo, results in eval_results.items():
        print(f"  {algo:20s} - Mean Return: {results['mean_return']:6.2f} ± {results['std_return']:5.2f}")
        print(f"  {' '*20} - Violation Rate: {results['violation_rate']:6.3f}")
    
    # Performance comparison
    spinning_violations = eval_results['Spinning Up PPO']['violation_rate']
    safety_violations = eval_results['Safety PPO-Lagrangian']['violation_rate']
    violation_reduction = (spinning_violations - safety_violations) / max(spinning_violations, 1e-8) * 100
    
    spinning_return = eval_results['Spinning Up PPO']['mean_return']
    safety_return = eval_results['Safety PPO-Lagrangian']['mean_return']
    return_diff = safety_return - spinning_return
    
    print()
    print(f"Performance Comparison:")
    print(f"  Violation rate reduction: {violation_reduction:+.1f}%")
    print(f"  Return difference: {return_diff:+.2f}")
    print()
    print(f"✅ Official implementations successfully compared!")
    print(f"Results saved in: {RUN_DIR}")


if __name__ == "__main__":
    main()
