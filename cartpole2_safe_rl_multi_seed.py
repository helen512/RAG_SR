import os
# Set environment variable to use legacy Keras (Keras 2) instead of Keras 3
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")
# Import TensorFlow to reset graph between runs
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Import from safety-starter-agents
sys.path.append('/home/dmy/gymtest/safety-starter-agents')
from safe_rl import ppo, ppo_lagrangian, cpo

from dataclasses import dataclass
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF
from scipy.optimize import minimize

# Configuration
BASE_SEED = 42
NUM_SEEDS = 10
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]
STEPS_PER_EPOCH = 4000
TOTAL_TIMESTEPS = STEPS_PER_EPOCH * 60
TIMESTEP_INTERVAL = 5000  # For interpolation grid
MAX_X_DISPLACEMENT = 1  # Constraint threshold
UPDATE_CORRECTION_ACTION = True
RUN_DIR = "runs_cartpole2_safe_rl_multi_seed"
os.makedirs(RUN_DIR, exist_ok=True)
save_index = "experiment2"

def log_barrier_quad(x, x_max, mu=1.0, eps=1e-12):
    z = (x / x_max)**2
    z = min(z, 1 - eps)   
    return -mu * np.log(1 - z)

def log_barrier_linear(x, x_max, mu=1.0, eps=1e-12):
    z_right = np.maximum(x_max - x, eps)  # add small eps to avoid log(0)
    z_left  = np.maximum(x_max + x, eps)
    return mu * (np.log(z_right)+ np.log(z_left))


class ConstraintViolationCounter:
    """Counter for tracking constraint violations and episode data"""

    def __init__(self, x_threshold: float = MAX_X_DISPLACEMENT):
        self.x_threshold = x_threshold
        self.total_episodes = 0  # total number of episodes
        self.total_timesteps = 0  # total number of timesteps
        
        # For timestep-based tracking
        self.returns = []  # episode returns
        self.timesteps = []  # timesteps when episodes ended
        self.episode_lengths = []  # length of each episode
        self.violated_episodes = 0  # number of episodes with violations
        
        # For epoch-based tracking (legacy)
        self.violations_per_epoch = []  # list of violations per epoch
        self.episodes_per_epoch = []  # list of episodes per epoch
        self.current_epoch_violations = 0 # number of violations in the current epoch
        self.current_epoch_episodes = 0 # number of episodes in the current epoch
        
        # Current episode tracking
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        
    def check_violation(self, obs) -> bool:
        """Check if current observation violates constraint"""
        x_pos = obs[0] if isinstance(obs, np.ndarray) else obs
        return abs(x_pos) > self.x_threshold
    
    def compute_cost(self, obs, info) -> float:
        """Compute smooth cost signal for constrained RL algorithms"""
        x_pos = abs(obs[0] if isinstance(obs, np.ndarray) else obs)
        current_timestep = info['episode_timestep']
        return log_barrier_quad(x_pos, self.x_threshold)/(current_timestep/100)
    
    def check_step_violation(self, obs) -> bool:
        """Record a timestep and return if violated"""
        self.total_timesteps += 1
        violated = self.check_violation(obs)
        return violated
    
    def step_reward(self, reward: float):
        """Record reward from a step"""
        self.current_episode_return += reward
        self.current_episode_length += 1

    def episode_ended(self, had_violation: bool):
        """Record episode completion"""
        self.total_episodes += 1
        self.current_epoch_episodes += 1
        
        # Record episode data for timestep-based tracking
        self.returns.append(self.current_episode_return)
        self.timesteps.append(self.total_timesteps)
        self.episode_lengths.append(self.current_episode_length)
        
        if had_violation:
            self.current_epoch_violations += 1
            self.violated_episodes += 1
        
        # Reset current episode tracking
        self.current_episode_return = 0.0
        self.current_episode_length = 0
    
    def epoch_ended(self):
        """Record epoch completion and reset current epoch counters"""
        self.violations_per_epoch.append(self.current_epoch_violations)
        self.episodes_per_epoch.append(self.current_epoch_episodes)
        self.current_epoch_violations = 0
        self.current_epoch_episodes = 0

    def reset(self):
        """Reset all counters"""
        self.total_episodes = 0
        self.total_timesteps = 0
        self.violations_per_epoch = []
        self.episodes_per_epoch = []
        self.current_epoch_violations = 0
        self.current_epoch_episodes = 0
        
        # Reset timestep-based tracking
        self.returns = []
        self.timesteps = []
        self.episode_lengths = []
        self.violated_episodes = 0
        self.current_episode_return = 0.0
        self.current_episode_length = 0

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_episodes': self.total_episodes,
            'violation_episodes': self.violated_episodes,
            'violation_rate': self.violated_episodes / self.total_episodes if self.total_episodes > 0 else 0.0,
            'total_timesteps': self.total_timesteps,
            'returns': self.returns,
            'timesteps': self.timesteps,
            'episode_lengths': self.episode_lengths,
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
        if self.counter.total_timesteps > 0:
            self.counter.episode_ended(self.episode_had_violation)
        
        self.episode_had_violation = False
        self.episode_timestep = 0  # Reset episode timestep counter

        # Handle both gym and gymnasium APIs
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            # Gymnasium API: returns (observation, info)
            return result[0]
        else:
            # Old gym API: returns observation
            return result

    def step(self, action):
        # Increment episode timestep
        self.episode_timestep += 1
        # Ensure action matches the underlying environment's expectation
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            if isinstance(action, np.ndarray):
                action = int(np.asarray(action).item())
            else:
                action = int(action)
        else:
            action = np.asarray(action)
            target_shape = self.env.action_space.shape
            flat_size = int(np.prod(target_shape))
            if action.size != flat_size:
                action = action.reshape(-1)
            if action.size == flat_size:
                action = action.reshape(target_shape)
            else:
                raise ValueError(
                    f"Action size {action.size} cannot be reshaped to expected shape {target_shape}"
                )
            if self.env.action_space.dtype is not None:
                action = action.astype(self.env.action_space.dtype, copy=False)
        # Handle both gym and gymnasium APIs
        result = self.env.step(action)
        if len(result) == 5:
            # Gymnasium API: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            
            done = terminated or truncated
        else:
            # Old gym API: (obs, reward, done, info)
            obs, reward, done, info = result
        
        # Add episode timestep to info dictionary
        info['episode_timestep'] = self.episode_timestep
        
        # Track reward for timestep-based analysis
        self.counter.step_reward(reward)
        
        # Check for violation
        violated = self.counter.check_step_violation(obs)
        if violated:
            self.episode_had_violation = True
            done = True  # let the episode end with x displacement violation
        
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
# CBF SAFETY FILTER IMPLEMENTATION
# =======================================

class CBFSafetyFilter:
    """
    Control Barrier Function (CBF) Safety Filter for CartPole
    
    Implements a simplified CBF that uses symbolic dynamics: x = x_dot * timestep
    The barrier function is an ellipsoid: h(x) = 1 - (x/x_max)² - (theta/theta_max)²
    """
    
    def __init__(self, x_max: float = 0.9, theta_max: float = 0.2, dt: float = 0.02, alpha: float = 1):
        """
        Initialize CBF Safety Filter
        
        Args:
            x_max: Maximum allowed cart position
            theta_max: Maximum allowed pole angle  
            dt: Timestep for symbolic model
            alpha: CBF class-K function slope
        """
        self.x_max = x_max
        self.theta_max = theta_max
        self.dt = dt
        self.alpha = alpha
        self.grid_search_interval = 0.2
        
        # Statistics tracking
        self.total_actions = 0
        self.corrected_actions = 0
        self.correction_magnitudes = []
        
        print(f"CBF Safety Filter initialized:")
        print(f"  x_max: {x_max}, theta_max: {theta_max}")
        print(f"  timestep: {dt}, alpha: {alpha}")
    
    def barrier_function(self, x: float, theta: float) -> float:
        """
        Ellipsoid barrier function: h(x) = 1 - (x/x_max)² 
        
        Safe set: {(x, theta)}
        """
        return 1.0 - (x / self.x_max)**2 
    
    def barrier_derivative(self, state: np.ndarray, action: float) -> float:
       
        x, theta, x_dot, theta_dot = state
        
        # CartPole physical parameters (matching gym defaults)
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5  # Half-pole length
        polemass_length = masspole * length
        force_mag = action   # Use actual force magnitude
        time_step = 0.02
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (
            force_mag + polemass_length * theta_dot**2 * sintheta
        ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        
        # Next velocities and positions using Euler integration
        x_dot_next = x_dot + xacc * time_step
        theta_dot_next = theta_dot + thetaacc * time_step
        
        # Next positions using x = x_dot * timestep (as requested)
        x_next = x + x_dot_next * time_step
        theta_next = theta + theta_dot_next * time_step
        
        # Barrier derivative: dh/dt ≈ (h(x_next) - h(x)) / dt
        h_current = self.barrier_function(x, theta)
        h_next = self.barrier_function(x_next, theta_next)
        
        h_dot = (h_next - h_current) / time_step
        return h_dot
        
    
    
    def certify_action(self, state: np.ndarray, uncertified_action: float) -> Tuple[float, bool]:
        
        self.total_actions += 1
        
        x, theta, x_dot, theta_dot = state
        h = self.barrier_function(x, theta)
        
        # Always check CBF constraint - no early return
        # Check if uncertified action satisfies CBF constraint
        h_dot = self.barrier_derivative(state, uncertified_action)
        cbf_constraint = h_dot + self.alpha * h
        
        if cbf_constraint >= 1e-6:  # Tighter tolerance for safety
            return uncertified_action, False
        
        # Action needs correction - solve QP
        try:
            # Use scipy minimize for simple 1D QP
            def objective(u):
                return 0.5 * (u[0] - uncertified_action)**2
            
            def constraint(u):
                h_dot_u = self.barrier_derivative(state, u[0])
                return h_dot_u + self.alpha * h  # >= 0
            
            constraints = [
                {'type': 'ineq', 'fun': constraint},
                {'type': 'ineq', 'fun': lambda u: u[0] + 3},  # u >= -3
                {'type': 'ineq', 'fun': lambda u: 3 - u[0]}   # u <= 3
            ]
            
            result = minimize(
                objective,
                x0=[uncertified_action],
                method='SLSQP',
                constraints=constraints,
                options={'ftol': 1e-6, 'disp': False}
            )
            
            if result.success:
                certified_action = np.clip(result.x[0], -3.0, 3.0)
                correction_magnitude = abs(certified_action - uncertified_action)
                
                self.corrected_actions += 1
                self.correction_magnitudes.append(correction_magnitude)
                
                return certified_action, True
            else:
                # Fallback: find safe action by brute force
                print(f"QP failed, using grid search")
                return self.grid_search_action(state, uncertified_action)
                
        except Exception as e:
            print(f"CBF optimization failed: {e}")
            return uncertified_action, False

    def grid_search_action(self, state: np.ndarray, uncertified_action: float) -> Tuple[float, bool]:
        # Try smaller intervals first, then larger ones
        for interval in [0.1, 0.2, 0.5, 1.0]:
            for direction in [1, -1]:
                for count in range(1, 51):  # More iterations
                    test_action = uncertified_action + direction * interval * count
                    test_action = np.clip(test_action, -3.0, 3.0)
                    
                    h = self.barrier_function(state[0], state[1])
                    h_dot = self.barrier_derivative(state, test_action)
                    if h_dot + self.alpha * h >= -1e-6:
                        return test_action, True
        sign = np.sign(state[0])
        return sign * 3, False

    def get_stats(self) -> Dict:
        """Get CBF statistics"""
        if self.total_actions == 0:
            return {'correction_rate': 0.0, 'avg_correction': 0.0}
        
        return {
            'total_actions': self.total_actions,
            'corrected_actions': self.corrected_actions,
            'correction_rate': self.corrected_actions / self.total_actions,
            'avg_correction': np.mean(self.correction_magnitudes) if self.correction_magnitudes else 0.0,
            'max_correction': np.max(self.correction_magnitudes) if self.correction_magnitudes else 0.0
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.total_actions = 0
        self.corrected_actions = 0
        self.correction_magnitudes = []



class CBFWrapper(gym.Wrapper):
    """
    Wrapper that applies CBF safety filter to actions
    """
    
    def __init__(self, env, cbf_filter: CBFSafetyFilter, counter: ConstraintViolationCounter, 
                 use_corrected_action_for_training: bool = False, steps_per_epoch: int = STEPS_PER_EPOCH):
        super().__init__(env)
        self.cbf_filter = cbf_filter
        self.counter = counter
        self.episode_had_violation = False
        self.steps_per_epoch = steps_per_epoch
        self.epoch_timesteps = 0
        self.episode_timestep = 0
        self.last_obs = None
        self.use_corrected_action_for_training = use_corrected_action_for_training
        
    def reset(self, **kwargs):
        # Record previous episode
        if self.counter.total_timesteps > 0:
            self.counter.episode_ended(self.episode_had_violation)
        
        self.episode_had_violation = False
        self.episode_timestep = 0
        
        # Handle both gym and gymnasium APIs
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            # Gymnasium API: returns (observation, info)
            obs = result[0]
        else:
            # Old gym API: returns observation
            obs = result
        
        self.last_obs = obs  # Store observation for CBF
        return obs
        
    def step(self, action):
        self.episode_timestep += 1
        
        # Store original uncertified action for info dict
        uncertified_action = action if isinstance(action, np.ndarray) else np.array([action])
        
        # Convert action from array to scalar if needed for continuous action spaces
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action = float(action.item())
            else:
                action = float(action[0])
        
        # Apply CBF safety filter using stored observation
        certified_action = action
        was_corrected = False
        if self.last_obs is not None and len(self.last_obs) >= 4:
            current_state = self.last_obs
            
            # Certify continuous action with CBF (InvertedPendulum-v4 uses continuous actions)
            certified_action, was_corrected = self.cbf_filter.certify_action(current_state, action)
            
            # Debug: print when action is corrected
            if was_corrected:
                print(f"CBF corrected action at timestep {self.counter.total_timesteps}: {action:.3f} -> {certified_action:.3f}")
        
        # Ensure action is properly shaped for InvertedPendulum-v4 (expects 1D array)
        if not isinstance(certified_action, np.ndarray):
            certified_action = np.array([certified_action], dtype=np.float32)
        
        # Handle both gym and gymnasium APIs
        result = self.env.step(certified_action)
        if len(result) == 5:
            # Gymnasium API: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            # Old gym API: (obs, reward, done, info)
            obs, reward, done, info = result
        
        self.last_obs = obs  # Update last observation
        
        # Add episode timestep to info
        info['episode_timestep'] = self.episode_timestep
        
        # Track reward for timestep-based analysis
        self.counter.step_reward(reward)
        
        # Check for violation
        violated = self.counter.check_step_violation(obs)
        if violated:
            self.episode_had_violation = True
            done = True  # Terminate episode immediately on violation
            # Debug: show violation details
            x, x_dot, theta, theta_dot = obs
            h_value = self.cbf_filter.barrier_function(x, theta)
            print(f"CBF VIOLATION at timestep {self.counter.total_timesteps}: x={x:.3f} (limit=±{self.cbf_filter.x_max}), θ={theta:.3f}")
            print(f"  Barrier value h(x,θ) = {h_value:.6f} (should be ≥ 0)")
            print(f"  Last action was certified: {certified_action}")
            print(f"  {'='*50}")
        
        # Track epoch boundaries
        self.epoch_timesteps += 1
        if self.epoch_timesteps % self.steps_per_epoch == 0:
            self.counter.epoch_ended()
            print(f"Epoch ended at timestep {self.counter.total_timesteps} "
                  f"({self.counter.current_epoch_violations} violations this epoch)")
        
        # Add CBF-related information to info dict
        info.update({
            'cbf_corrected': was_corrected,
            'uncertified_action': uncertified_action,
            'certified_action': certified_action,
            'constraint_violated': violated,
        })
        
        # Add cost information
        cost = self.counter.compute_cost(obs, info)
        info['cost'] = cost
        
        # Add training action info if requested (for corrected action training)
        if self.use_corrected_action_for_training:
            info['training_action'] = certified_action
        
        return obs, reward, done, info


# =======================================
# TRAINING FUNCTIONS
# =======================================

def train_ppo(counter: ConstraintViolationCounter, seed: int):
    """Train standard PPO"""
    print("\n" + "=" * 50)
    print(f"Training Standard PPO (seed {seed})")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    # Create a shared counter that persists across environment resets
    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)
    
    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'ppo_{save_index}_seed_{seed}'),
        'exp_name': f'ppo_cartpole_seed_{seed}'
    }
    
    start_time = time.time()
    
    # Train PPO with optimized hyperparameters
    # Note: PPO in safe_rl uses different parameter names than other implementations
    ppo(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        seed=seed,
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
    
    return summary


def train_ppo_lagrangian(counter: ConstraintViolationCounter, seed: int):
    """Train PPO-Lagrangian"""
    print("\n" + "=" * 50)
    print(f"Training PPO-Lagrangian (seed {seed})")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)
    
    # Cost limit: tighter constraint for better safety  
    cost_lim = 2.0  
    
    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'ppo_lagrangian_{save_index}_seed_{seed}'),
        'exp_name': f'ppo_lagrangian_cartpole_seed_{seed}'
    }
    
    start_time = time.time()
    
    # Train PPO-Lagrangian with optimized hyperparameters
    ppo_lagrangian(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),  # Keep same for CartPole
        seed=seed,
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

    return summary


def train_cpo(counter: ConstraintViolationCounter, seed: int):
    """Train Constrained Policy Optimization (CPO)"""
    print("\n" + "=" * 50)
    print(f"Training CPO (Constrained Policy Optimization) (seed {seed})")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)
    
    # Cost limit: stricter for CPO to enforce better constraint satisfaction
    cost_lim = 0.5  
    
    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'cpo_{save_index}_seed_{seed}'),
        'exp_name': f'cpo_cartpole_seed_{seed}'
    }
    
    start_time = time.time()
    
    # Train CPO with optimized hyperparameters
    # Note: CPO in safe_rl uses same interface as PPO/PPO-Lagrangian
    cpo(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),  # Keep same for CartPole
        seed=seed,
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
    
    return summary


def train_ppo_with_cbf(counter: ConstraintViolationCounter, seed: int):
    """Train PPO with CBF Safety Filter"""
    print("\n" + "=" * 50)
    print(f"Training PPO with CBF Safety Filter (seed {seed})")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    # Create shared CBF instance to accumulate statistics across all env instances
    shared_cbf = InvertedPendulumCBF(x_max=MAX_X_DISPLACEMENT*0.9)
    
    cbf_filter = CBFSafetyFilter(
        x_max=MAX_X_DISPLACEMENT*0.9,
        theta_max=0.2,  # 0.2 radians ~ 11.5 degrees
        dt=0.02,        # 20ms timestep
        alpha=1      # CBF slope parameter
    )
    
    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return CBFWrapper(env, cbf_filter, counter, 
                         use_corrected_action_for_training=UPDATE_CORRECTION_ACTION, 
                         steps_per_epoch=steps_per_epoch)
    
    
    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'ppo_cbf_{save_index}_seed_{seed}'),
        'exp_name': f'ppo_cbf_cartpole_seed_{seed}'
    }
    
    start_time = time.time()
    
    # Train PPO with CBF-wrapped environment
    ppo(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=0.99,
        lam=0.95,
        target_kl=0.01,
        vf_lr=3e-4,
        vf_iters=80,
        logger_kwargs=logger_kwargs,
        save_freq=10
    )
    
    training_time = time.time() - start_time
    
    # Get CBF statistics from shared instance
    cbf_stats = shared_cbf.get_stats()

    
    print(f"\nPPO+CBF Training Complete!")
    print(f"  Time: {training_time:.1f} seconds")
    print(f"  CBF Statistics:")
    print(f"    - Total actions processed: {cbf_stats['total_actions']}")
    print(f"    - Actions corrected: {cbf_stats['corrected_actions']}")
    print(f"    - Correction rate: {cbf_stats['correction_rate']:.3f}")
    print(f"    - Average correction magnitude: {cbf_stats['avg_correction']:.4f}")
    print(f"    - Maximum correction magnitude: {cbf_stats['max_correction']:.4f}")
    print(f"  Training violation summary:")
    summary = counter.get_summary()
    print(f"    - Total episodes: {summary['total_episodes']}")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}")
    # Add CBF statistics to summary
    summary.update(cbf_stats)
    
    return summary







# =======================================
# =======================================
# EVALUATION FUNCTION
# =======================================

def evaluate_trained_policy(policy_path: str, n_episodes: int = 50) -> Dict:
    """
    Evaluate a trained policy and count constraint violations.
    
    Note: This is a placeholder since loading policies from safe_rl requires
    their specific format. In practice, you'd use their test_policy.py script.
    """
    print(f"\nEvaluating policy from: {policy_path}")
    print(f"  Running {n_episodes} episodes...")
    
    counter = ConstraintViolationCounter()
    env = ConstrainedCartPoleWrapper(gym.make('InvertedPendulum-v4'), counter, steps_per_epoch=STEPS_PER_EPOCH)
    
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
    
    return summary


# =======================================
# AGGREGATION FUNCTIONS
# =======================================

def _interpolate_episode_returns(counter: ConstraintViolationCounter, grid: np.ndarray) -> np.ndarray:
    """Interpolate episode returns over a regular timestep grid"""
    if len(counter.timesteps) == 0 or len(counter.returns) == 0:
        raise ValueError("Timesteps or returns are empty")
        
    timesteps = np.asarray(counter.timesteps, dtype=np.float64)
    returns = np.asarray(counter.returns, dtype=np.float64)

    order = np.argsort(timesteps)
    timesteps = timesteps[order]
    returns = returns[order]

    unique_timesteps, unique_indices = np.unique(timesteps, return_index=True)
    unique_returns = returns[unique_indices]

    return np.interp(
        grid,
        unique_timesteps,
        unique_returns,
        left=unique_returns[0],
        right=unique_returns[-1],
    )


def aggregate_returns_by_timestep(counters_dict: Dict[str, List[ConstraintViolationCounter]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate returns across multiple seeds for each algorithm"""
    grid = np.arange(0, TOTAL_TIMESTEPS + TIMESTEP_INTERVAL, TIMESTEP_INTERVAL, dtype=np.float64)
    aggregated: Dict[str, Dict[str, np.ndarray]] = {}

    for alg_name, alg_counters in counters_dict.items():
        if len(alg_counters) == 0:
            continue
        interpolated_curves = []
        for counter in alg_counters:
            try:
                interpolated_curves.append(_interpolate_episode_returns(counter, grid))
            except ValueError as e:
                print(f"Warning: Skipping counter for {alg_name}: {e}")
                continue

        if len(interpolated_curves) == 0:
            continue
            
        curves = np.stack(interpolated_curves, axis=0)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)

        aggregated[alg_name] = {
            'timesteps': grid,
            'mean_return': mean_curve,
            'std_return': std_curve,
        }

    return aggregated


# =======================================
# VISUALIZATION
# =======================================

def plot_training_comparison(aggregated_data: Dict[str, Dict[str, np.ndarray]], save_path: str):
    """Plot single timestep-based comparison with std shading"""
    print("\nGenerating comparison plot...")
    
    try:
        plt.figure(figsize=(12, 8))
        
        colors = {'PPO': 'blue', 'PPO-Lagrangian': 'green', 'CPO': 'red', 'PPO+CBF': 'purple'}
        
        for alg_name, stats in aggregated_data.items():
            timesteps = stats['timesteps']
            mean = stats['mean_return']
            std = stats['std_return']
            
            color = colors.get(alg_name, 'black')
            
            # Plot mean line
            plt.plot(timesteps, mean, label=alg_name, linewidth=2, color=color, alpha=0.8)
            
            # Plot std shading
            plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)
        
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Average Episode Return', fontsize=12)
        plt.title(f'Safe RL Comparison: Average Return vs Timesteps over {NUM_SEEDS} Seeds', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"  Warning: Could not generate plot: {e}")
        import traceback
        traceback.print_exc()


# =======================================
# MAIN EXECUTION
# =======================================

def print_multi_seed_summary(counters_dict: Dict[str, List[ConstraintViolationCounter]]):
    """Print summary statistics across multiple seeds"""
    print("\n" + "=" * 80)
    print("MULTI-SEED TRAINING SUMMARY")
    print("=" * 80)
    
    print("Configuration:")
    print(f"  - Total training timesteps per seed: {TOTAL_TIMESTEPS:,}")
    print(f"  - Number of seeds: {NUM_SEEDS}")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Results directory: {RUN_DIR}")
    
    print(f"\nViolation Statistics (over {NUM_SEEDS} seeds):")
    for alg_name, counters in counters_dict.items():
        total_episodes_per_seed = [counter.total_episodes for counter in counters]
        violated_episodes_per_seed = [counter.violated_episodes for counter in counters]
        violation_rates_per_seed = [v/t if t > 0 else 0.0 for v, t in zip(violated_episodes_per_seed, total_episodes_per_seed)]
        
        mean_total_episodes = np.mean(total_episodes_per_seed)
        std_total_episodes = np.std(total_episodes_per_seed)
        mean_violated_episodes = np.mean(violated_episodes_per_seed)
        std_violated_episodes = np.std(violated_episodes_per_seed)
        mean_violation_rate = np.mean(violation_rates_per_seed)
        std_violation_rate = np.std(violation_rates_per_seed)
        
        print(f"  - {alg_name:15s}:")
        print(f"    Total episodes:    {mean_total_episodes:8.2f} ± {std_total_episodes:6.2f}")
        print(f"    Violated episodes: {mean_violated_episodes:8.2f} ± {std_violated_episodes:6.2f}")
        print(f"    Violation rate:    {mean_violation_rate:8.4f} ± {std_violation_rate:6.4f}")


def main():
    """Main execution function"""
    
    # Store counters for each algorithm across seeds
    counters_dict = {
        'PPO': [],
        'PPO-Lagrangian': [],
        'CPO': [],
        'PPO+CBF': []
    }
    
    # Train all algorithms across all seeds
    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"RUNNING SEED {seed} ({SEEDS.index(seed) + 1}/{NUM_SEEDS})")
        print(f"{'='*80}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Train PPO
        ppo_counter = ConstraintViolationCounter()
        train_ppo(ppo_counter, seed)
        counters_dict['PPO'].append(ppo_counter)
        
        # Reset TensorFlow graph between training runs (required for TF 1.x)
        tf.reset_default_graph()
        print("\n[TensorFlow graph reset]\n")
        
        # Train PPO-Lagrangian
        ppo_lag_counter = ConstraintViolationCounter()
        train_ppo_lagrangian(ppo_lag_counter, seed)
        counters_dict['PPO-Lagrangian'].append(ppo_lag_counter)
        
        # Reset TensorFlow graph between training runs (required for TF 1.x)
        tf.reset_default_graph()
        print("\n[TensorFlow graph reset]\n")
        
        # Train CPO
        cpo_counter = ConstraintViolationCounter()
        train_cpo(cpo_counter, seed)
        counters_dict['CPO'].append(cpo_counter)
        
        # Reset TensorFlow graph between training runs (required for TF 1.x)
        tf.reset_default_graph()
        print("\n[TensorFlow graph reset]\n")
        
        # Train PPO with CBF
        cbf_counter = ConstraintViolationCounter()
        train_ppo_with_cbf(cbf_counter, seed)
        counters_dict['PPO+CBF'].append(cbf_counter)
        
        # Reset TensorFlow graph between training runs (required for TF 1.x)
        tf.reset_default_graph()
        print("\n[TensorFlow graph reset]\n")
    
    # Aggregate results across seeds
    print("\nAggregating results across seeds...")
    aggregated_data = aggregate_returns_by_timestep(counters_dict)
    
    # Generate timestep-based plot
    plot_path = os.path.join(RUN_DIR, f'average_return_vs_timesteps_{save_index}.png')
    plot_training_comparison(aggregated_data, plot_path)
    
    # Print summary statistics
    print_multi_seed_summary(counters_dict)
    
    # Save aggregated data to CSV
    print("\nSaving aggregated data...")
    aggregated_rows = []
    for alg_name, stats in aggregated_data.items():
        for timestep, mean, std in zip(stats['timesteps'], stats['mean_return'], stats['std_return']):
            aggregated_rows.append({
                'algorithm': alg_name,
                'timesteps': timestep,
                'mean_return': mean,
                'std_return': std,
            })
    
    aggregated_df = pd.DataFrame(aggregated_rows)
    aggregated_csv_path = os.path.join(RUN_DIR, f'aggregated_returns_{save_index}.csv')
    aggregated_df.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated data saved to: {aggregated_csv_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"Results saved in: {RUN_DIR}")
    print(f"  - Timestep-based plot: {plot_path}")
    print(f"  - Aggregated data CSV: {aggregated_csv_path}")
    print(f"  - Individual seed logs in subdirectories")
    

if __name__ == "__main__":
    main()


