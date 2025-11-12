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
from safe_rl import ppo

from dataclasses import dataclass
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF
from scipy.optimize import minimize
import cvxpy as cp

# Configuration
BASE_SEED = 42
NUM_SEEDS = 1
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]
STEPS_PER_EPOCH = 4000
TOTAL_TIMESTEPS = STEPS_PER_EPOCH * 30
TIMESTEP_INTERVAL = 5000  # For interpolation grid
MAX_X_DISPLACEMENT = 1  # Constraint threshold
UPDATE_CORRECTION_ACTION = True
RUN_DIR = "runs_cartpole2_safe_rl_multi_seed"
os.makedirs(RUN_DIR, exist_ok=True)
save_index = "experiment4"

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
        
        # CBF statistics tracking
        self.cbf_total_actions = 0
        self.cbf_corrected_actions = 0
        self.cbf_correction_magnitudes = []
        
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

    def record_cbf_action(self, was_corrected: bool, correction_magnitude: float = 0.0):
        """Record a CBF action (corrected or not)"""
        self.cbf_total_actions += 1
        if was_corrected:
            self.cbf_corrected_actions += 1
            if correction_magnitude > 0:
                self.cbf_correction_magnitudes.append(correction_magnitude)
    
    def get_cbf_stats(self) -> Dict:
        """Get CBF statistics"""
        if self.cbf_total_actions == 0:
            return {
                'total_actions': 0,
                'corrected_actions': 0,
                'correction_rate': 0.0,
                'avg_correction': 0.0,
                'max_correction': 0.0
            }
        
        return {
            'total_actions': self.cbf_total_actions,
            'corrected_actions': self.cbf_corrected_actions,
            'correction_rate': self.cbf_corrected_actions / self.cbf_total_actions,
            'avg_correction': np.mean(self.cbf_correction_magnitudes) if self.cbf_correction_magnitudes else 0.0,
            'max_correction': np.max(self.cbf_correction_magnitudes) if self.cbf_correction_magnitudes else 0.0
        }
    
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
        
        # Reset CBF statistics
        self.cbf_total_actions = 0
        self.cbf_corrected_actions = 0
        self.cbf_correction_magnitudes = []

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
        
        # Add cost information for logging or constrained algorithms
        cost = self.counter.compute_cost(obs, info)
        info['cost'] = cost
        
        return obs, reward, done, info



# =======================================
# CBF SAFETY FILTER IMPLEMENTATION
# =======================================


# ----- same helper as you have -----
def cartpole_f_g(x):
    # Use exact parameters from find_param.py
    g = 9.81
    mc, mp, l = 10.472, 5.019, 0.3  # half-pole length
    total_mass = mc + mp
    mp_l = mp * l
    gear = 100.0
    pos, th, vel, thd = x

    def accel(u):
        s, c = np.sin(th), np.cos(th)
        temp = (u*gear + mp_l * thd * thd * s) / total_mass
        thdd = (g * s - c * temp) / (l * (4.0/3.0 - mp * c * c / total_mass))
        xdd  = temp - mp_l * thdd * c / total_mass
        return xdd, thdd

    xdd0, thdd0 = accel(0.0)
    xdd1, thdd1 = accel(1.0)
    f = np.array([vel, thd, xdd0, thdd0])
    g = np.array([0.0, 0.0, xdd1 - xdd0, thdd1 - thdd0])  # g2 = g[1]
    return f, g

# ----- build A,b for ONE wall chosen by sign(x) -----
def hocbf_A_b_one_wall(x, x_max, c1=0.05, c2=0.5):
    """
    Returns A (1,1), b (1,) using the active wall:
      if x >= 0 -> right wall (x <= x_max)
      if x <  0 -> left wall  (-x <= x_max)
    """
    f, g = cartpole_f_g(x)
    pos, vel = x[0], x[2]
    f2, g2 = f[2], g[2]

    if pos >= 0.0:
        # right wall: h+ = x_max - x
        h   = x_max - pos
        hdot = -vel
        A = np.array([[-g2]])                                 # (1,1)
        b = np.array([f2 - c2*(hdot + c1*h)], dtype=float)    # (1,)
    else:
        # left wall: h- = x_max + x
        h   = x_max + pos
        hdot = +vel
        A = np.array([[+g2]])                                 # (1,1)
        b = np.array([-f2 - c2*(hdot + c1*h)], dtype=float)   # (1,)
    return A, b

class CartPoleCBF_QP_HOCBF:
    def __init__(self, x_max=0.95, u_min=-3.0, u_max=3.0, W=1.0, lam=1e3,
                 c1=0.05, c2=0.5, tolerance=2.0):
        self.x_max = x_max
        self.u_min, self.u_max = u_min, u_max
        self.W = np.array([[W]]) if np.ndim(W) == 0 else np.array(W)
        self.lam = lam
        self.c1, self.c2 = c1, c2
        self.tolerance = tolerance

        self.u = cp.Variable(1)
        self.delta = cp.Variable(1, nonneg=True)   # single slack now
        self.Ap = cp.Parameter((1,1))
        self.bp = cp.Parameter(1)
        self.u_des = cp.Parameter(1)

        obj = 0.5 * cp.quad_form(self.u - self.u_des, self.W) + self.lam * cp.sum(self.delta)
        cons = [ self.Ap @ self.u + self.delta >= self.bp,
                 self.u >= self.u_min, self.u <= self.u_max ]
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def step(self, x, u_nom):
        A, b = hocbf_A_b_one_wall(x, self.x_max, self.c1, self.c2)
        
        # Check if nominal action already satisfies constraint with tolerance
        constraint_value = float(A @ np.array([u_nom]) - b)
        
        # Debug: print constraint details occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1 == 0:  # Print every 1000 steps
            pos = x[0]
            h_value = self.x_max - abs(pos)
            #print(f"DEBUG CBF: pos={pos:.3f}, h={h_value:.3f}, u_nom={u_nom:.3f}, A={A[0,0]:.6f}, b={b[0]:.6f}, constraint={constraint_value:.6f}, tolerance={self.tolerance}")
        
        if constraint_value >= -self.tolerance:
            # Nominal action is safe enough, no correction needed
            return u_nom, True
        
        # Need to solve QP for safety
        self.Ap.value = A
        self.bp.value = b
        self.u_des.value = np.array([u_nom], dtype=float)

        self.prob.solve(solver=cp.OSQP, verbose=False)
        ok = self.prob.status in ("optimal", "optimal_inaccurate")
        u_star = float(self.u.value.item()) if ok and self.u.value is not None \
                 else float(np.clip(u_nom, self.u_min, self.u_max))
        return u_star, ok





class CBFWrapper(gym.Wrapper):
    """
    Wrapper that applies CBF safety filter to actions using CartPoleCBF_QP_HOCBF
    """
    
    def __init__(self, env, counter: ConstraintViolationCounter, 
                 x_max: float = 0.95, u_min: float = -3.0, u_max: float = 3.0, 
                 W: float = 1.0, lam: float = 1e3, c1: float = 0.5, c2: float = 0.5,
                 tolerance: float = 2.0, use_corrected_action_for_training: bool = False,
                 steps_per_epoch: int = STEPS_PER_EPOCH):
        super().__init__(env)
        self.cbf_filter = CartPoleCBF_QP_HOCBF(x_max=x_max, u_min=u_min, u_max=u_max, 
                                               W=W, lam=lam, c1=c1, c2=c2, tolerance=tolerance)
        self.counter = counter
        self.episode_had_violation = False
        self.steps_per_epoch = steps_per_epoch
        self.epoch_timesteps = 0
        self.episode_timestep = 0
        self.last_obs = None
        self.use_corrected_action_for_training = use_corrected_action_for_training
        self.x_max = x_max
        
        print(f"CBF Wrapper initialized with HOCBF:")
        print(f"  x_max: {x_max}, u_range: [{u_min}, {u_max}]")
        print(f"  c1: {c1}, c2: {c2}, W: {W}, lambda: {lam}, tolerance: {tolerance}")
        
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
        correction_magnitude = 0.0
        
        # Always count the action, even if CBF can't be applied
        if self.last_obs is not None and len(self.last_obs) >= 4:
            current_state = self.last_obs
            
            # Use HOCBF to certify action
            certified_action, qp_success = self.cbf_filter.step(current_state, action)
            
            # Check if action was corrected
            was_corrected = abs(certified_action - action) > 1e-6
            if was_corrected:
                correction_magnitude = abs(certified_action - action)
                print(f"CBF corrected action at timestep {self.counter.total_timesteps}: {action:.3f} -> {certified_action:.3f}")
            
            if not qp_success:
                print(f"CBF QP failed at timestep {self.counter.total_timesteps}, using clipped action")
        
        # Track statistics in counter (always record, even if CBF wasn't applied)
        self.counter.record_cbf_action(was_corrected, correction_magnitude)
        
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
            x, theta, x_dot, theta_dot = obs
            # Simple barrier function for debugging: h(x) = x_max - |x|
            h_value = self.x_max - abs(x)
            print(f"CBF VIOLATION at timestep {self.counter.total_timesteps}: x={x:.3f} (limit=±{self.x_max}), θ={theta:.3f}")
            print(f"  Barrier value h(x) = {h_value:.6f} (should be ≥ 0)")
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


def train_ppo_with_cbf(counter: ConstraintViolationCounter, seed: int):
    """Train PPO with CBF Safety Filter"""
    print("\n" + "=" * 50)
    print(f"Training PPO with CBF Safety Filter (seed {seed})")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    # CBF parameters for HOCBF - use more conservative but realistic limits
    x_max = 0.95  # Slightly less than actual limit (1.0) for safety margin
    
    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return CBFWrapper(env, counter, 
                         x_max=x_max,
                         u_min=-3.0, u_max=3.0,
                         W=1.0, lam=1e3, c1=5, c2=10, tolerance=1e-3,
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
    
    # Get CBF statistics from the counter
    cbf_stats = counter.get_cbf_stats()
    
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
        
        colors = {'PPO': 'blue', 'PPO+CBF': 'red'}
        
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
        'PPO+CBF': []
    }
    
    # Train all algorithms across all seeds
    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"RUNNING SEED {seed} ({SEEDS.index(seed) + 1}/{NUM_SEEDS})")
        print(f"{'='*80}")
        
        # Set random seed
        np.random.seed(seed)
        
        # # Train PPO
        # ppo_counter = ConstraintViolationCounter()
        # train_ppo(ppo_counter, seed)
        # counters_dict['PPO'].append(ppo_counter)
        
        # # Reset TensorFlow graph between training runs (required for TF 1.x)
        # tf.reset_default_graph()
        # print("\n[TensorFlow graph reset]\n")
        
        # Train PPO with CBF
        cbf_counter = ConstraintViolationCounter()
        train_ppo_with_cbf(cbf_counter, seed)
        counters_dict['PPO+CBF'].append(cbf_counter)
        
        # Reset TensorFlow graph between training runs (required for TF 1.x)
        tf.reset_default_graph()
        print("\n[TensorFlow graph reset]\n")
    
    # Aggregate results across seeds
    # print("\nAggregating results across seeds...")
    # aggregated_data = aggregate_returns_by_timestep(counters_dict)
    
    # # Generate timestep-based plot
    # plot_path = os.path.join(RUN_DIR, f'average_return_vs_timesteps_{save_index}.png')
    # plot_training_comparison(aggregated_data, plot_path)
    
    # # Print summary statistics
    # print_multi_seed_summary(counters_dict)
    
    # # Save aggregated data to CSV
    # print("\nSaving aggregated data...")
    # aggregated_rows = []
    # for alg_name, stats in aggregated_data.items():
    #     for timestep, mean, std in zip(stats['timesteps'], stats['mean_return'], stats['std_return']):
    #         aggregated_rows.append({
    #             'algorithm': alg_name,
    #             'timesteps': timestep,
    #             'mean_return': mean,
    #             'std_return': std,
    #         })
    
    # aggregated_df = pd.DataFrame(aggregated_rows)
    # aggregated_csv_path = os.path.join(RUN_DIR, f'aggregated_returns_{save_index}.csv')
    # aggregated_df.to_csv(aggregated_csv_path, index=False)
    # print(f"Aggregated data saved to: {aggregated_csv_path}")
    
    # print("\n" + "=" * 80)
    # print("EXPERIMENT COMPLETE!")
    # print("=" * 80)
    # print(f"Results saved in: {RUN_DIR}")
    # print(f"  - Timestep-based plot: {plot_path}")
    # print(f"  - Aggregated data CSV: {aggregated_csv_path}")
    # print(f"  - Individual seed logs in subdirectories")
    

if __name__ == "__main__":
    main()


