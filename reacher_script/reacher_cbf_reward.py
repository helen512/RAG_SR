import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium_robotics  # Register robotics environments
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from dataclasses import dataclass

from scipy.optimize import minimize
import cvxpy as cp

import torch
import torch.nn as nn


class CBFDebugLogger:
    """Logger for CBF debug information, corrections, and violations"""
    
    def __init__(self, log_dir: str, experiment_name: str = "cbf_debug"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{experiment_name}_debug.txt")
        
        # Initialize log file with header
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"CBF Debug Log - Experiment: {experiment_name}\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        self.debug_count = 0
        self.correction_count = 0
        self.violation_count = 0
        
    def log_debug(self, timestep: int, theta2: float, theta2_dot: float, 
                  h_actual: float, hdot_actual: float, u_nom: np.ndarray, 
                  Au_value: float, b_value: float, constraint_value: float, tolerance: float):
        """Log general debug information"""
        self.debug_count += 1
        message = (
            f"DEBUG CBF #{self.debug_count} (timestep {timestep}):\n"
            f"  State: theta2={theta2:.3f}, theta2_dot={theta2_dot:.3f}\n"
            f"  Barrier: h={h_actual:.3f}, hdot={hdot_actual:.3f}\n"
            f"  Action: u_nom={u_nom}\n"
            f"  Constraint: A·u={Au_value:.3f}, b={b_value:.3f}, (A·u-b)={constraint_value:.3f}\n"
            f"  Tolerance: {tolerance}\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        # print(message.strip())  # Also print to console
        
    def log_correction(self, timestep: int, original_action: np.ndarray, 
                      corrected_action: np.ndarray, correction_magnitude: float):
        """Log CBF action corrections"""
        self.correction_count += 1
        message = (
            f"CBF CORRECTION #{self.correction_count} (timestep {timestep}):\n"
            f"  Original:  [{original_action[0]:.4f}, {original_action[1]:.4f}]\n"
            f"  Corrected: [{corrected_action[0]:.4f}, {corrected_action[1]:.4f}]\n"
            f"  Magnitude: {correction_magnitude:.4f}\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        # print(message.strip())  # Also print to console
        
    def log_violation(self, timestep: int, theta1: float, theta2_max: float, 
                     h_value: float, certified_action: np.ndarray):
        """Log constraint violations"""
        self.violation_count += 1
        message = (
            f"CBF VIOLATION #{self.violation_count} (timestep {timestep}):\n"
            f"  θ1 = {theta1:.4f} (limit = ±{theta2_max})\n"
            f"  Barrier value h(θ1) = {h_value:.6f} (should be ≥ 0)\n"
            f"  Last certified action: [{certified_action[0]:.4f}, {certified_action[1]:.4f}]\n"
            f"  {'='*50}\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        print(message.strip())  # Also print to console
        
    def log_qp_failure(self, timestep: int):
        """Log QP solver failures"""
        message = f"CBF QP FAILED at timestep {timestep}, using clipped action\n"
        
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        print(message.strip())
        
    def close(self):
        """Close the log file with summary"""
        summary = (
            f"\n{'='*80}\n"
            f"CBF Debug Log Summary\n"
            f"Total debug entries: {self.debug_count}\n"
            f"Total corrections: {self.correction_count}\n"
            f"Total violations: {self.violation_count}\n"
            f"Ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*80}\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(summary)
        print(summary.strip())

# Configuration
TOTAL_TIMESTEPS = 700_000
MAX_THETA2_ANGLE = 2.55  # Constraint threshold for theta2
UPDATE_CORRECTION_ACTION = True
RUN_DIR = "runs_reacher_cbfreward"
os.makedirs(RUN_DIR, exist_ok=True)
save_index = "final"
REWARD_SHAPING_SIGMA = 1.0  # Parameter for reward shaping: exp(- ||uncertified - corrected||^2 / sigma^2)


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

    def __init__(self, theta2_threshold: float = MAX_THETA2_ANGLE):
        self.theta2_threshold = theta2_threshold
        self.total_episodes = 0  # total number of episodes
        self.total_timesteps = 0  # total number of timesteps
        
        # For timestep-based tracking
        self.original_returns = []  # episode original returns
        self.total_returns = []     # episode total returns (with shaping)
        self.timesteps = []  # timesteps when episodes ended
        self.episode_lengths = []  # length of each episode
        self.violated_episodes = 0  # number of episodes with violations
        
        # Current episode tracking
        self.current_episode_original_return = 0.0
        self.current_episode_total_return = 0.0
        self.current_episode_length = 0
        
        # CBF statistics tracking
        self.cbf_total_actions = 0
        self.cbf_corrected_actions = 0
        self.cbf_correction_magnitudes = []
        
    def check_violation(self, obs) -> bool:
        """Check if current observation violates constraint"""
        # For Reacher-v4: obs = [cos(theta0), cos(theta1), sin(theta0), sin(theta1), target_x, target_y, ang_vel_0, ang_vel_1, x_tip, y_tip, vector_to_target]
        # Calculate theta1 (second joint angle) from cos/sin values
        if isinstance(obs, np.ndarray) and len(obs) >= 4:
            cos_theta1, sin_theta1 = obs[1], obs[3]
            theta1 = np.arctan2(sin_theta1, cos_theta1)  # Second joint angle
            return abs(theta1) > self.theta2_threshold
        return False
    
    def compute_cost(self, obs, info) -> float:
        """Compute smooth cost signal for constrained RL algorithms"""
        if isinstance(obs, np.ndarray) and len(obs) >= 4:
            cos_theta1, sin_theta1 = obs[1], obs[3]
            theta1 = np.arctan2(sin_theta1, cos_theta1)  # Second joint angle
            current_timestep = info.get('episode_timestep', 1)
            return log_barrier_quad(abs(theta1), self.theta2_threshold)/(current_timestep/100)
        return 0.0
    
    def check_step_violation(self, obs) -> bool:
        """Record a timestep and return if violated"""
        self.total_timesteps += 1
        violated = self.check_violation(obs)
        return violated
    
    def step_reward(self, original_reward: float, total_reward: float):
        """Record reward from a step"""
        self.current_episode_original_return += original_reward
        self.current_episode_total_return += total_reward
        self.current_episode_length += 1

    def episode_ended(self, had_violation: bool):
        """Record episode completion"""
        self.total_episodes += 1
        
        # Record episode data for timestep-based tracking
        self.original_returns.append(self.current_episode_original_return)
        self.total_returns.append(self.current_episode_total_return)
        self.timesteps.append(self.total_timesteps)
        self.episode_lengths.append(self.current_episode_length)
        
        if had_violation:
            self.violated_episodes += 1
        
        # Reset current episode tracking
        self.current_episode_original_return = 0.0
        self.current_episode_total_return = 0.0
        self.current_episode_length = 0
    
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
        
        # Reset timestep-based tracking
        self.original_returns = []
        self.total_returns = []
        self.timesteps = []
        self.episode_lengths = []
        self.violated_episodes = 0
        self.current_episode_original_return = 0.0
        self.current_episode_total_return = 0.0
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
            'original_returns': self.original_returns,
            'total_returns': self.total_returns,
            'timesteps': self.timesteps,
            'episode_lengths': self.episode_lengths,
        }


# =======================================
# CBF SAFETY FILTER IMPLEMENTATION
# =======================================


# Reacher dynamics helper function
def reacher_f_g(x):
    """
    Compute f and g for Reacher dynamics.
    x = [theta0, theta1, theta0_dot, theta1_dot] - simplified state for CBF
    Returns f, g where dx/dt = f + g*u for the second joint (theta1/theta2)
    """
    theta1, theta2, theta1_dot, theta2_dot = x
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)

    ALPHA = 6.86512e-4
    BETA  = 2.24100e-4
    DELTA = 1.69004e-4

    def accel_theta(u):
        u1 = u[0]
        u2 = u[1]

            # Mass matrix (with armature)
        m11 = 1.0 + ALPHA + 2.0 * BETA * c2
        m12 =        DELTA +       BETA * c2
        m22 = 1.0 +  DELTA

        # Coriolis / centrifugal (links)
        h1 = -2.0 * BETA * s2 * theta1_dot * theta2_dot - BETA * s2 * theta2_dot * theta2_dot
        h2 =  BETA * s2 * theta1_dot * theta1_dot

        # Damping
        d1 = theta1_dot
        d2 = theta2_dot

        # RHS = tau - h - damping
        r1 = 200.0 * u1 - h1 - d1
        r2 = 200.0 * u2 - h2 - d2

        # Solve 2x2 system explicitly
        det = m11 * m22 - m12 * m12
        ddq1 = ( r1 * m22 - r2 * m12 ) / det
        ddq2 = ( m11 * r2 - m12 * r1 ) / det

        return ddq1, ddq2

    
    # Compute f and g vectors
    theta0_ddot_0, theta1_ddot_0 = accel_theta(np.array([0.0, 0.0]))
    theta0_ddot_u1, theta1_ddot_u1 = accel_theta(np.array([1.0, 0.0]))
    theta0_ddot_u2, theta1_ddot_u2 = accel_theta(np.array([0.0, 1.0]))
    
    # State vector: [theta0, theta1, theta0_dot, theta1_dot]
    f = np.array([theta1_dot, theta2_dot, theta0_ddot_0, theta1_ddot_0]) # (4,)
    g = np.array([[0.0, 0.0, theta0_ddot_u1 - theta0_ddot_0, theta1_ddot_u1 - theta1_ddot_0], [0.0, 0.0, theta0_ddot_u2 - theta0_ddot_0, theta1_ddot_u2 - theta1_ddot_0]]) # (2,4)
    
    return f, g

# ----- build A,b for theta2 constraint -----
def hocbf_A_b_theta2_constraint(x, theta2_max, c1=0.05, c2=0.5):

    f, g = reacher_f_g(x)
    theta1, theta1_dot = x[1], x[3]  # theta2 is theta1 in our state representation
    f3, g31, g32 = f[3], g[0,3], g[1,3]

    h_plus = theta2_max - theta1
    h_minus = theta2_max + theta1
    hdot_plus = -theta1_dot
    hdot_minus = theta1_dot

    A_plus = np.array([-g31, -g32])
    b_plus = np.array([f3 -c1 * hdot_plus - c2*h_plus])

    A_minus = np.array([g31, g32])
    b_minus = np.array([-f3 -c1 * hdot_minus - c2*h_minus])
    return A_plus, b_plus, A_minus, b_minus

    # # Barrier function: h = theta2_max^2 - theta1^2
    # h = theta2_max**2 - theta1**2
    # hdot = -2 * theta1 * theta1_dot
    
    # # HOCBF constraint: A·u >= b
    # A = -2 * theta1*np.array([[g31, g32]])  # (1,2)
    # b = np.array([2 * theta1_dot**2 + 2 * theta1* f3 - (c1+c2)*hdot - c1*c2*h], dtype=float)  # (1,)
    
    # return A, b

class ReacherCBF_QP_HOCBF:
    def __init__(self, theta2_max=2.55, u_min=np.array([-1.0, -1.0]), u_max=np.array([1.0, 1.0]), W=1.0, lam=1e2,
                 c1=0.05, c2=0.5, tolerance=2.0):
        self.theta2_max = theta2_max
        self.u_min, self.u_max = u_min, u_max
        self.W = W * np.eye(2) if np.ndim(W) == 0 else np.array(W)
        self.lam = lam
        self.c1, self.c2 = c1, c2
        self.tolerance = tolerance

        self.u = cp.Variable(2)
        self.delta = cp.Variable(1, nonneg=True)   # single slack now
        self.Ap = cp.Parameter((2,2))
        self.bp = cp.Parameter(2)
        self.u_des = cp.Parameter(2)
        

        obj = 0.5 * cp.quad_form(self.u- self.u_des, self.W) + self.lam * cp.sum(self.delta)
        cons = [ self.Ap @ self.u  + self.delta >= self.bp,
                 self.u >= self.u_min, self.u <= self.u_max ]
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def _evaluate_constraint(self, x, u):
        
        A_plus, b_plus, A_minus, b_minus = hocbf_A_b_theta2_constraint(x, self.theta2_max, self.c1, self.c2)
        # Correctly stack the constraints: each row is one constraint
        A = np.array([A_plus, A_minus])  # shape (2, 2) - 2 constraints x 2 control inputs
        b = np.array([b_plus[0], b_minus[0]])    # shape (2,) - extract scalar values

        u_array = np.array(u, dtype=np.float32).flatten()
        constraint_value = A @ u_array - b
        return constraint_value
    
    def step(self, x, u_nom, logger=None, timestep=0):
        A_plus, b_plus, A_minus, b_minus = hocbf_A_b_theta2_constraint(x, self.theta2_max, self.c1, self.c2)
        
        # Correctly stack the constraints: each row is one constraint  
        A = np.array([A_plus, A_minus])  # shape (2, 2) - 2 constraints x 2 control inputs
        b = np.array([b_plus[0], b_minus[0]])    # shape (2,) - extract scalar values
        
        # Ensure u_nom is a numpy array
        u_nom = np.array(u_nom, dtype=np.float32).flatten()
        
        # Compute h and h_dot for recording
        theta2 = x[1] 
        theta2_dot = x[3] 
        h_plus = self.theta2_max - theta2
        h_minus = self.theta2_max + theta2
        h_actual = min(h_plus, h_minus)
        hdot_actual = -theta2_dot if h_plus < h_minus else theta2_dot
        
        # Check if nominal action already satisfies constraint with tolerance
        constraint_values = A @ u_nom - b  # Now returns array of length 2
        # Check if any constraint is violated (any element < -tolerance)
        constraint_satisfied = np.all(constraint_values >= -self.tolerance)
        
        # Debug: print constraint details occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1 == 0 and logger is not None:  # Print every step if requested
            # Use minimum constraint value for debugging
            min_constraint_value = np.min(constraint_values)
            Au_values = A @ u_nom
            # logger.log_debug(timestep, theta2, theta2_dot, h_actual, hdot_actual, 
            #                u_nom, Au_values[0], b[0], min_constraint_value, self.tolerance)
            pass
        
        if constraint_satisfied:
            # Nominal action is safe enough, no correction needed
            return u_nom, True, h_actual, hdot_actual
        
        # Need to solve QP for safety
        self.Ap.value = A
        self.bp.value = b
        self.u_des.value = u_nom.astype(float)

        self.prob.solve(solver=cp.OSQP, verbose=False)
        ok = self.prob.status in ("optimal", "optimal_inaccurate")
        u_star = self.u.value if ok else np.clip(u_nom, self.u_min, self.u_max)
        return u_star, ok, h_actual, hdot_actual

class CBFWrapper(gym.Wrapper):
    """
    Wrapper that applies CBF safety filter to actions using ReacherCBF_QP_HOCBF
    """
    
    def __init__(self, env, counter: ConstraintViolationCounter, 
                 theta2_max: float = 2.55, u_min: np.array = np.array([-1.0, -1.0]), u_max: np.array = np.array([1.0, 1.0]), 
                 W: float = 1.0, lam: float = 1e3, c1: float = 0.5, c2: float = 0.5,
                 tolerance: float = 2.0, use_corrected_action_for_training: bool = False,
                 log_dir: str = None):
        super().__init__(env)
        self.cbf_filter = ReacherCBF_QP_HOCBF(theta2_max=theta2_max, u_min=u_min, u_max=u_max, 
                                               W=W, lam=lam, c1=c1, c2=c2, tolerance=tolerance)
        self.counter = counter
        self.episode_had_violation = False
        self.episode_timestep = 0
        self.last_obs = None
        self.use_corrected_action_for_training = use_corrected_action_for_training
        self.theta2_max = theta2_max
        
        # Data storage for h, h_dot visualization
        self.h_values = []
        self.hdot_values = []
        self.correction_status = []  # 0: no correction, 1: corrected, 2: not feasible
        self.c1 = c1
        self.c2 = c2
        self.episode_distances = []
        
        # Initialize logger
        if log_dir is None:
            log_dir = RUN_DIR
        self.logger = CBFDebugLogger(log_dir, f"cbf_wrapper_{save_index}")
        
        print(f"CBF Wrapper initialized with HOCBF:")
        print(f"  theta2_max: {theta2_max}, u_min: {u_min}, u_max: {u_max}")
        print(f"  c1: {c1}, c2: {c2}, W: {W}, lambda: {lam}, tolerance: {tolerance}")
        print(f"  Debug log file: {self.logger.log_file}")
        
    def reset(self, **kwargs):
        # Record previous episode
        if self.counter.total_timesteps > 0:
            self.counter.episode_ended(self.episode_had_violation)
        
        self.episode_had_violation = False
        self.episode_timestep = 0
        self.episode_distances = []
        
        # Handle both gym and gymnasium APIs
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            # Gymnasium API: returns (observation, info)
            obs = result[0]
            info = result[1]
            self.last_obs = obs  # Store observation for CBF
            return obs, info
        else:
            # Old gym API: returns observation
            obs = result
            self.last_obs = obs  # Store observation for CBF
            return obs, {}
        
    def step(self, action):
        self.episode_timestep += 1
        
        # Ensure action is a proper numpy array and flatten if needed
        if isinstance(action, np.ndarray):
            action = action.flatten()
        else:
            action = np.array(action, dtype=np.float32).flatten()
        
        # CLIP ACTION TO VALID RANGE [-1, 1] (Reacher-v4 action space bounds)
        # Neural network policies can output values outside bounds
        action = np.clip(action, -1.0, 1.0)
        uncertified_action = action.copy()
        
        # Extract joint actions safely - ensure we get scalars
        if len(action) >= 2:
            joint0_action = float(action[0])
            joint1_action = float(action[1])
        elif len(action) == 1:
            joint0_action = float(action[0])
            joint1_action = 0.0
        else:
            raise ValueError(f"Action must have at least 1 dimension, got {len(action)}")
        
        # Apply CBF safety filter to both joints
        was_corrected = False
        correction_magnitude = 0.0
        
        # Initialize certified_action with the original action
        certified_action = np.array([joint0_action, joint1_action], dtype=np.float32)
        
        current_state = None
        
        # Always count the action, even if CBF can't be applied
        if self.last_obs is not None and len(self.last_obs) >= 8:
            # Extract simplified state for CBF: [theta0, theta1, theta0_dot, theta1_dot]
            # Reacher-v4 obs: [cos(θ0), cos(θ1), sin(θ0), sin(θ1), target_x, target_y, ang_vel_0, ang_vel_1, x_tip, y_tip, vector_to_target]
            obs = self.last_obs
            # Calculate joint angles from cos/sin
            cos_theta0, cos_theta1 = obs[0], obs[1]
            sin_theta0, sin_theta1 = obs[2], obs[3]
            theta0 = np.arctan2(sin_theta0, cos_theta0)
            theta1 = np.arctan2(sin_theta1, cos_theta1)
            # Angular velocities
            theta0_dot, theta1_dot = obs[6], obs[7]
            current_state = np.array([theta0, theta1, theta0_dot, theta1_dot])
            
            # Use HOCBF to certify both joint actions
            certified_action, qp_success, h_value, hdot_value = self.cbf_filter.step(
                current_state, np.array([joint0_action, joint1_action]), 
                logger=self.logger, timestep=self.counter.total_timesteps
            )
            
            # Check if action was corrected
            original_action = np.array([joint0_action, joint1_action])
            was_corrected = (abs(certified_action[0] - joint0_action) > 1e-6) or (abs(certified_action[1] - joint1_action) > 1e-6)
            
            # Record h, hdot, and correction status
            self.h_values.append(h_value)
            self.hdot_values.append(hdot_value)
            
            if not qp_success:
                # QP not feasible
                self.correction_status.append(2)
                print(f"CBF QP failed at timestep {self.counter.total_timesteps}")
                if theta1 > 0:
                    certified_action = np.array([joint0_action, -1.0])
                else:
                    certified_action = np.array([joint0_action, 1.0])
                was_corrected = True
                correction_magnitude = np.linalg.norm(certified_action - original_action)
                self.logger.log_qp_failure(self.counter.total_timesteps)
            elif was_corrected:
                # Action was corrected
                self.correction_status.append(1)
                correction_magnitude = np.linalg.norm(certified_action - np.array([joint0_action, joint1_action]))
                # print(f"CBF corrected action at timestep {self.counter.total_timesteps}:")
                # print(f"  joint0: {joint0_action:.3f} -> {certified_action[0]:.3f}")
                # print(f"  joint1: {joint1_action:.3f} -> {certified_action[1]:.3f}")
                correction_magnitude = np.linalg.norm(certified_action - original_action)
                self.logger.log_correction(self.counter.total_timesteps, original_action, 
                                         certified_action, correction_magnitude)
            else:
                # No correction needed
                self.correction_status.append(0)
        
        # Track statistics in counter (always record, even if CBF wasn't applied)
        self.counter.record_cbf_action(was_corrected, correction_magnitude)
        
        
        # Handle both gym and gymnasium APIs
        result = self.env.step(certified_action)
        if len(result) == 5:
            # Gymnasium API: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            # Old gym API: (obs, reward, done, info)
            obs, reward, done, info = result
            terminated = done
            truncated = False
        
        self.last_obs = obs  # Update last observation
        
        # Check for violation first to determine reward
        violated = self.counter.check_step_violation(obs)
        
        # Calculate Reward Shaping (r_cbf)
        r_cbf = 0.0
        if current_state is not None:
             # Recalculate A, b for reward
            A_plus, b_plus, A_minus, b_minus = hocbf_A_b_theta2_constraint(current_state, self.theta2_max, self.c1, self.c2)
            A = np.array([A_plus, A_minus])
            b = np.array([b_plus[0], b_minus[0]])
            
            # Use certified_action (the one actually taken)
            # Flatten if needed
            cert_act_flat = certified_action.flatten()
            margin = np.min(A @ cert_act_flat - b)
            
            # Correction distance
            # uncertified_action is float array of flattened action
            dist_sq = np.sum((uncertified_action - cert_act_flat)**2)
            
            # r_cbf calculation
            r_cbf = 10e-4*(max(margin, 0) + np.exp(-dist_sq / (REWARD_SHAPING_SIGMA**2)) - 1)
        
        # Apply heavy negative reward for violations instead of positive reward  
        original_reward = reward
        modified_reward = reward + r_cbf
        
        if violated:
            self.episode_had_violation = True
            modified_reward = -200.0 + reward + r_cbf  # Heavy negative penalty for violation + shaping
            terminated = True  # Terminate episode immediately on violation
            done = True
        
        # Add episode timestep to info
        info['episode_timestep'] = self.episode_timestep
        info['original_reward'] = reward  # Store original reward for reference
        info['cbf_reward'] = r_cbf
        
        # Calculate Euclidean distance and track
        if len(obs) >= 10:
            target = obs[4:6]
            tip = obs[8:10]
            dist = np.linalg.norm(target - tip)
            self.episode_distances.append(dist)
            
        if terminated or truncated:
            last_10 = self.episode_distances[-10:] if self.episode_distances else [0.0]
            info["last_10_dist_sum"] = sum(last_10)
        
        # Track reward for timestep-based analysis
        self.counter.step_reward(original_reward, modified_reward)
        
        # Log violation details if violated
        if violated:
            if len(obs) >= 4:
                cos_theta1, sin_theta1 = obs[1], obs[3]
                theta1 = np.arctan2(sin_theta1, cos_theta1)
            else:
                theta1 = 0.0
            # Barrier function for debugging: h(theta1) = theta2_max^2 - theta1^2
            h_value = self.theta2_max**2 - theta1**2
            print(f"CBF VIOLATION at timestep {self.counter.total_timesteps}: θ1={theta1:.3f} (limit=±{self.theta2_max})")
            print(f"  Barrier value h(θ1) = {h_value:.6f} (should be ≥ 0)")
            print(f"  Last action was certified: {certified_action}")
            print(f"  Heavy negative reward applied: {modified_reward}")
            print(f"  {'='*50}")
            self.logger.log_violation(self.counter.total_timesteps, theta1, 
                                    self.theta2_max, h_value, certified_action)
        
        # Add CBF-related information to info dict
        info.update({
            'cbf_corrected': was_corrected,
            'uncertified_action': uncertified_action,
            'certified_action': certified_action,
            'constraint_violated': violated,
            'r_cbf': r_cbf
        })
        
        # Add cost information
        cost = self.counter.compute_cost(obs, info)
        info['cost'] = cost
        
        # Add training action info if requested (for corrected action training)
        if self.use_corrected_action_for_training:
            info['training_action'] = certified_action
        
        return obs, modified_reward, terminated, truncated, info
    
    def plot_h_hdot_phase(self, save_path=None):
        """Plot h vs h_dot phase diagram with color coding for correction status"""
        if len(self.h_values) == 0:
            print("No data to plot")
            return
        
        # Convert to numpy arrays for easier manipulation
        h_arr = np.array(self.h_values)
        hdot_arr = np.array(self.hdot_values)
        status_arr = np.array(self.correction_status)
        
        # Separate data by status
        no_correction = status_arr == 0
        corrected = status_arr == 1
        not_feasible = status_arr == 2
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot different statuses with different colors - same size, more transparent
        marker_size = 15
        transparency = 0.15
        
        if np.any(no_correction):
            plt.scatter(h_arr[no_correction], hdot_arr[no_correction], 
                       c='blue', alpha=transparency, s=marker_size, label='No correction needed')
        if np.any(corrected):
            plt.scatter(h_arr[corrected], hdot_arr[corrected], 
                       c='red', alpha=transparency, s=marker_size, label='Corrected')
        if np.any(not_feasible):
            plt.scatter(h_arr[not_feasible], hdot_arr[not_feasible], 
                       c='purple', alpha=transparency*2, s=marker_size, label='Not feasible')
        
        plt.xlabel('h (barrier function)', fontsize=12)
        plt.ylabel('h_dot (barrier derivative)', fontsize=12)
        plt.title(f'CBF Phase Diagram: h vs h_dot (c1={self.c1}, c2={self.c2})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add a line at h=0 to show the safety boundary
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Safety boundary (h=0)')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.logger.log_dir, f'cbf_phase_diagram_c1_{self.c1}_c2_{self.c2}.png')
        
        plt.savefig(save_path, dpi=150)
        print(f"Phase diagram saved to: {save_path}")
        plt.close()
        
        # Print statistics
        print(f"\nPhase diagram statistics:")
        print(f"  Total steps: {len(self.h_values)}")
        print(f"  No correction: {np.sum(no_correction)} ({100*np.sum(no_correction)/len(self.h_values):.1f}%)")
        print(f"  Corrected: {np.sum(corrected)} ({100*np.sum(corrected)/len(self.h_values):.1f}%)")
        print(f"  Not feasible: {np.sum(not_feasible)} ({100*np.sum(not_feasible)/len(self.h_values):.1f}%)")
    
    def close(self):
        """Close the wrapper and logger"""
        # Generate and save phase diagram before closing
        self.plot_h_hdot_phase()
        
        if hasattr(self, 'logger'):
            self.logger.close()
        super().close()


class CBFCorrectionCallback(BaseCallback):
    """
    Callback to update the action buffer with the corrected action from CBF
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access the infos from the environment
        infos = self.locals["infos"]
        actions = self.locals["actions"]
        
        # Check if we have corrected actions in infos
        for i, info in enumerate(infos):
            if "training_action" in info:
                # Replace the action that will be stored in the buffer
                # We modify the array in-place to ensure it affects the local variable in PPO
                corrected_action = info["training_action"]
                # Ensure dimensions match
                if actions[i].shape == corrected_action.shape:
                    actions[i] = corrected_action
                else:
                    # Handle potential shape mismatch (e.g. if one is flattened)
                    actions[i] = corrected_action.reshape(actions[i].shape)
                    
        return True


def train_ppo_with_cbf(counter: ConstraintViolationCounter, c1: float, c2: float):
    """Train PPO with CBF Safety Filter"""
    print("\n" + "=" * 50)
    print(f"Training PPO with CBF Safety Filter (c1={c1}, c2={c2})")
    print("=" * 50)
    
    # CBF parameters for HOCBF - use more conservative but realistic limits
    theta2_max = 2.4  # Slightly less than actual limit (2.55) for safety margin
    
    # Store reference to wrapper for plotting
    cbf_wrapper_instance = None
    
    log_dir = os.path.join(RUN_DIR, f'ppo_cbf_{save_index}_c1_{c1}_c2_{c2}')
    os.makedirs(log_dir, exist_ok=True)
    video_dir = os.path.join(log_dir, "videos")

    def env_fn():
        nonlocal cbf_wrapper_instance
        env = gym.make('Reacher-v5', render_mode='rgb_array')
        # Record video every 200 episodes
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: x % 200 == 0)
        cbf_wrapper_instance = CBFWrapper(env, counter, 
                         theta2_max=theta2_max,
                         u_min=np.array([-1.0, -1.0]), u_max=np.array([1.0, 1.0]),
                         W=1.0, lam=1e3, c1=c1, c2=c2, tolerance=10,
                         use_corrected_action_for_training=UPDATE_CORRECTION_ACTION, 
                         log_dir=log_dir)
        
        # Wrap with Monitor
        env = Monitor(cbf_wrapper_instance, log_dir, info_keywords=("original_reward", "constraint_violated", "last_10_dist_sum"))
        return env
    
    start_time = time.time()
    
    # Train PPO with CBF-wrapped environment
    env = env_fn()
    
    model = PPO(
        "MlpPolicy",
        env,
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
        tensorboard_log=log_dir,
        device="auto" 
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=700_000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=f"ppo_cbf"
    )

    # Create the callback for CBF action correction
    cbf_callback = CBFCorrectionCallback()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, cbf_callback]
    )
    
    model.save(os.path.join(log_dir, "final_model"))
    
    training_time = time.time() - start_time
    
    # Generate phase diagram plot
    if cbf_wrapper_instance is not None:
        print("\nGenerating CBF phase diagram...")
        cbf_wrapper_instance.plot_h_hdot_phase()
        cbf_wrapper_instance.close()
    
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
    
    # Plot: Original and Total Reward vs Timesteps
    plt.figure(figsize=(12, 8))
    
    # Plot Original Rewards
    plt.plot(counter.timesteps, counter.original_returns, alpha=0.2, color='blue', label='Original Episode Reward')
    if len(counter.original_returns) >= 50:
        rolling_mean_orig = pd.Series(counter.original_returns).rolling(window=50).mean()
        plt.plot(counter.timesteps, rolling_mean_orig, color='blue', linewidth=2, label='Original Mean (50)')
        
    # Plot Total Rewards
    plt.plot(counter.timesteps, counter.total_returns, alpha=0.2, color='green', label='Total Episode Reward (with CBF shaping)')
    if len(counter.total_returns) >= 50:
        rolling_mean_total = pd.Series(counter.total_returns).rolling(window=50).mean()
        plt.plot(counter.timesteps, rolling_mean_total, color='green', linewidth=2, label='Total Mean (50)')
        
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Reward Comparison: Original vs Total (Shaped)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(log_dir, "reward_comparison.png")
    plt.savefig(save_path)
    print(f"Saved reward comparison plot to {save_path}")
    plt.close()

    return summary


def main():
    counter = ConstraintViolationCounter()
    train_ppo_with_cbf(counter,15, 70)
    
if __name__ == "__main__":
    main()
