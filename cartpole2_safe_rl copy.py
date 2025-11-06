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

# Configuration
SEED = 42
# TOTAL_TIMESTEPS = 10_000
# STEPS_PER_EPOCH = 1000
STEPS_PER_EPOCH = 4000
TOTAL_TIMESTEPS = STEPS_PER_EPOCH * 10
MAX_X_DISPLACEMENT = 1  # Constraint threshold
UPDATE_CORRECTION_ACTION = True
CBF_REWARD_SHAPING_SIGMA = 1 # Default sigma for reward shaping (can be overridden per training run)
RUN_DIR = "runs_cartpole2_safe_rl"
os.makedirs(RUN_DIR, exist_ok=True)
save_index = "cbf_reward2"
np.random.seed(SEED)

def log_barrier_quad(x, x_max, mu=1.0, eps=1e-12):
    z = (x / x_max)**2
    z = min(z, 1 - eps)   
    return -mu * np.log(1 - z)

def log_barrier_linear(x, x_max, mu=1.0, eps=1e-12):
    z_right = np.maximum(x_max - x, eps)  # add small eps to avoid log(0)
    z_left  = np.maximum(x_max + x, eps)
    return mu * (np.log(z_right)+ np.log(z_left))


class ConstraintViolationCounter:
    """Counter for tracking constraint violations"""

    def __init__(self, x_threshold: float = MAX_X_DISPLACEMENT):
        self.x_threshold = x_threshold
        self.total_episodes = 0  # total number of episodes
        self.total_timesteps = 0  # total number of timesteps
        
        self.violations_per_epoch = []  # list of violations per epoch
        self.episodes_per_epoch = []  # list of episodes per epoch
        self.current_epoch_violations = 0 # number of violations in the current epoch
        self.current_epoch_episodes = 0 # number of episodes in the current epoch
        
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
    
    def episode_ended(self, had_violation: bool):
        """Record episode completion"""
        self.total_episodes += 1
        self.current_epoch_episodes += 1
        if had_violation:
            self.current_epoch_violations += 1
    
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

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_episodes': self.total_episodes,
            'violation_episodes': np.sum(self.violations_per_epoch),
            'total_timesteps': self.total_timesteps,
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

class CBFActionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        cbf_filter: InvertedPendulumCBF,
        counter: ConstraintViolationCounter,
        use_corrected_action_for_training: bool = False,
        reward_shaping_sigma: float | None = None,
        steps_per_epoch: int = STEPS_PER_EPOCH,
    ):
        super().__init__(env)
        self.cbf_filter = cbf_filter
        self.counter = counter
        self._last_obs = None
        self.episode_timestep = 0
        self.use_corrected_action_for_training = use_corrected_action_for_training
        if reward_shaping_sigma is not None and reward_shaping_sigma <= 0:
            raise ValueError("reward_shaping_sigma must be positive when provided.")
        self.reward_shaping_sigma = reward_shaping_sigma
        self.steps_per_epoch = steps_per_epoch
        self.epoch_timesteps = 0

    @staticmethod
    def _reorder_for_cbf(obs):
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 4:
            raise ValueError(
                "CBFActionWrapper expects observation with at least 4 entries"
            )
        return np.array([arr[0], arr[2], arr[1], arr[3]], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
        else:  # pragma: no cover - legacy API
            obs, info = result, {}
        self._last_obs = self._reorder_for_cbf(obs)
        self.episode_timestep = 0
        return obs, info

    def step(self,action):

        self.episode_timestep += 1
        uncertified = np.asarray(action, dtype=np.float64)
        cbf_state = self._last_obs
        constraint_value = self.cbf_filter._evaluate_constraint(cbf_state, uncertified)
        
        # Always call certify_action to track statistics properly
        # certify_action will return the same action if already safe
        certified_action, was_corrected = self.cbf_filter.certify_action(cbf_state, uncertified)
        
        # Ensure correct action shape for MuJoCo environment
        certified_action = np.asarray(certified_action, dtype=np.float32).reshape(-1)
        uncertified_action = uncertified.astype(np.float32).reshape(-1)
        
        step_result = self.env.step(certified_action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:  # pragma: no cover - legacy API
            obs, reward, done, info = step_result
            terminated, truncated = done, False

        cbf_state = self._reorder_for_cbf(obs)
        violated = self.counter.check_step_violation(obs)
        if violated:
            terminated = True

        info = dict(info)
        info["episode_timestep"] = self.episode_timestep
        info.update(
            {
                "cbf_corrected": bool(was_corrected),
                "uncertified_action": uncertified_action,
                "certified_action": certified_action,
                "constraint_violated": violated,
            }
        )
        info["constraint_value"] = constraint_value
        cost = self.counter.compute_cost(obs, info)
        info["cost"] = cost
        if self.reward_shaping_sigma is not None:
            sigma_sq = float(self.reward_shaping_sigma) ** 2
            correction_norm_sq = float(np.sum((uncertified_action - certified_action) ** 2))
            shaping_bonus = max(0.0, constraint_value) + np.exp(-correction_norm_sq / sigma_sq) - 1.0
            reward += shaping_bonus
            info["reward_shaping_bonus"] = shaping_bonus
            info["reward_shaping_sigma"] = float(self.reward_shaping_sigma)

        if self.use_corrected_action_for_training:
            info["training_action"] = certified_action

        if terminated or truncated:
            self.counter.episode_ended(violated)

        self.epoch_timesteps += 1
        if self.steps_per_epoch > 0 and self.epoch_timesteps % self.steps_per_epoch == 0:
            epoch_violations = self.counter.current_epoch_violations
            epoch_episodes = self.counter.current_epoch_episodes
            self.counter.epoch_ended()
            print(
                "CBF epoch ended at timestep "
                f"{self.counter.total_timesteps} "
                f"({epoch_violations} violations over {epoch_episodes} episodes this epoch)"
            )

        self._last_obs = cbf_state
        return obs, reward, terminated, truncated, info




# =======================================
# TRAINING FUNCTIONS
# =======================================

def train_ppo(counter: ConstraintViolationCounter):
    """Train standard PPO"""
    print("\n" + "=" * 50)
    print("Training Standard PPO")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    # Create a shared counter that persists across environment resets
    def env_fn():
        env = gym.make('InvertedPendulum-v4')
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
    
    return summary


def train_ppo_lagrangian(counter: ConstraintViolationCounter):
    """Train PPO-Lagrangian"""
    print("\n" + "=" * 50)
    print("Training PPO-Lagrangian")
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
        env = gym.make('InvertedPendulum-v4')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)
    
    # Cost limit: stricter for CPO to enforce better constraint satisfaction
    cost_lim = 0.5  
    
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
    
    return summary


def train_ppo_with_cbf(counter: ConstraintViolationCounter):
    """Train PPO with CBF Safety Filter"""
    print("\n" + "=" * 50)
    print("Training PPO with CBF Safety Filter")
    print("=" * 50)
    
    # Calculate training parameters
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch
    
    # Create shared CBF instance to accumulate statistics across all env instances
    shared_cbf = InvertedPendulumCBF(x_max=MAX_X_DISPLACEMENT)
    
    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return CBFActionWrapper(
            env,
            shared_cbf,  # Use shared CBF instance
            counter,
            use_corrected_action_for_training=UPDATE_CORRECTION_ACTION,
            reward_shaping_sigma=None,
        )
    
    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'ppo_cbf_{save_index}'),
        'exp_name': f'ppo_cbf_cartpole'
    }
    
    start_time = time.time()
    
    # Train PPO with CBF-wrapped environment
    ppo(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        seed=SEED,
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




def train_ppo_with_cbf_reward_shaping(
    counter: ConstraintViolationCounter,
    sigma: float = CBF_REWARD_SHAPING_SIGMA,
):
    """Train PPO with a CBF filter that applies reward shaping."""
    print("\n" + "=" * 50)
    print("Training PPO with CBF Safety Filter + Reward Shaping")
    print("=" * 50)

    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch

    # Create shared CBF instance to accumulate statistics across all env instances
    shared_cbf = InvertedPendulumCBF(x_max=MAX_X_DISPLACEMENT)

    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return CBFActionWrapper(
            env,
            shared_cbf,  # Use shared CBF instance
            counter,
            use_corrected_action_for_training=True,
            reward_shaping_sigma=sigma,
        )

    logger_kwargs = {
        'output_dir': os.path.join(RUN_DIR, f'ppo_cbf_reward_shaping_{save_index}'),
        'exp_name': 'ppo_cbf_reward_shaping_cartpole'
    }

    start_time = time.time()

    ppo(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        seed=SEED,
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
    cbf_filter_stats = shared_cbf.get_stats()

    print(f"\nPPO+CBF (Reward Shaping) Training Complete!")
    print(f"  Time: {training_time:.1f} seconds")
    print(f"  CBF Statistics:")
    print(f"    - Total actions processed: {cbf_filter_stats['total_actions']}")
    print(f"    - Actions corrected: {cbf_filter_stats['corrected_actions']}")
    print(f"    - Correction rate: {cbf_filter_stats['correction_rate']:.3f}")
    print(f"    - Average correction magnitude: {cbf_filter_stats['avg_correction']:.4f}")
    print(f"    - Maximum correction magnitude: {cbf_filter_stats['max_correction']:.4f}")
    print(f"  Training violation summary:")
    summary = counter.get_summary()
    print(f"    - Total episodes: {summary['total_episodes']}")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}")

    summary.update(cbf_filter_stats)
    summary['reward_shaping_sigma'] = sigma

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
# VISUALIZATION
# =======================================

def extract_training_data(ppo_dir: str, ppo_lag_dir: str, cpo_dir: str, cbf_dir: str, cbf_reward_dir: str,
                         ppo_counter: ConstraintViolationCounter,
                         ppo_lag_counter: ConstraintViolationCounter,
                         cpo_counter: ConstraintViolationCounter,
                         cbf_counter: ConstraintViolationCounter,
                         cbf_reward_counter: ConstraintViolationCounter) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract and combine training data from safe_rl logs and our violation counters"""
    
    # Load progress files from safe_rl logs
    ppo_progress = pd.read_csv(os.path.join(ppo_dir, 'progress.txt'), sep='\t')
    ppo_lag_progress = pd.read_csv(os.path.join(ppo_lag_dir, 'progress.txt'), sep='\t')
    cpo_progress = pd.read_csv(os.path.join(cpo_dir, 'progress.txt'), sep='\t')
    cbf_progress = pd.read_csv(os.path.join(cbf_dir, 'progress.txt'), sep='\t')
    cbf_reward_progress = pd.read_csv(os.path.join(cbf_reward_dir, 'progress.txt'), sep='\t')

    ppo_violations_per_epoch = ppo_counter.violations_per_epoch
    ppo_lag_violations_per_epoch = ppo_lag_counter.violations_per_epoch
    cpo_violations_per_epoch = cpo_counter.violations_per_epoch
    cbf_violations_per_epoch = cbf_counter.violations_per_epoch
    cbf_reward_violations_per_epoch = cbf_reward_counter.violations_per_epoch

    
    # Add actual violation data per epoch (pad with zeros if lengths don't match)
    def pad_or_truncate(data, target_length):
        if len(data) == target_length:
            return data[:]
        elif len(data) < target_length:
            return data[:] + [0] * (target_length - len(data))
        else:
            return data[:target_length]
    
    ppo_progress['ViolationsPerEpoch'] = pad_or_truncate(ppo_violations_per_epoch, len(ppo_progress))
    ppo_lag_progress['ViolationsPerEpoch'] = pad_or_truncate(ppo_lag_violations_per_epoch, len(ppo_lag_progress))
    cpo_progress['ViolationsPerEpoch'] = pad_or_truncate(cpo_violations_per_epoch, len(cpo_progress))
    cbf_progress['ViolationsPerEpoch'] = pad_or_truncate(cbf_violations_per_epoch, len(cbf_progress))
    cbf_reward_progress['ViolationsPerEpoch'] = pad_or_truncate(cbf_reward_violations_per_epoch, len(cbf_reward_progress))
    cbf_reward_progress['OriginalAverageEpRet'] = cbf_reward_progress['EpLen']
    
    return ppo_progress, ppo_lag_progress, cpo_progress, cbf_progress, cbf_reward_progress


def save_training_data_csv(ppo_progress: pd.DataFrame, ppo_lag_progress: pd.DataFrame, 
                          cpo_progress: pd.DataFrame, cbf_progress: pd.DataFrame,
                          cbf_reward_progress: pd.DataFrame, save_dir: str):
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
    
    # Save CBF data
    cbf_csv_path = os.path.join(save_dir, 'cbf_training_data.csv')
    cbf_progress.to_csv(cbf_csv_path, index=False)
    print(f"  CBF data saved to: {cbf_csv_path}")

    cbf_reward_csv_path = os.path.join(save_dir, 'cbf_reward_shaping_training_data.csv')
    cbf_reward_progress.to_csv(cbf_reward_csv_path, index=False)
    print(f"  CBF reward shaping data saved to: {cbf_reward_csv_path}")
    
    # Save summary comparison
    summary_data = {
        'Algorithm': ['PPO', 'PPO-Lagrangian', 'CPO', 'PPO+CBF', 'PPO+CBF (Reward)'],
        'Final_Avg_Return': [
            ppo_progress['AverageEpRet'].iloc[-1],
            ppo_lag_progress['AverageEpRet'].iloc[-1],
            cpo_progress['AverageEpRet'].iloc[-1],
            cbf_progress['AverageEpRet'].iloc[-1],
            cbf_reward_progress['OriginalAverageEpRet'].iloc[-1]
        ],
        'Final_Avg_Length': [
            ppo_progress['EpLen'].iloc[-1],
            ppo_lag_progress['EpLen'].iloc[-1],
            cpo_progress['EpLen'].iloc[-1],
            cbf_progress['EpLen'].iloc[-1],
            cbf_reward_progress['EpLen'].iloc[-1]
        ],
        'Total_Cost': [
            ppo_progress['CumulativeCost'].iloc[-1] if 'CumulativeCost' in ppo_progress.columns else 0,
            ppo_lag_progress['CumulativeCost'].iloc[-1] if 'CumulativeCost' in ppo_lag_progress.columns else 0,
            cpo_progress['CumulativeCost'].iloc[-1] if 'CumulativeCost' in cpo_progress.columns else 0,
            cbf_progress['CumulativeCost'].iloc[-1] if 'CumulativeCost' in cbf_progress.columns else 0,
            cbf_reward_progress['CumulativeCost'].iloc[-1] if 'CumulativeCost' in cbf_reward_progress.columns else 0,
        ]
    }
    
    # Add penalty information for algorithms that use it
    lambda_values = [0, 0, 0, 0, 0]  # Default values
    if 'Penalty' in ppo_lag_progress.columns:
        lambda_values[1] = ppo_lag_progress['Penalty'].iloc[-1]
    summary_data['Final_Lambda'] = lambda_values
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(save_dir, 'training_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Summary saved to: {summary_csv_path}")


def plot_training_comparison(ppo_progress: pd.DataFrame, ppo_lag_progress: pd.DataFrame, 
                            cpo_progress: pd.DataFrame, cbf_progress: pd.DataFrame,
                            cbf_reward_progress: pd.DataFrame,
                            ppo_counter: ConstraintViolationCounter,
                            ppo_lag_counter: ConstraintViolationCounter,
                            cpo_counter: ConstraintViolationCounter,
                            cbf_counter: ConstraintViolationCounter,
                            cbf_reward_counter: ConstraintViolationCounter,
                            save_path: str):
    """Plot comprehensive training comparison for PPO, PPO-Lagrangian, CPO, and PPO+CBF"""
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
        axes[0,0].plot(cbf_progress['Epoch'], cbf_progress['AverageEpRet'], 
                       label='PPO+CBF', linewidth=2, alpha=0.8, marker='d', markersize=4, color='purple')
        axes[0,0].plot(cbf_reward_progress['Epoch'], cbf_reward_progress['OriginalAverageEpRet'], 
                       label='PPO+CBF (Reward, original return)', linewidth=2, alpha=0.8, marker='*', markersize=4, color='orange')
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
        axes[0,1].plot(cbf_progress['Epoch'], cbf_progress['ViolationsPerEpoch'], 
                       label='PPO+CBF', linewidth=2, alpha=0.8, marker='d', markersize=4, color='purple')
        axes[0,1].plot(cbf_reward_progress['Epoch'], cbf_reward_progress['ViolationsPerEpoch'], 
                       label='PPO+CBF (Reward)', linewidth=2, alpha=0.8, marker='*', markersize=4, color='orange')
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
        if 'CumulativeCost' in cbf_progress.columns:
            axes[1,1].plot(cbf_progress['Epoch'], cbf_progress['CumulativeCost'], 
                           label='PPO+CBF', linewidth=2, alpha=0.8, marker='d', markersize=4, color='purple')
            cost_plotted = True
        if 'CumulativeCost' in cbf_reward_progress.columns:
            axes[1,1].plot(cbf_reward_progress['Epoch'], cbf_reward_progress['CumulativeCost'], 
                           label='PPO+CBF (Reward)', linewidth=2, alpha=0.8, marker='*', markersize=4, color='orange')
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
        fig.suptitle('PPO vs PPO-Lagrangian vs CPO vs PPO+CBF (with/without reward shaping): Safe RL Comparison', fontsize=16, fontweight='bold')
        
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
    
    # # Train PPO
    # ppo_counter = ConstraintViolationCounter()
    # training_results['PPO'] = train_ppo(ppo_counter)
    
    # # Reset TensorFlow graph between training runs (required for TF 1.x)
    # tf.reset_default_graph()
    # print("\n[TensorFlow graph reset]\n")
    
    # # Train PPO-Lagrangian
    # ppo_lag_counter = ConstraintViolationCounter()
    # training_results['PPO-Lagrangian'] = train_ppo_lagrangian(ppo_lag_counter)
    
    # # Reset TensorFlow graph between training runs (required for TF 1.x)
    # tf.reset_default_graph()
    # print("\n[TensorFlow graph reset]\n")
    
    # # Train CPO
    # cpo_counter = ConstraintViolationCounter()
    # training_results['CPO'] = train_cpo(cpo_counter)
    
    # # Reset TensorFlow graph between training runs (required for TF 1.x)
    # tf.reset_default_graph()
    # print("\n[TensorFlow graph reset]\n")
    
    # Train PPO with CBF
    cbf_counter = ConstraintViolationCounter()
    training_results['PPO+CBF'] = train_ppo_with_cbf(cbf_counter)
    
    # Reset TensorFlow graph between training runs (required for TF 1.x)
    tf.reset_default_graph()
    print("\n[TensorFlow graph reset]\n")

    # Train PPO with CBF + reward shaping
    cbf_reward_counter = ConstraintViolationCounter()
    training_results['PPO+CBF (Reward)'] = train_ppo_with_cbf_reward_shaping(
        cbf_reward_counter,
        sigma=CBF_REWARD_SHAPING_SIGMA,
    )
    
    # # Print comparison
    # print("\n" + "=" * 80)
    # print("TRAINING SUMMARY - Constraint Violation Comparison")
    # print("=" * 80)
    
    
    # # Compare totoal violation counts
    # ppo_viol_count = training_results['PPO']['violation_episodes']
    # ppo_lag_viol_count = training_results['PPO-Lagrangian']['violation_episodes']
    # cpo_viol_count = training_results['CPO']['violation_episodes']
    # cbf_viol_count = training_results['PPO+CBF']['violation_episodes']
    # cbf_reward_viol_count = training_results['PPO+CBF (Reward)']['violation_episodes']
    
    # print(f"\n" + "=" * 50)
    # print("VIOLATION RATE COMPARISON:")
    # print(f"  PPO:           {ppo_viol_count:.3f}")
    # print(f"  PPO-Lagrangian: {ppo_lag_viol_count:.3f}")
    # print(f"  CPO:           {cpo_viol_count:.3f}")
    # print(f"  PPO+CBF:       {cbf_viol_count:.3f}")
    # print(f"  PPO+CBF (Reward): {cbf_reward_viol_count:.3f}")
    
    # # Extract training data from logs
    # print("\nProcessing training data...")
    # ppo_progress, ppo_lag_progress, cpo_progress, cbf_progress, cbf_reward_progress = extract_training_data(
    #     os.path.join(RUN_DIR, f'ppo_{save_index}'),
    #     os.path.join(RUN_DIR, f'ppo_lagrangian_{save_index}'),
    #     os.path.join(RUN_DIR, f'cpo_{save_index}'),
    #     os.path.join(RUN_DIR, f'ppo_cbf_{save_index}'),
    #     os.path.join(RUN_DIR, f'ppo_cbf_reward_shaping_{save_index}'),
    #     ppo_counter,
    #     ppo_lag_counter,
    #     cpo_counter,
    #     cbf_counter,
    #     cbf_reward_counter,
    # )
    
    # # Save to CSV
    # save_training_data_csv(ppo_progress, ppo_lag_progress, cpo_progress, cbf_progress, cbf_reward_progress, RUN_DIR)
    
    # # Generate comprehensive plots
    # plot_training_comparison(
    #     ppo_progress,
    #     ppo_lag_progress,
    #     cpo_progress,
    #     cbf_progress,
    #     cbf_reward_progress,
    #     ppo_counter,
    #     ppo_lag_counter,
    #     cpo_counter,
    #     cbf_counter,
    #     cbf_reward_counter,
    #     os.path.join(RUN_DIR, f'comparison_{save_index}.png')
    # )
    
    # print("\n" + "=" * 80)
    # print("EXPERIMENT COMPLETE!")
    # print("=" * 80)
    # print(f"Results saved in: {RUN_DIR}")
    # print(f"  - PPO logs: {os.path.join(RUN_DIR, f'ppo_{save_index}')}")
    # print(f"  - PPO-Lagrangian logs: {os.path.join(RUN_DIR, f'ppo_lagrangian_{save_index}')}")
    # print(f"  - CPO logs: {os.path.join(RUN_DIR, f'cpo_{save_index}')}")
    # print(f"  - PPO+CBF logs: {os.path.join(RUN_DIR, f'ppo_cbf_{save_index}')}")
    # print(f"  - PPO+CBF (Reward) logs: {os.path.join(RUN_DIR, f'ppo_cbf_reward_shaping_{save_index}')}")
    # print(f"  - Comparison plot: {os.path.join(RUN_DIR, f'comparison_{save_index}.png')}")
    # print(f"  - PPO training CSV: {os.path.join(RUN_DIR, 'ppo_training_data.csv')}")
    # print(f"  - PPO-Lagrangian training CSV: {os.path.join(RUN_DIR, 'ppo_lagrangian_training_data.csv')}")
    # print(f"  - CPO training CSV: {os.path.join(RUN_DIR, 'cpo_training_data.csv')}")
    # print(f"  - PPO+CBF training CSV: {os.path.join(RUN_DIR, 'cbf_training_data.csv')}")
    # print(f"  - PPO+CBF (Reward) training CSV: {os.path.join(RUN_DIR, 'cbf_reward_shaping_training_data.csv')}")
    # print(f"  - Summary CSV: {os.path.join(RUN_DIR, 'training_summary.csv')}")
    # print("\nTo evaluate trained policies, use:")
    # print(f"  cd {os.path.join(RUN_DIR, f'ppo_{save_index}')} && python ../../safety-starter-agents/scripts/test_policy.py")
    # print(f"  cd {os.path.join(RUN_DIR, f'ppo_lagrangian_{save_index}')} && python ../../safety-starter-agents/scripts/test_policy.py")
    # print(f"  cd {os.path.join(RUN_DIR, f'cpo_{save_index}')} && python ../../safety-starter-agents/scripts/test_policy.py")
    # print(f"  cd {os.path.join(RUN_DIR, f'ppo_cbf_{save_index}')} && python ../../safety-starter-agents/scripts/test_policy.py")
    # print(f"  cd {os.path.join(RUN_DIR, f'ppo_cbf_reward_shaping_{save_index}')} && python ../../safety-starter-agents/scripts/test_policy.py")
    

if __name__ == "__main__":
    main()


