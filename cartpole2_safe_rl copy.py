import os
# Set environment variable to use legacy Keras (Keras 2) instead of Keras 3
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Callable, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")
# Import TensorFlow to reset graph between runs
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Import from safety-starter-agents
sys.path.append('/home/dmy/gymtest/safety-starter-agents')
from safe_rl import ppo, ppo_lagrangian, cpo

# Configuration
BASE_SEED = 42
NUM_SEEDS = 10
SEED_LIST = [BASE_SEED + i for i in range(NUM_SEEDS)]
# TOTAL_TIMESTEPS = 10_000
# STEPS_PER_EPOCH = 1000
STEPS_PER_EPOCH = 4000
TOTAL_TIMESTEPS = STEPS_PER_EPOCH * 10
MAX_X_DISPLACEMENT = 1  # Constraint threshold
RUN_DIR = "runs_cartpole2_safe_rl"
os.makedirs(RUN_DIR, exist_ok=True)
EXPERIMENT_TAG = "multi_seed"

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


def set_global_seeds(seed: int):
    """Set all relevant random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def get_output_dir(algo_key: str, seed: int) -> str:
    """Construct and create the output directory for a given algorithm/seed."""
    dir_path = os.path.join(RUN_DIR, f"{algo_key}_{EXPERIMENT_TAG}", f"seed_{seed}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def read_progress_file(output_dir: str) -> pd.DataFrame:
    """Load the safe_rl progress.txt file for a completed run."""
    progress_path = os.path.join(output_dir, 'progress.txt')
    if not os.path.exists(progress_path):
        raise FileNotFoundError(f"Expected progress.txt at {progress_path}")
    return pd.read_csv(progress_path, sep='\t')


def _extract_timesteps(progress_df: pd.DataFrame) -> np.ndarray:
    """Retrieve a timestep axis from a progress dataframe."""
    for column in ('TotalEnvInteracts', 'TotalEnvSteps', 'TimestepsSoFar'):
        if column in progress_df.columns:
            return progress_df[column].to_numpy()
    if 'Epoch' in progress_df.columns:
        return (progress_df['Epoch'] * STEPS_PER_EPOCH).to_numpy()
    return np.arange(len(progress_df)) * STEPS_PER_EPOCH


def aggregate_returns_across_seeds(progress_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate AverageEpRet across multiple seeds."""
    if not progress_dfs:
        raise ValueError("No progress data provided for aggregation.")

    min_len = min(len(df) for df in progress_dfs)
    truncated_returns = []
    timesteps = None

    for df in progress_dfs:
        if 'AverageEpRet' not in df.columns:
            raise KeyError("progress.txt is missing the 'AverageEpRet' column.")
        trimmed = df.head(min_len).reset_index(drop=True)
        if timesteps is None:
            timesteps = _extract_timesteps(trimmed)
        truncated_returns.append(trimmed['AverageEpRet'].to_numpy())

    returns_matrix = np.vstack(truncated_returns)
    mean_returns = returns_matrix.mean(axis=0)
    std_returns = returns_matrix.std(axis=0)

    return pd.DataFrame(
        {
            'Timesteps': timesteps,
            'MeanReturn': mean_returns,
            'StdReturn': std_returns,
        }
    )


def plot_average_return_vs_timesteps(aggregated: Dict[str, pd.DataFrame], save_path: str) -> None:
    """Plot mean return vs timesteps with a shaded std region for each algorithm."""
    if not aggregated:
        print("\nNo aggregated data provided; skipping plot generation.")
        return

    print("\nGenerating average return vs timesteps plot...")
    plt.figure(figsize=(10, 6))

    for algo_name, df in aggregated.items():
        if df.empty:
            continue
        timesteps = df['Timesteps'].to_numpy()
        mean_return = df['MeanReturn'].to_numpy()
        std_return = df['StdReturn'].to_numpy()
        plt.plot(timesteps, mean_return, label=algo_name, linewidth=2)
        plt.fill_between(
            timesteps,
            mean_return - std_return,
            mean_return + std_return,
            alpha=0.2
        )

    plt.xlabel('Timesteps')
    plt.ylabel('Average Episode Return')
    plt.title('Average Return vs Timesteps (Mean ± Std over Seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {save_path}")


def _mean_and_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


def print_violation_statistics(summary_records: Dict[str, List[Dict]]) -> None:
    """Print mean/stdev statistics for constraint violations across seeds."""
    print("\n" + "=" * 72)
    print("Constraint Violation Summary Across Seeds")
    for method, summaries in summary_records.items():
        if not summaries:
            continue
        total_eps = [s.get('total_episodes', 0) for s in summaries]
        viol_eps = [s.get('violation_episodes', 0) for s in summaries]
        violation_rates = [
            (v / t) if t else 0.0
            for v, t in zip(viol_eps, total_eps)
        ]

        total_mean, total_std = _mean_and_std(total_eps)
        viol_mean, viol_std = _mean_and_std(viol_eps)
        rate_mean, rate_std = _mean_and_std(violation_rates)

        print(f"\n{method}:")
        print(f"  Mean total episodes: {total_mean:.2f} +/- {total_std:.2f}")
        print(f"  Mean violated episodes: {viol_mean:.2f} +/- {viol_std:.2f}")
        print(f"  Mean violation rate: {rate_mean:.4f} +/- {rate_std:.4f}")
    print("=" * 72 + "\n")


def run_algorithm_over_seeds(
    label: str,
    train_fn: Callable[[ConstraintViolationCounter, int], Tuple[Dict, pd.DataFrame]],
    seeds: List[int],
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    """Execute one training algorithm across all requested seeds."""
    summaries: List[Dict] = []
    progress: List[pd.DataFrame] = []

    for seed in seeds:
        print(f"\n--- {label}: seed {seed} ---")
        tf.reset_default_graph()
        set_global_seeds(seed)
        counter = ConstraintViolationCounter()
        summary, progress_df = train_fn(counter, seed)
        summaries.append(summary)
        progress.append(progress_df)

    return summaries, progress


# =======================================
# TRAINING FUNCTIONS
# =======================================

def train_ppo(counter: ConstraintViolationCounter, seed: int) -> Tuple[Dict, pd.DataFrame]:
    """Train standard PPO for a single seed run."""
    print("\n" + "=" * 50)
    print(f"Training Standard PPO | seed {seed}")
    print("=" * 50)

    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch

    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)

    output_dir = get_output_dir('ppo', seed)
    logger_kwargs = {
        'output_dir': output_dir,
        'exp_name': f'ppo_cartpole_seed_{seed}'
    }

    start_time = time.time()

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

    print(f"\nPPO Training Complete! (seed {seed})")
    print(f"  Time: {training_time:.1f} seconds")
    summary = counter.get_summary()
    print(f"  Training violation summary:")
    print(f"    - Total episodes: {summary['total_episodes']}")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}")

    progress_df = read_progress_file(output_dir)
    progress_df['Seed'] = seed

    return summary, progress_df


def train_ppo_lagrangian(counter: ConstraintViolationCounter, seed: int) -> Tuple[Dict, pd.DataFrame]:
    """Train PPO-Lagrangian for a single seed run."""
    print("\n" + "=" * 50)
    print(f"Training PPO-Lagrangian | seed {seed}")
    print("=" * 50)

    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch

    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)

    cost_lim = 2.0

    output_dir = get_output_dir('ppo_lagrangian', seed)
    logger_kwargs = {
        'output_dir': output_dir,
        'exp_name': f'ppo_lagrangian_cartpole_seed_{seed}'
    }

    start_time = time.time()

    ppo_lagrangian(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=0.99,
        lam=0.95,
        cost_gamma=0.99,
        cost_lam=0.95,
        target_kl=0.01,
        cost_lim=cost_lim,
        penalty_init=0.85,
        penalty_lr=0.035,
        vf_lr=3e-4,
        vf_iters=80,
        logger_kwargs=logger_kwargs,
        save_freq=10
    )

    training_time = time.time() - start_time

    print(f"\nPPO-Lagrangian Training Complete! (seed {seed})")
    print(f"  Time: {training_time:.1f} seconds")
    summary = counter.get_summary()
    print(f"  Training violation summary:")
    print(f"    - Total episodes: {summary['total_episodes']}")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}")

    progress_df = read_progress_file(output_dir)
    progress_df['Seed'] = seed

    return summary, progress_df


def train_cpo(counter: ConstraintViolationCounter, seed: int) -> Tuple[Dict, pd.DataFrame]:
    """Train CPO for a single seed run."""
    print("\n" + "=" * 50)
    print(f"Training CPO (Constrained Policy Optimization) | seed {seed}")
    print("=" * 50)

    steps_per_epoch = STEPS_PER_EPOCH
    epochs = TOTAL_TIMESTEPS // steps_per_epoch

    def env_fn():
        env = gym.make('InvertedPendulum-v4')
        return ConstrainedCartPoleWrapper(env, counter, steps_per_epoch)

    cost_lim = 0.5

    output_dir = get_output_dir('cpo', seed)
    logger_kwargs = {
        'output_dir': output_dir,
        'exp_name': f'cpo_cartpole_seed_{seed}'
    }

    start_time = time.time()

    cpo(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=0.99,
        lam=0.95,
        cost_gamma=0.99,
        cost_lam=0.95,
        target_kl=0.005,
        cost_lim=cost_lim,
        vf_lr=3e-4,
        vf_iters=80,
        logger_kwargs=logger_kwargs,
        save_freq=10
    )

    training_time = time.time() - start_time

    print(f"\nCPO Training Complete! (seed {seed})")
    print(f"  Time: {training_time:.1f} seconds")
    summary = counter.get_summary()
    print(f"  Training violation summary:")
    print(f"    - Total episodes: {summary['total_episodes']}")
    print(f"    - Episodes with x-violations: {summary['violation_episodes']}")

    progress_df = read_progress_file(output_dir)
    progress_df['Seed'] = seed

    return summary, progress_df


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
# MAIN EXECUTION
# =======================================

def main():
    """Run PPO, PPO-Lagrangian, and CPO across multiple seeds."""
    seeds = SEED_LIST
    print("\n" + "=" * 80)
    print(f"Running multi-seed experiments for seeds: {seeds}")
    print("=" * 80)

    algorithms: List[Tuple[str, str, Callable[[ConstraintViolationCounter, int], Tuple[Dict, pd.DataFrame]]]] = [
        ("PPO", "ppo", train_ppo),
        ("PPO-Lagrangian", "ppo_lagrangian", train_ppo_lagrangian),
        ("CPO", "cpo", train_cpo),
    ]

    aggregated_returns: Dict[str, pd.DataFrame] = {}
    violation_summaries: Dict[str, List[Dict]] = {}

    for label, key, train_fn in algorithms:
        summaries, progress_dfs = run_algorithm_over_seeds(label, train_fn, seeds)
        violation_summaries[label] = summaries
        aggregated_returns[label] = aggregate_returns_across_seeds(progress_dfs)

    plot_path = os.path.join(RUN_DIR, f'average_return_vs_timesteps_{EXPERIMENT_TAG}.png')
    plot_average_return_vs_timesteps(aggregated_returns, plot_path)

    print_violation_statistics(violation_summaries)

    print("Summary of saved artifacts:")
    print(f"  Average return plot: {plot_path}")
    for label, key, _ in algorithms:
        algo_dir = os.path.join(RUN_DIR, f"{key}_{EXPERIMENT_TAG}")
        print(f"  {label} logs: {algo_dir}")


if __name__ == "__main__":
    main()


