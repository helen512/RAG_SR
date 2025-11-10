import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Set environment variable to use legacy Keras (Keras 2) instead of Keras 3
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Import from safety-starter-agents
sys.path.append('/home/dmy/gymtest/safety-starter-agents')
from safe_rl.utils.load_utils import load_policy

# Import classes from the main training script
from cartpole2_safe_rl_multi_seed_copy import (
    ConstraintViolationCounter, 
    ConstrainedCartPoleWrapper,
    MAX_X_DISPLACEMENT,
    STEPS_PER_EPOCH
)

# Configuration
RUN_DIR = "runs_cartpole2_safe_rl_multi_seed"
SAVE_INDEX = "experiment2"
TRAINING_SEEDS = [42, 43, 44, 45, 46]  # Seeds used for training
N_EVAL_EPISODES = 60
EVAL_SEEDS = [1000 + i for i in range(N_EVAL_EPISODES)]  # Evaluation seeds: 1000-1059

# Algorithm configurations  
ALGORITHMS = {
    'PPO': 'ppo',
    'PPO-Lagrangian': 'ppo_lagrangian', 
    'CPO': 'cpo',
    'PPO+CBF': 'ppo_cbf'  # PPO+CBF has its own directory structure
}


def find_model_directories() -> Dict[str, Dict[int, str]]:
    """Find all trained model directories organized by algorithm and seed"""
    model_dirs = {}
    
    for alg_name, alg_prefix in ALGORITHMS.items():
        model_dirs[alg_name] = {}
        
        for seed in TRAINING_SEEDS:
            model_dir = os.path.join(RUN_DIR, f'{alg_prefix}_{SAVE_INDEX}_seed_{seed}')
            
            if os.path.exists(model_dir):
                # Check if there are saved models in this directory
                saved_models = [f for f in os.listdir(model_dir) if 'simple_save' in f]
                if saved_models:
                    model_dirs[alg_name][seed] = model_dir
                    print(f"Found model: {alg_name} seed {seed} -> {model_dir}")
                else:
                    print(f"Warning: No saved models found in {model_dir}")
            else:
                print(f"Warning: Model directory not found: {model_dir}")
    
    return model_dirs


def evaluate_policy(model_path: str, algorithm: str, seed: int) -> Dict:
    """Evaluate a single trained policy over multiple episodes"""
    print(f"\nEvaluating {algorithm} (seed {seed})")
    print(f"  Model path: {model_path}")
    
    # Create evaluation counter
    eval_counter = ConstraintViolationCounter()
    
    # Load the trained policy
    try:
        # Try loading with 'last' first, then try without iteration number
        try:
            env, get_action, sess = load_policy(model_path, itr='last', deterministic=True)
        except:
            # If 'last' fails, try loading the single simple_save directory
            env, get_action, sess = load_policy(model_path, itr='', deterministic=True)
        print(f"  Successfully loaded policy")
    except Exception as e:
        print(f"  Error loading policy: {e}")
        return None
    
    # Create evaluation environment
    eval_env = gym.make('InvertedPendulum-v4')
    wrapped_env = ConstrainedCartPoleWrapper(eval_env, eval_counter, steps_per_epoch=STEPS_PER_EPOCH)
    
    episode_returns = []
    episode_lengths = []
    
    # Run evaluation episodes
    for ep in range(N_EVAL_EPISODES):
        # Set evaluation seed for reproducibility
        eval_seed = EVAL_SEEDS[ep]
        try:
            wrapped_env.seed(eval_seed)  # Try old gym API
        except AttributeError:
            pass  # Newer gym versions handle seeding differently
        np.random.seed(eval_seed)
        
        # Reset environment (handle both old and new gym APIs)
        reset_result = wrapped_env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # New gym API returns (obs, info)
        else:
            obs = reset_result  # Old gym API returns obs
        
        ep_ret = 0.0
        ep_len = 0
        done = False
        
        while not done and ep_len < 1000:  # CartPole max episode length
            try:
                # Get action from trained policy
                action = get_action(obs)
                obs, reward, done, info = wrapped_env.step(action)
                ep_ret += reward
                ep_len += 1
                
            except Exception as e:
                print(f"    Error during episode {ep} at step {ep_len}: {e}")
                done = True
                break
        
        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)
        
        if (ep + 1) % 20 == 0:
            print(f"    Completed {ep + 1}/{N_EVAL_EPISODES} episodes")
    
    # Handle final episode
    if wrapped_env.episode_had_violation:
        eval_counter.episode_ended(True)
    
    # Close TensorFlow session
    sess.close()
    
    # Compute summary statistics
    summary = eval_counter.get_summary()
    summary.update({
        'algorithm': algorithm,
        'training_seed': seed,
        'model_path': model_path,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    })
    
    print(f"  Results: Return = {summary['mean_return']:.2f} ± {summary['std_return']:.2f}, "
          f"Violations = {summary['violation_episodes']}/{summary['total_episodes']}")
    
    return summary


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("MULTI-SEED POLICY EVALUATION")
    print("=" * 80)
    print(f"Evaluation configuration:")
    print(f"  - Episodes per policy: {N_EVAL_EPISODES}")
    print(f"  - Evaluation seeds: {EVAL_SEEDS[0]} to {EVAL_SEEDS[-1]}")
    print(f"  - Training seeds: {TRAINING_SEEDS}")
    print(f"  - Results directory: {RUN_DIR}")
    
    # Find all trained model directories
    model_dirs = find_model_directories()
    
    # Store evaluation results
    all_results = []
    algorithm_results = {alg: [] for alg in ALGORITHMS.keys()}
    
    # Evaluate each policy
    for alg_name in ALGORITHMS.keys():
        if alg_name not in model_dirs or not model_dirs[alg_name]:
            print(f"\nNo models found for {alg_name}, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"EVALUATING {alg_name}")
        print(f"{'='*60}")
        
        for seed in TRAINING_SEEDS:
            if seed in model_dirs[alg_name]:
                model_path = model_dirs[alg_name][seed]
                
                # Reset TensorFlow graph before loading new policy
                tf.reset_default_graph()
                
                result = evaluate_policy(model_path, alg_name, seed)
                
                if result is not None:
                    all_results.append(result)
                    algorithm_results[alg_name].append(result)
                
                # Small delay to ensure clean separation between evaluations
                time.sleep(1)
            else:
                print(f"  Skipping {alg_name} seed {seed} (model not found)")
    
    # Compute and print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_data = []
    
    for alg_name, results in algorithm_results.items():
        if not results:
            print(f"\n{alg_name}: No results available")
            continue
            
        # Extract metrics across all seeds
        returns = [r['mean_return'] for r in results]
        violation_rates = [r['violation_rate'] for r in results]
        episode_counts = [r['total_episodes'] for r in results]
        
        # Compute statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_violation_rate = np.mean(violation_rates)
        std_violation_rate = np.std(violation_rates)
        mean_episodes = np.mean(episode_counts)
        
        # Print results
        print(f"\n{alg_name} (across {len(results)} seeds):")
        print(f"  Mean Return:      {mean_return:8.2f} ± {std_return:6.2f}")
        print(f"  Violation Rate:   {mean_violation_rate:8.4f} ± {std_violation_rate:6.4f}")
        print(f"  Episodes:         {mean_episodes:8.1f}")
        
        # Store for CSV export
        summary_data.append({
            'algorithm': alg_name,
            'n_seeds': len(results),
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_violation_rate': mean_violation_rate,
            'std_violation_rate': std_violation_rate,
            'mean_episodes': mean_episodes
        })
    
    # Save detailed results to CSV
    detailed_rows = []
    for result in all_results:
        detailed_rows.append({
            'algorithm': result['algorithm'],
            'training_seed': result['training_seed'],
            'mean_return': result['mean_return'],
            'std_return': result['std_return'],
            'violation_episodes': result['violation_episodes'],
            'total_episodes': result['total_episodes'],
            'violation_rate': result['violation_rate'],
            'mean_length': result['mean_length']
        })
    
    # Save results
    if detailed_rows:
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv_path = os.path.join(RUN_DIR, f'detailed_evaluation_results_{SAVE_INDEX}.csv')
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"\nDetailed results saved to: {detailed_csv_path}")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(RUN_DIR, f'summary_evaluation_results_{SAVE_INDEX}.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary results saved to: {summary_csv_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
