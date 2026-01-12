"""
Script to verify CBF dynamics formulation against actual Reacher-v4 environment.

This script:
1. Takes states from the Reacher environment
2. Predicts next states using CBF dynamics (f + g*u)
3. Compares predictions with actual environment transitions
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, List

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Skipping visualizations.")

# =======================================
# CBF DYNAMICS (from reacher_safe_rl_multi_seed.py)
# =======================================

def reacher_f_g(x):
    """
    Compute drift vector f and control matrix g for Reacher dynamics.
    
    Args:
        x: State vector [theta0, theta1, theta0_dot, theta1_dot]
    
    Returns:
        f: Drift vector (4,)
        g: Control matrix (2, 4)
    """
    theta1, theta2, theta1_dot, theta2_dot = x
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    
    # Dynamics parameters (from the script)
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
    f = np.array([theta1_dot, theta2_dot, theta0_ddot_0, theta1_ddot_0])  # (4,)
    g = np.array([
        [0.0, 0.0, theta0_ddot_u1 - theta0_ddot_0, theta1_ddot_u1 - theta1_ddot_0],
        [0.0, 0.0, theta0_ddot_u2 - theta0_ddot_0, theta1_ddot_u2 - theta1_ddot_0]
    ])  # (2, 4)
    
    return f, g


def predict_next_state(state, action, dt=0.02):
    """
    Predict next state using CBF dynamics with Euler integration.
    
    Args:
        state: Current state [theta0, theta1, theta0_dot, theta1_dot]
        action: Control input [u0, u1]
        dt: Time step (Reacher default is 0.02s)
    
    Returns:
        next_state: Predicted next state
    """
    f, g = reacher_f_g(state)
    
    # Compute state derivative: x_dot = f + g^T * u
    # Note: g is (2, 4), so g^T is (4, 2), and g^T @ u gives (4,)
    state_dot = f + g.T @ action
    
    # Euler integration
    next_state = state + state_dot * dt
    
    return next_state


# =======================================
# ENVIRONMENT UTILITIES
# =======================================

def obs_to_state(obs):
    """
    Convert Reacher-v4 observation to simplified state.
    
    Reacher-v4 obs: [cos(θ0), cos(θ1), sin(θ0), sin(θ1), target_x, target_y, 
                     ang_vel_0, ang_vel_1, x_tip, y_tip, vector_to_target_x, vector_to_target_y]
    
    Returns:
        state: [theta0, theta1, theta0_dot, theta1_dot]
    """
    cos_theta0, cos_theta1 = obs[0], obs[1]
    sin_theta0, sin_theta1 = obs[2], obs[3]
    theta0 = np.arctan2(sin_theta0, cos_theta0)
    theta1 = np.arctan2(sin_theta1, cos_theta1)
    theta0_dot, theta1_dot = obs[6], obs[7]
    
    return np.array([theta0, theta1, theta0_dot, theta1_dot])


def get_actual_next_state(env, action):
    """
    Apply action in environment and return next state.
    
    Returns:
        next_state: Actual next state from environment
        obs: Full observation from environment
    """
    obs, reward, terminated, truncated, info = env.step(action)
    next_state = obs_to_state(obs)
    return next_state, obs


# =======================================
# TESTING FUNCTIONS
# =======================================

def test_single_step(env, action, verbose=True):
    """
    Test a single step: predict vs actual.
    
    Returns:
        Dict with prediction error metrics
    """
    # Get current state
    obs, _ = env.reset()
    current_state = obs_to_state(obs)
    
    # Predict next state using CBF dynamics
    predicted_state = predict_next_state(current_state, action)
    
    # Get actual next state from environment
    actual_state, next_obs = get_actual_next_state(env, action)
    
    # Compute errors
    error = actual_state - predicted_state
    abs_error = np.abs(error)
    rel_error = np.abs(error / (np.abs(actual_state) + 1e-8))
    
    if verbose:
        print(f"\nAction: {action}")
        print(f"Current state: {current_state}")
        print(f"Predicted next state: {predicted_state}")
        print(f"Actual next state:    {actual_state}")
        print(f"Absolute error:       {abs_error}")
        print(f"Relative error (%):   {rel_error * 100}")
        print(f"Max absolute error:   {np.max(abs_error):.6f}")
        print(f"Mean absolute error:  {np.mean(abs_error):.6f}")
    
    return {
        'action': action,
        'current_state': current_state,
        'predicted_state': predicted_state,
        'actual_state': actual_state,
        'error': error,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'max_abs_error': np.max(abs_error),
        'mean_abs_error': np.mean(abs_error)
    }


def test_multiple_actions(env, actions, verbose=True):
    """
    Test multiple actions and aggregate results.
    
    Args:
        env: Gym environment
        actions: List of actions to test
        verbose: Print individual results
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    print("=" * 80)
    print("TESTING MULTIPLE ACTIONS")
    print("=" * 80)
    
    for i, action in enumerate(actions):
        if verbose:
            print(f"\n--- Test {i+1}/{len(actions)} ---")
        
        result = test_single_step(env, action, verbose=verbose)
        results.append(result)
    
    return results


def test_trajectory(env, actions, seed=42, verbose=True):
    """
    Test a trajectory: apply multiple actions sequentially and compare predictions.
    
    Args:
        env: Gym environment
        actions: List of actions to apply sequentially
        seed: Random seed for reproducibility
        verbose: Print details
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    print("\n" + "=" * 80)
    print("TESTING TRAJECTORY (Sequential Actions)")
    print("=" * 80)
    
    # Reset with seed
    obs, _ = env.reset(seed=seed)
    
    for i, action in enumerate(actions):
        if verbose:
            print(f"\n--- Step {i+1}/{len(actions)} ---")
        
        # Get current state
        current_state = obs_to_state(obs)
        
        # Predict next state using CBF dynamics
        predicted_state = predict_next_state(current_state, action)
        
        # Get actual next state from environment
        obs, reward, terminated, truncated, info = env.step(action)
        actual_state = obs_to_state(obs)
        
        # Compute errors
        error = actual_state - predicted_state
        abs_error = np.abs(error)
        rel_error = np.abs(error / (np.abs(actual_state) + 1e-8))
        
        result = {
            'step': i,
            'action': action,
            'current_state': current_state,
            'predicted_state': predicted_state,
            'actual_state': actual_state,
            'error': error,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'max_abs_error': np.max(abs_error),
            'mean_abs_error': np.mean(abs_error)
        }
        results.append(result)
        
        if verbose:
            print(f"Action: {action}")
            print(f"Current state: {current_state}")
            print(f"Predicted next state: {predicted_state}")
            print(f"Actual next state:    {actual_state}")
            print(f"Absolute error:       {abs_error}")
            print(f"Max absolute error:   {np.max(abs_error):.6f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {i+1}")
            break
    
    return results


def print_summary_statistics(results):
    """Print summary statistics across all tests."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    max_errors = [r['max_abs_error'] for r in results]
    mean_errors = [r['mean_abs_error'] for r in results]
    
    print(f"\nAcross {len(results)} tests:")
    print(f"  Max absolute error:")
    print(f"    Mean:   {np.mean(max_errors):.6f}")
    print(f"    Std:    {np.std(max_errors):.6f}")
    print(f"    Min:    {np.min(max_errors):.6f}")
    print(f"    Max:    {np.max(max_errors):.6f}")
    print(f"\n  Mean absolute error:")
    print(f"    Mean:   {np.mean(mean_errors):.6f}")
    print(f"    Std:    {np.std(mean_errors):.6f}")
    print(f"    Min:    {np.min(mean_errors):.6f}")
    print(f"    Max:    {np.max(mean_errors):.6f}")
    
    # Per-component error analysis
    print(f"\n  Per-component mean absolute error:")
    all_abs_errors = np.array([r['abs_error'] for r in results])
    mean_abs_error_per_component = np.mean(all_abs_errors, axis=0)
    print(f"    theta0:      {mean_abs_error_per_component[0]:.6f}")
    print(f"    theta1:      {mean_abs_error_per_component[1]:.6f}")
    print(f"    theta0_dot:  {mean_abs_error_per_component[2]:.6f}")
    print(f"    theta1_dot:  {mean_abs_error_per_component[3]:.6f}")


def visualize_trajectory_comparison(results, save_path='trajectory_comparison.png'):
    """Visualize predicted vs actual trajectory."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping trajectory comparison plot (matplotlib not available)")
        return
    
    steps = [r['step'] for r in results]
    
    # Extract state components
    predicted_states = np.array([r['predicted_state'] for r in results])
    actual_states = np.array([r['actual_state'] for r in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CBF Dynamics Prediction vs Actual Environment', fontsize=14, fontweight='bold')
    
    state_names = ['theta0', 'theta1', 'theta0_dot', 'theta1_dot']
    
    for i, (ax, name) in enumerate(zip(axes.flat, state_names)):
        ax.plot(steps, predicted_states[:, i], 'b-o', label='CBF Predicted', markersize=4)
        ax.plot(steps, actual_states[:, i], 'r--s', label='Actual', markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Prediction vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTrajectory comparison plot saved to: {save_path}")
    plt.close()


def visualize_error_evolution(results, save_path='error_evolution.png'):
    """Visualize how prediction error evolves over trajectory."""
    if not HAS_MATPLOTLIB:
        print("Skipping error evolution plot (matplotlib not available)")
        return
    
    steps = [r['step'] for r in results]
    max_errors = [r['max_abs_error'] for r in results]
    mean_errors = [r['mean_abs_error'] for r in results]
    
    # Per-component errors
    abs_errors = np.array([r['abs_error'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Prediction Error Evolution', fontsize=14, fontweight='bold')
    
    # Overall errors
    ax = axes[0]
    ax.plot(steps, max_errors, 'r-o', label='Max Error', markersize=4)
    ax.plot(steps, mean_errors, 'b-s', label='Mean Error', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Overall Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-component errors
    ax = axes[1]
    state_names = ['theta0', 'theta1', 'theta0_dot', 'theta1_dot']
    for i, name in enumerate(state_names):
        ax.plot(steps, abs_errors[:, i], '-o', label=name, markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Per-Component Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error evolution plot saved to: {save_path}")
    plt.close()


# =======================================
# MAIN EXECUTION
# =======================================

def main():
    """Main testing function."""
    print("=" * 80)
    print("CBF DYNAMICS VERIFICATION FOR REACHER-V4")
    print("=" * 80)
    print("\nThis script verifies if the CBF dynamics formulation matches")
    print("the actual Reacher-v4 environment dynamics.\n")
    
    # Create environment
    env = gym.make('Reacher-v4')
    
    # Test 1: Various fixed actions from reset state
    print("\n" + "=" * 80)
    print("TEST 1: Fixed Actions from Reset State")
    print("=" * 80)
    
    test_actions = [
        np.array([0.0, 0.0]),    # No action
        np.array([1.0, 0.0]),    # Max action on joint 0
        np.array([0.0, 1.0]),    # Max action on joint 1
        np.array([1.0, 1.0]),    # Max action on both joints
        np.array([-1.0, 0.0]),   # Negative action on joint 0
        np.array([0.0, -1.0]),   # Negative action on joint 1
        np.array([0.5, 0.5]),    # Medium action
        np.array([-0.5, -0.5]),  # Medium negative action
        np.array([0.3, -0.7]),   # Mixed action
    ]
    
    results_fixed = test_multiple_actions(env, test_actions, verbose=True)
    print_summary_statistics(results_fixed)
    
    # Test 2: Sequential trajectory with random actions
    print("\n" + "=" * 80)
    print("TEST 2: Sequential Trajectory (Random Actions)")
    print("=" * 80)
    
    np.random.seed(42)
    n_steps = 20
    random_actions = [np.random.uniform(-1.0, 1.0, size=2) for _ in range(n_steps)]
    
    results_trajectory = test_trajectory(env, random_actions, seed=42, verbose=True)
    print_summary_statistics(results_trajectory)
    
    # Visualizations
    visualize_trajectory_comparison(results_trajectory, 'cbf_trajectory_comparison.png')
    visualize_error_evolution(results_trajectory, 'cbf_error_evolution.png')
    
    # Test 3: Multiple random initial states
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Random Initial States")
    print("=" * 80)
    
    results_random_states = []
    for i in range(10):
        env.reset(seed=100 + i)
        # Take a few random steps to get to a random state
        for _ in range(np.random.randint(5, 15)):
            env.step(env.action_space.sample())
        
        # Now test a fixed action from this random state
        test_action = np.array([0.5, 0.5])
        result = test_single_step(env, test_action, verbose=False)
        results_random_states.append(result)
        print(f"Test {i+1}/10: Max error = {result['max_abs_error']:.6f}")
    
    print_summary_statistics(results_random_states)
    
    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nKey Metrics:")
    print(f"  Test 1 (Fixed actions): Mean max error = {np.mean([r['max_abs_error'] for r in results_fixed]):.6f}")
    print(f"  Test 2 (Trajectory):    Mean max error = {np.mean([r['max_abs_error'] for r in results_trajectory]):.6f}")
    print(f"  Test 3 (Random states): Mean max error = {np.mean([r['max_abs_error'] for r in results_random_states]):.6f}")
    
    print("\nInterpretation:")
    print("  - Errors < 1e-3: Excellent match (likely correct dynamics)")
    print("  - Errors < 1e-2: Good match (minor discrepancies)")
    print("  - Errors > 1e-1: Poor match (dynamics formulation may be incorrect)")
    
    env.close()


if __name__ == "__main__":
    main()

