"""Test script to verify discrete-time CBF fixes the forward invariance issue.

This script compares:
1. Continuous-time CBF (old, broken) - shows violations
2. Discrete-time CBF (new, fixed) - no violations

The test follows the instructions from the issue:
- Log per step: h_k, constraint_k, predicted h_{k+1}, actual h_{k+1}
- Flag any step where constraint_k >= 0 but h_{k+1} < 0 as "discrete-violation"
"""

import numpy as np
import gymnasium as gym
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF


def test_cbf_violations(use_discrete_cbf: bool, n_episodes: int = 5, max_steps: int = 500):
    """Test for CBF violations.
    
    Args:
        use_discrete_cbf: If True, use discrete-time CBF; else continuous-time
        n_episodes: Number of test episodes
        max_steps: Maximum steps per episode
    
    Returns:
        dict with violation statistics
    """
    env = gym.make("InvertedPendulum-v5", render_mode=None)
    
    # Create CBF with specified mode
    cbf = InvertedPendulumCBF(
        x_max=1.0,
        dt=0.02,
        alpha=0.3,
        kappa=0.1,
        use_discrete_cbf=use_discrete_cbf,
    )
    
    # Enable logging
    cbf.enable_logging(True)
    
    total_steps = 0
    total_violations = 0
    violation_details = []
    
    for episode in range(n_episodes):
        state, _ = env.reset(seed=42 + episode)
        
        for step in range(max_steps):
            # Random action (to stress-test the CBF)
            random_action = env.action_space.sample()
            
            # Certify action with CBF
            safe_action, corrected = cbf.certify_action(state, random_action)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(safe_action)
            
            # Log the step and check for violations
            log_entry = cbf.log_step(state, safe_action, next_state, step)
            
            if log_entry["discrete_violation"]:
                total_violations += 1
                violation_details.append({
                    "episode": episode,
                    "step": step,
                    "state_x": log_entry["state_x"],
                    "state_x_dot": log_entry["state_x_dot"],
                    "h_k": log_entry["h_k"],
                    "constraint_k": log_entry["constraint_k"],
                    "h_next_predicted": log_entry["h_next_predicted"],
                    "h_next_actual": log_entry["h_next_actual"],
                })
            
            total_steps += 1
            state = next_state
            
            if terminated or truncated:
                break
    
    env.close()
    
    violation_rate = total_violations / total_steps if total_steps > 0 else 0
    
    return {
        "mode": "Discrete-time CBF" if use_discrete_cbf else "Continuous-time CBF",
        "total_steps": total_steps,
        "total_violations": total_violations,
        "violation_rate": violation_rate,
        "violation_details": violation_details[:10],  # Show first 10
        "cbf_stats": cbf.get_stats(),
    }


def main():
    print("=" * 80)
    print("Testing CBF Forward Invariance Fix")
    print("=" * 80)
    print()
    
    # Test 1: Continuous-time CBF (old, should show violations)
    print("TEST 1: Continuous-time CBF (old approach)")
    print("-" * 80)
    results_continuous = test_cbf_violations(use_discrete_cbf=False, n_episodes=5)
    print(f"Mode: {results_continuous['mode']}")
    print(f"Total steps: {results_continuous['total_steps']}")
    print(f"Total violations: {results_continuous['total_violations']}")
    print(f"Violation rate: {results_continuous['violation_rate']:.4f}")
    print(f"CBF corrections: {results_continuous['cbf_stats']['corrected_actions']} / {results_continuous['cbf_stats']['total_actions']}")
    
    if results_continuous['violation_details']:
        print("\nSample violations:")
        for i, v in enumerate(results_continuous['violation_details'][:3]):
            print(f"  Violation {i+1}: Episode {v['episode']}, Step {v['step']}")
            print(f"    x={v['state_x']:.4f}, x_dot={v['state_x_dot']:.4f}")
            print(f"    h_k={v['h_k']:.4f}, constraint_k={v['constraint_k']:.4f}")
            print(f"    h_next_pred={v['h_next_predicted']:.4f}, h_next_actual={v['h_next_actual']:.4f}")
    
    print("\n")
    
    # Test 2: Discrete-time CBF (new, should have zero or minimal violations)
    print("TEST 2: Discrete-time CBF (new approach - FIXED)")
    print("-" * 80)
    results_discrete = test_cbf_violations(use_discrete_cbf=True, n_episodes=5)
    print(f"Mode: {results_discrete['mode']}")
    print(f"Total steps: {results_discrete['total_steps']}")
    print(f"Total violations: {results_discrete['total_violations']}")
    print(f"Violation rate: {results_discrete['violation_rate']:.4f}")
    print(f"CBF corrections: {results_discrete['cbf_stats']['corrected_actions']} / {results_discrete['cbf_stats']['total_actions']}")
    
    if results_discrete['violation_details']:
        print("\nSample violations:")
        for i, v in enumerate(results_discrete['violation_details'][:3]):
            print(f"  Violation {i+1}: Episode {v['episode']}, Step {v['step']}")
            print(f"    x={v['state_x']:.4f}, x_dot={v['state_x_dot']:.4f}")
            print(f"    h_k={v['h_k']:.4f}, constraint_k={v['constraint_k']:.4f}")
            print(f"    h_next_pred={v['h_next_predicted']:.4f}, h_next_actual={v['h_next_actual']:.4f}")
    
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Continuous-time CBF violations: {results_continuous['total_violations']} ({results_continuous['violation_rate']:.2%})")
    print(f"Discrete-time CBF violations: {results_discrete['total_violations']} ({results_discrete['violation_rate']:.2%})")
    
    if results_discrete['total_violations'] < results_continuous['total_violations']:
        reduction = (results_continuous['total_violations'] - results_discrete['total_violations'])
        print(f"\n✓ SUCCESS: Discrete-time CBF reduced violations by {reduction} steps!")
    elif results_discrete['total_violations'] == 0:
        print(f"\n✓ PERFECT: Discrete-time CBF has ZERO violations!")
    else:
        print(f"\n⚠ NOTE: Some violations may remain due to model mismatch with environment.")
    
    print()


if __name__ == "__main__":
    main()

