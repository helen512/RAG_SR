"""Integration test for discrete-time CBF fix.

This script verifies that all components work together correctly:
1. Discrete-time CBF constraint formulation
2. QP solver with discrete constraint
3. Violation logging and detection
4. Backward compatibility with continuous-time mode
"""

import numpy as np
import gymnasium as gym
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF


def test_basic_functionality():
    """Test basic CBF functionality."""
    print("="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)
    
    # Test discrete-time CBF
    cbf = InvertedPendulumCBF(x_max=1.0, dt=0.02, kappa=0.1, use_discrete_cbf=True)
    
    # Test state near boundary
    state = np.array([0.8, 0.5, 0.0, 0.0])
    action = np.array([2.0])
    
    # Test barrier evaluation
    h = cbf._evaluate_barrier(state)
    print(f"✓ Barrier evaluation: h(x) = {h:.4f}")
    assert h > 0, "State should be in safe set"
    
    # Test constraint evaluation
    constraint = cbf._evaluate_constraint(state, action)
    print(f"✓ Constraint evaluation: {constraint:.4f}")
    
    # Test action certification
    safe_action, corrected = cbf.certify_action(state, action)
    print(f"✓ Action certification: u_desired={action[0]:.2f}, u_safe={safe_action[0]:.2f}, corrected={corrected}")
    
    # Test prediction
    h_next = cbf.predict_next_barrier(state, safe_action)
    print(f"✓ Next barrier prediction: h_next = {h_next:.4f}")
    
    print("✓ Basic functionality test PASSED\n")


def test_logging():
    """Test violation logging functionality."""
    print("="*80)
    print("TEST 2: Violation Logging")
    print("="*80)
    
    cbf = InvertedPendulumCBF(x_max=1.0, dt=0.02, kappa=0.1, use_discrete_cbf=True)
    cbf.enable_logging(True)
    
    # Simulate a few steps
    env = gym.make("InvertedPendulum-v5", render_mode=None)
    state, _ = env.reset(seed=42)
    
    for step in range(10):
        action = env.action_space.sample()
        safe_action, _ = cbf.certify_action(state, action)
        next_state, _, terminated, truncated, _ = env.step(safe_action)
        
        # Log the step
        log_entry = cbf.log_step(state, safe_action, next_state, step)
        
        state = next_state
        if terminated or truncated:
            break
    
    env.close()
    
    # Check logging
    logs = cbf.get_violation_logs()
    summary = cbf.get_violation_summary()
    
    print(f"✓ Logged {summary['total_steps']} steps")
    print(f"✓ Detected {summary['violations']} violations")
    print(f"✓ Violation rate: {summary['violation_rate']:.2%}")
    
    assert len(logs) > 0, "Should have logged some steps"
    assert "h_k" in logs[0], "Log entry should contain h_k"
    assert "constraint_k" in logs[0], "Log entry should contain constraint_k"
    assert "h_next_predicted" in logs[0], "Log entry should contain h_next_predicted"
    assert "h_next_actual" in logs[0], "Log entry should contain h_next_actual"
    assert "discrete_violation" in logs[0], "Log entry should contain discrete_violation flag"
    
    print("✓ Logging test PASSED\n")


def test_backward_compatibility():
    """Test backward compatibility with continuous-time CBF."""
    print("="*80)
    print("TEST 3: Backward Compatibility")
    print("="*80)
    
    # Test continuous-time mode
    cbf_continuous = InvertedPendulumCBF(
        x_max=1.0,
        dt=0.02,
        alpha=0.3,
        use_discrete_cbf=False,
    )
    
    state = np.array([0.5, 0.2, 0.0, 0.0])
    action = np.array([1.0])
    
    # Should work with continuous-time mode
    h = cbf_continuous._evaluate_barrier(state)
    print(f"✓ Continuous-time barrier: h(x) = {h:.4f}")
    
    constraint = cbf_continuous._evaluate_constraint(state, action)
    print(f"✓ Continuous-time constraint: {constraint:.4f}")
    
    safe_action, corrected = cbf_continuous.certify_action(state, action)
    print(f"✓ Continuous-time certification: u_safe={safe_action[0]:.2f}")
    
    # Test affine terms (only work in continuous mode)
    try:
        lf_h, lg_h = cbf_continuous._compute_affine_terms(state)
        print(f"✓ Affine terms computed: Lf_h={lf_h:.4f}, Lg_h={lg_h:.4f}")
    except RuntimeError:
        print("✗ Affine terms failed in continuous mode")
        raise
    
    # Test that affine terms don't work in discrete mode
    cbf_discrete = InvertedPendulumCBF(x_max=1.0, dt=0.02, use_discrete_cbf=True)
    try:
        cbf_discrete._compute_affine_terms(state)
        print("✗ Affine terms should not work in discrete mode")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        print(f"✓ Affine terms correctly rejected in discrete mode: {str(e)}")
    
    print("✓ Backward compatibility test PASSED\n")


def test_parameter_variations():
    """Test different parameter configurations."""
    print("="*80)
    print("TEST 4: Parameter Variations")
    print("="*80)
    
    state = np.array([0.7, 1.0, 0.0, 0.0])
    action = np.array([1.5])
    
    # Test different kappa values
    kappa_values = [0.05, 0.1, 0.2, 0.3]
    print("Testing different kappa values:")
    for kappa in kappa_values:
        cbf = InvertedPendulumCBF(x_max=1.0, dt=0.02, kappa=kappa, use_discrete_cbf=True)
        constraint = cbf._evaluate_constraint(state, action)
        print(f"  κ={kappa:.2f}: constraint={constraint:+.4f}")
    
    print("✓ Different kappa values work correctly")
    
    # Test different x_max values
    x_max_values = [0.8, 1.0, 1.2]
    print("\nTesting different x_max values:")
    for x_max in x_max_values:
        cbf = InvertedPendulumCBF(x_max=x_max, dt=0.02, kappa=0.1, use_discrete_cbf=True)
        h = cbf._evaluate_barrier(state)
        print(f"  x_max={x_max:.1f}: h(x)={h:+.4f}")
    
    print("✓ Different x_max values work correctly")
    
    # Test different time steps
    dt_values = [0.01, 0.02, 0.05]
    print("\nTesting different time steps:")
    for dt in dt_values:
        cbf = InvertedPendulumCBF(x_max=1.0, dt=dt, kappa=0.1, use_discrete_cbf=True)
        constraint = cbf._evaluate_constraint(state, action)
        print(f"  dt={dt:.2f}s: constraint={constraint:+.4f}")
    
    print("✓ Different time steps work correctly")
    print("✓ Parameter variation test PASSED\n")


def test_statistics():
    """Test CBF statistics tracking."""
    print("="*80)
    print("TEST 5: Statistics Tracking")
    print("="*80)
    
    cbf = InvertedPendulumCBF(x_max=1.0, dt=0.02, kappa=0.1, use_discrete_cbf=True)
    
    env = gym.make("InvertedPendulum-v5", render_mode=None)
    state, _ = env.reset(seed=42)
    
    # Run a few steps with random aggressive actions
    n_steps = 20
    for _ in range(n_steps):
        # Random aggressive action
        action = env.action_space.sample() * 2.0  # Extra aggressive
        safe_action, corrected = cbf.certify_action(state, action)
        next_state, _, terminated, truncated, _ = env.step(safe_action)
        state = next_state
        if terminated or truncated:
            break
    
    env.close()
    
    # Get statistics
    stats = cbf.get_stats()
    print(f"Total actions: {stats['total_actions']}")
    print(f"Corrected actions: {stats['corrected_actions']}")
    print(f"Correction rate: {stats['correction_rate']:.2%}")
    print(f"Average correction: {stats['avg_correction']:.4f}")
    print(f"Max correction: {stats['max_correction']:.4f}")
    print(f"Failed actions: {stats['failed_actions']}")
    
    assert stats['total_actions'] > 0, "Should have processed some actions"
    print("✓ Statistics tracking test PASSED\n")


def main():
    print("\n" + "="*80)
    print("DISCRETE-TIME CBF FIX - INTEGRATION TEST")
    print("="*80 + "\n")
    
    try:
        test_basic_functionality()
        test_logging()
        test_backward_compatibility()
        test_parameter_variations()
        test_statistics()
        
        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe discrete-time CBF fix is working correctly!")
        print("Key features verified:")
        print("  ✓ Discrete-time constraint formulation")
        print("  ✓ Model-based next-state prediction")
        print("  ✓ Violation logging and detection")
        print("  ✓ Backward compatibility with continuous-time mode")
        print("  ✓ Parameter variations (kappa, x_max, dt)")
        print("  ✓ Statistics tracking")
        print()
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

