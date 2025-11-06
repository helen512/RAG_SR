"""Visualize the difference between continuous and discrete-time CBF constraints.

This script shows how the two different CBF formulations behave
when evaluating safety constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF


def compare_constraint_landscapes():
    """Compare continuous vs discrete CBF constraint landscapes."""
    
    # Create both CBF versions
    cbf_continuous = InvertedPendulumCBF(
        x_max=1.0,
        dt=0.02,
        alpha=0.3,
        use_discrete_cbf=False,
    )
    
    cbf_discrete = InvertedPendulumCBF(
        x_max=1.0,
        dt=0.02,
        kappa=0.1,
        use_discrete_cbf=True,
    )
    
    # Test on a grid of states near the boundary
    x_positions = np.linspace(0.5, 1.0, 20)
    x_velocities = np.linspace(-3.0, 3.0, 20)
    
    # Fixed angle and angular velocity (focus on cart position)
    theta = 0.0
    theta_dot = 0.0
    
    # Test action
    test_action = np.array([1.0])  # Positive force
    
    # Compute constraint values
    constraint_continuous = np.zeros((len(x_velocities), len(x_positions)))
    constraint_discrete = np.zeros((len(x_velocities), len(x_positions)))
    barrier_values = np.zeros((len(x_velocities), len(x_positions)))
    
    for i, x_vel in enumerate(x_velocities):
        for j, x_pos in enumerate(x_positions):
            state = np.array([x_pos, x_vel, theta, theta_dot])
            
            # Barrier value
            barrier_values[i, j] = cbf_continuous._evaluate_barrier(state)
            
            # Constraint values
            constraint_continuous[i, j] = cbf_continuous._evaluate_constraint(state, test_action)
            constraint_discrete[i, j] = cbf_discrete._evaluate_constraint(state, test_action)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Barrier function h(x)
    ax = axes[0, 0]
    contour = ax.contourf(x_positions, x_velocities, barrier_values, levels=20, cmap='RdYlGn')
    ax.contour(x_positions, x_velocities, barrier_values, levels=[0], colors='black', linewidths=2)
    ax.set_xlabel('Cart Position (x)')
    ax.set_ylabel('Cart Velocity (ẋ)')
    ax.set_title('Barrier Function h(x) = 1 - (x/x_max)²')
    ax.axvline(x=1.0, color='red', linestyle='--', label='Boundary x_max')
    ax.legend()
    plt.colorbar(contour, ax=ax, label='h(x)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Continuous-time CBF constraint
    ax = axes[0, 1]
    contour = ax.contourf(x_positions, x_velocities, constraint_continuous, 
                          levels=20, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.contour(x_positions, x_velocities, constraint_continuous, 
               levels=[0], colors='black', linewidths=2, label='Constraint = 0')
    ax.set_xlabel('Cart Position (x)')
    ax.set_ylabel('Cart Velocity (ẋ)')
    ax.set_title('Continuous-time CBF: ḣ + α·h ≥ 0')
    ax.axvline(x=1.0, color='red', linestyle='--', label='Boundary')
    ax.legend()
    plt.colorbar(contour, ax=ax, label='Constraint value')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Discrete-time CBF constraint
    ax = axes[1, 0]
    contour = ax.contourf(x_positions, x_velocities, constraint_discrete, 
                          levels=20, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.contour(x_positions, x_velocities, constraint_discrete, 
               levels=[0], colors='black', linewidths=2, label='Constraint = 0')
    ax.set_xlabel('Cart Position (x)')
    ax.set_ylabel('Cart Velocity (ẋ)')
    ax.set_title('Discrete-time CBF: h(x_{k+1}) - (1-κ)·h(x_k) ≥ 0')
    ax.axvline(x=1.0, color='red', linestyle='--', label='Boundary')
    ax.legend()
    plt.colorbar(contour, ax=ax, label='Constraint value')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Difference between constraints
    ax = axes[1, 1]
    difference = constraint_discrete - constraint_continuous
    contour = ax.contourf(x_positions, x_velocities, difference, 
                          levels=20, cmap='RdBu_r')
    ax.contour(x_positions, x_velocities, difference, 
               levels=[0], colors='black', linewidths=2)
    ax.set_xlabel('Cart Position (x)')
    ax.set_ylabel('Cart Velocity (ẋ)')
    ax.set_title('Difference: Discrete - Continuous')
    ax.axvline(x=1.0, color='red', linestyle='--', label='Boundary')
    ax.legend()
    plt.colorbar(contour, ax=ax, label='Δ Constraint')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cbf_constraint_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: cbf_constraint_comparison.png")
    plt.close()
    
    # Print analysis
    print("\n" + "="*80)
    print("Constraint Landscape Analysis")
    print("="*80)
    print(f"Test action: {test_action[0]:.2f} N")
    print(f"State space: x ∈ [{x_positions[0]:.2f}, {x_positions[-1]:.2f}], "
          f"ẋ ∈ [{x_velocities[0]:.2f}, {x_velocities[-1]:.2f}]")
    print()
    
    # Find where constraints differ most
    max_diff_idx = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
    max_diff = difference[max_diff_idx]
    max_diff_x = x_positions[max_diff_idx[1]]
    max_diff_v = x_velocities[max_diff_idx[0]]
    
    print(f"Maximum difference: {max_diff:.4f}")
    print(f"  at x={max_diff_x:.3f}, ẋ={max_diff_v:.3f}")
    print()
    
    # Count safe regions
    safe_continuous = np.sum(constraint_continuous >= 0)
    safe_discrete = np.sum(constraint_discrete >= 0)
    total = constraint_continuous.size
    
    print(f"Safe regions (constraint ≥ 0):")
    print(f"  Continuous-time: {safe_continuous}/{total} ({100*safe_continuous/total:.1f}%)")
    print(f"  Discrete-time:   {safe_discrete}/{total} ({100*safe_discrete/total:.1f}%)")
    print()
    
    if safe_discrete < safe_continuous:
        print("⚠ Discrete-time CBF is MORE CONSERVATIVE (smaller safe region)")
        print("  This is expected and provides better guarantees in discrete time.")
    elif safe_discrete > safe_continuous:
        print("✓ Discrete-time CBF is less conservative in this case")
    else:
        print("≈ Both constraints have similar safe regions")


def demonstrate_single_step():
    """Demonstrate how constraints differ on a single step."""
    
    print("\n" + "="*80)
    print("Single-Step Constraint Evaluation")
    print("="*80)
    
    # Create both CBF versions
    cbf_continuous = InvertedPendulumCBF(
        x_max=1.0,
        dt=0.02,
        alpha=0.3,
        use_discrete_cbf=False,
    )
    
    cbf_discrete = InvertedPendulumCBF(
        x_max=1.0,
        dt=0.02,
        kappa=0.1,
        use_discrete_cbf=True,
    )
    
    # Test scenario: Near boundary with high velocity
    state = np.array([0.9, 1.5, 0.0, 0.0])  # x=0.9, x_dot=1.5
    action = np.array([2.0])  # Push right
    
    print(f"State: x={state[0]:.2f}, ẋ={state[1]:.2f}")
    print(f"Action: u={action[0]:.2f} N")
    print()
    
    # Evaluate barrier
    h_k = cbf_continuous._evaluate_barrier(state)
    print(f"Barrier h(x_k) = {h_k:.4f}")
    print()
    
    # Evaluate constraints
    constraint_cont = cbf_continuous._evaluate_constraint(state, action)
    constraint_disc = cbf_discrete._evaluate_constraint(state, action)
    
    print("Constraint values:")
    print(f"  Continuous (ḣ + α·h):                 {constraint_cont:+.4f} {'✓ Safe' if constraint_cont >= 0 else '✗ Unsafe'}")
    print(f"  Discrete (h_next - (1-κ)·h_current): {constraint_disc:+.4f} {'✓ Safe' if constraint_disc >= 0 else '✗ Unsafe'}")
    print()
    
    # Predict next state
    h_next_pred = cbf_discrete.predict_next_barrier(state, action)
    print(f"Predicted h(x_{{k+1}}) = {h_next_pred:.4f}")
    print()
    
    # Show discrete-time breakdown
    print("Discrete-time constraint breakdown:")
    print(f"  h(x_{{k+1}})     = {h_next_pred:.4f}")
    print(f"  (1-κ)·h(x_k) = {(1-cbf_discrete.kappa)*h_k:.4f}")
    print(f"  Constraint   = {h_next_pred:.4f} - {(1-cbf_discrete.kappa)*h_k:.4f} = {constraint_disc:+.4f}")
    print()


if __name__ == "__main__":
    print("Visualizing CBF Constraint Comparison")
    print()
    
    # Run demonstrations
    demonstrate_single_step()
    compare_constraint_landscapes()
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)

