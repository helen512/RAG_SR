#!/usr/bin/env python3
"""Debug CBF projection to see why actions aren't being corrected."""

import numpy as np
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF

# Create CBF
cbf = InvertedPendulumCBF(x_max=1.0)

print("=" * 80)
print("CBF Projection Debugging")
print("=" * 80)

# Test case: x=0.95, moving right with velocity 0.5
# State format for CBF: [x, ẋ, θ, θ̇]
state = np.array([0.95, 0.5, 0.1, 0.0])  # Near boundary, moving right

print(f"\nTest State: x={state[0]:.2f}, ẋ={state[1]:.2f}, θ={state[2]:.2f}, θ̇={state[3]:.2f}")

# Check barrier and constraint
h = cbf._evaluate_barrier(state)
print(f"Barrier h(x, ẋ) = {h:.4f}")

# Test action: push right
action = 3.0
constraint_before = cbf._evaluate_constraint(state, np.array([action]))
print(f"\nProposed action: {action:.2f}")
print(f"Constraint value: {constraint_before:.4f} (safe: {constraint_before >= 0.0})")

# Get affine terms
lf_h, lg_h = cbf._compute_affine_terms(state)
print(f"\nAffine decomposition:")
print(f"  L_f h (drift): {lf_h:.4f}")
print(f"  L_g h (control): {lg_h:.4f}")

# Compute threshold
c = lf_h + cbf.alpha * h
print(f"  c = L_f h + α*h = {c:.4f}")

if abs(lg_h) > 1e-8:
    u_threshold = -c / lg_h
    print(f"  Threshold: u = -c / L_g h = {u_threshold:.4f}")
    
    if lg_h > 0:
        lower = max(cbf.action_low, u_threshold)
        upper = cbf.action_high
    else:
        lower = cbf.action_low
        upper = min(cbf.action_high, u_threshold)
    
    print(f"  Safe range: [{lower:.4f}, {upper:.4f}]")
    print(f"  Action {action:.2f} in safe range? {lower <= action <= upper}")
else:
    print(f"  L_g h ≈ 0: No control authority!")

# Call analytic projection
certified, feasible = cbf._analytic_projection(state, action)
print(f"\nAnalytic projection:")
print(f"  Certified action: {certified:.4f}")
print(f"  Feasible: {feasible}")

# Verify constraint after projection
constraint_after = cbf._evaluate_constraint(state, np.array([certified]))
print(f"  Constraint after: {constraint_after:.4f} (safe: {constraint_after >= 0.0})")

# If not feasible, try grid search
if not feasible:
    print(f"\nNot feasible! Trying grid search...")
    grid_certified = cbf._grid_search_fallback(state, action)
    constraint_grid = cbf._evaluate_constraint(state, np.array([grid_certified]))
    print(f"  Grid search action: {grid_certified:.4f}")
    print(f"  Constraint: {constraint_grid:.4f}")

# Test what certify_action returns
print(f"\n" + "=" * 80)
certified_array, was_corrected = cbf.certify_action(state, np.array([action]))
print(f"certify_action() returned:")
print(f"  Certified action: {certified_array[0]:.4f}")
print(f"  Was corrected: {was_corrected}")
print(f"  Changed? {not np.isclose(certified_array[0], action)}")

stats = cbf.get_stats()
print(f"\nStatistics:")
print(f"  Total actions: {stats['total_actions']}")
print(f"  Corrected: {stats['corrected_actions']}")
print(f"  Failed: {stats['failed_actions']}")
print("=" * 80)



