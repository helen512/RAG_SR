#!/usr/bin/env python3
"""Detailed test of CBF with actual dynamics to understand why it's not preventing violations."""

import numpy as np
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF

# Create CBF with x_max = 1.0
cbf = InvertedPendulumCBF(x_max=1.0, alpha=1.0)

print("=" * 80)
print("Detailed CBF Testing - Understanding Lie Derivatives")
print("=" * 80)

# Test case: Cart moving toward boundary
test_cases = [
    ("Safe, stationary", np.array([0.5, 0.0, 0.0, 0.0])),
    ("Safe, moving right", np.array([0.5, 0.5, 0.0, 0.0])),  # Positive velocity
    ("Near boundary, moving right", np.array([0.9, 0.3, 0.0, 0.0])),
    ("Near boundary, moving left", np.array([0.9, -0.3, 0.0, 0.0])),
]

print("\nTest different actions for states approaching boundary:")
print("-" * 80)

for name, state in test_cases:
    h = cbf._evaluate_barrier(state)
    print(f"\n{name}:")
    print(f"  State: x={state[0]:.2f}, ẋ={state[1]:.2f}, θ={state[2]:.2f}, θ̇={state[3]:.2f}")
    print(f"  Barrier h(x) = {h:.4f}")
    
    # Test actions: push left, neutral, push right
    test_actions = [-3.0, 0.0, 3.0]
    action_names = ["Push LEFT (-3.0)", "Neutral (0.0)", "Push RIGHT (+3.0)"]
    
    for action, action_name in zip(test_actions, action_names):
        constraint = cbf._evaluate_constraint(state, np.array([action]))
        safe = constraint >= 0.0
        
        # Get certified action
        certified, was_corrected = cbf.certify_action(state, np.array([action]))
        constraint_after = cbf._evaluate_constraint(state, certified)
        
        print(f"    {action_name}:")
        print(f"      Original constraint: {constraint:.4f} (safe: {safe})")
        print(f"      Certified action: {certified[0]:.4f} (corrected: {was_corrected})")
        print(f"      After constraint: {constraint_after:.4f}")

print("\n" + "=" * 80)
print("Testing affine decomposition (L_f h + L_g h * u):")
print("-" * 80)

state = np.array([0.9, 0.3, 0.0, 0.0])  # Near boundary, moving right
h = cbf._evaluate_barrier(state)
lf_h, lg_h = cbf._compute_affine_terms(state)

print(f"\nState: x={state[0]:.2f}, ẋ={state[1]:.2f}")
print(f"Barrier h = {h:.4f}")
print(f"L_f h (drift term) = {lf_h:.4f}")
print(f"L_g h (control term) = {lg_h:.4f}")
print(f"\nConstraint: ḣ + α*h = (L_f h + L_g h * u) + α*h >= 0")
print(f"          = ({lf_h:.4f} + {lg_h:.4f} * u) + {1.0:.4f} * {h:.4f} >= 0")
print(f"          = ({lf_h:.4f} + {1.0 * h:.4f}) + {lg_h:.4f} * u >= 0")
print(f"          = {lf_h + 1.0 * h:.4f} + {lg_h:.4f} * u >= 0")

c = lf_h + 1.0 * h
if abs(lg_h) > 1e-8:
    u_threshold = -c / lg_h
    print(f"\nFor constraint to be satisfied: u >= {u_threshold:.4f}" if lg_h > 0 else f"For constraint to be satisfied: u <= {u_threshold:.4f}")
    print(f"Action limits: [{cbf.action_low:.1f}, {cbf.action_high:.1f}]")
    
    if lg_h > 0:
        safe_range = f"[{max(cbf.action_low, u_threshold):.2f}, {cbf.action_high:.1f}]"
    else:
        safe_range = f"[{cbf.action_low:.1f}, {min(cbf.action_high, u_threshold):.2f}]"
    print(f"Safe action range: {safe_range}")

print("\n" + "=" * 80)
print("Key Question: Can CBF prevent violations if L_g h ≈ 0?")
print("-" * 80)
print("If L_g h (control authority) is very small, the action has little effect")
print("on the constraint, making it hard to prevent violations.")
print("=" * 80)



