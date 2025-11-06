#!/usr/bin/env python3
"""Test CBF constraint logic to understand when corrections should happen."""

import numpy as np
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF

# Create CBF with x_max = 1.0
cbf = InvertedPendulumCBF(x_max=1.0)

print("=" * 80)
print("Testing CBF Constraint Logic")
print("=" * 80)

# Test cases: [x, x_dot, theta, theta_dot]
test_states = [
    ("Safe state (x=0.5)", np.array([0.5, 0.0, 0.0, 0.0])),
    ("Near boundary (x=0.9)", np.array([0.9, 0.0, 0.0, 0.0])),
    ("At boundary (x=1.0)", np.array([1.0, 0.0, 0.0, 0.0])),
    ("Unsafe (x=1.1)", np.array([1.1, 0.0, 0.0, 0.0])),
    ("Very unsafe (x=1.5)", np.array([1.5, 0.0, 0.0, 0.0])),
]

action = np.array([0.0])  # Neutral action

for name, state in test_states:
    h = cbf._evaluate_barrier(state)
    constraint = cbf._evaluate_constraint(state, action)
    
    print(f"\n{name}:")
    print(f"  State: x={state[0]:.2f}, x_dot={state[1]:.2f}")
    print(f"  Barrier h(x): {h:.4f}")
    print(f"  Constraint (ḣ + α*h): {constraint:.4f}")
    print(f"  Safe? {constraint >= 0.0}")
    print(f"  Should correct? {constraint < 0.0}")
    
    # Try correction
    if constraint < 0.0:
        certified, was_corrected = cbf.certify_action(state, action)
        print(f"  After correction: action {action[0]:.4f} -> {certified[0]:.4f}")
        new_constraint = cbf._evaluate_constraint(state, certified)
        print(f"  New constraint: {new_constraint:.4f} (safe: {new_constraint >= 0.0})")

print("\n" + "=" * 80)
print("Current code logic (WRONG):")
print("  if constraint_value >= 0.0:")
print("    certified_action = action  # Don't correct")
print("  else:")
print("    certified_action, was_corrected = cbf.certify_action(...)  # Correct")
print("\nThis is BACKWARDS! Should be:")
print("  if constraint_value < 0.0:")
print("    certified_action, was_corrected = cbf.certify_action(...)  # Correct when unsafe")
print("  else:")
print("    certified_action = action  # Don't correct when safe")
print("=" * 80)

