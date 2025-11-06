#!/usr/bin/env python3
"""Force cart near boundary to verify CBF triggers corrections."""

import gymnasium as gym
import numpy as np
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF
from cartpole2_safe_rl import CBFActionWrapper, ConstraintViolationCounter

print("=" * 80)
print("Forced Boundary Test - Verify CBF Corrections")
print("=" * 80)

# Create shared CBF and counter
shared_cbf = InvertedPendulumCBF(x_max=1.0)
counter = ConstraintViolationCounter(x_threshold=1.0)

# Create environment
env = gym.make('InvertedPendulum-v4')
wrapped_env = CBFActionWrapper(env, shared_cbf, counter, steps_per_epoch=4000)

print("\nManually testing states near boundary...")

# Create wrapper to manually set state
test_states = [
    ("Near boundary, moving right", np.array([0.85, 0.1, 0.3, 0.0, 0.0])),  # qpos=[0.85, 0.1], qvel=[0.3, 0.0]
    ("Near boundary, moving left", np.array([0.85, 0.1, -0.3, 0.0])),
    ("At boundary, moving right", np.array([0.95, 0.1, 0.5, 0.0])),
]

obs, info = wrapped_env.reset(seed=42)

for name, test_state in test_states:
    print(f"\n{name}:")
    print(f"  Test state (qpos, qvel): x={test_state[0]:.2f}, θ={test_state[1]:.2f}, "
          f"ẋ={test_state[2]:.2f}, θ̇={test_state[3]:.2f}")
    
    # Manually set the state
    env.unwrapped.set_state(test_state[:2], test_state[2:4])
    obs = env.unwrapped._get_obs()
    
    # Try pushing right (toward boundary)
    action = np.array([3.0])
    print(f"  Proposed action: {action[0]:.2f} (push right)")
    
    obs_new, reward, terminated, truncated, info = wrapped_env.step(action)
    
    print(f"  CBF corrected: {info.get('cbf_corrected', False)}")
    print(f"  Certified action: {info.get('certified_action', [0])[0]:.2f}")
    print(f"  Constraint value: {info.get('constraint_value', 0):.4f}")
    print(f"  New position: x={obs_new[0]:.3f}")

# Get statistics
stats = shared_cbf.get_stats()
print("\n" + "=" * 80)
print("CBF Statistics:")
print("=" * 80)
print(f"Total actions processed: {stats['total_actions']}")
print(f"Actions corrected: {stats['corrected_actions']}")
print(f"Correction rate: {stats['correction_rate']:.3%}")
print(f"Average correction magnitude: {stats['avg_correction']:.4f}")
print(f"Maximum correction magnitude: {stats['max_correction']:.4f}")

print("\n" + "=" * 80)
if stats['corrected_actions'] > 0:
    print("✅ SUCCESS: CBF is correcting actions near boundary!")
else:
    print("⚠️  WARNING: No corrections detected. Check if states were set correctly.")
print("=" * 80)



