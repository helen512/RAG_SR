#!/usr/bin/env python3
"""Debug the wrapper to see what's happening with state reordering."""

import gymnasium as gym
import numpy as np
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF
from cartpole2_safe_rl import CBFActionWrapper, ConstraintViolationCounter

# Create shared CBF and counter
shared_cbf = InvertedPendulumCBF(x_max=1.0)
counter = ConstraintViolationCounter(x_threshold=1.0)

# Create environment
env = gym.make('InvertedPendulum-v4')
wrapped_env = CBFActionWrapper(env, shared_cbf, counter, steps_per_epoch=4000)

print("=" * 80)
print("Wrapper State Reordering Debug")
print("=" * 80)

# Reset and get initial state
obs, info = wrapped_env.reset(seed=42)
print(f"\nAfter reset:")
print(f"  Gym obs: {obs} (shape: {obs.shape})")
print(f"  Order: [x, θ, ẋ, θ̇]")
print(f"  wrapped_env._last_obs: {wrapped_env._last_obs}")

# Manually set a state near boundary
print(f"\nManually setting state to x=0.95, ẋ=0.5...")
test_qpos = np.array([0.95, 0.1])  # [x, theta]
test_qvel = np.array([0.5, 0.0])    # [x_dot, theta_dot]
env.unwrapped.set_state(test_qpos, test_qvel)

# Get observation
obs = env.unwrapped._get_obs()
print(f"  Gym obs after set_state: {obs}")
print(f"  x={obs[0]:.3f}, θ={obs[1]:.3f}, ẋ={obs[2]:.3f}, θ̇={obs[3]:.3f}")

# Reorder for CBF
cbf_state = wrapped_env._reorder_for_cbf(obs)
print(f"  CBF state after reordering: {cbf_state}")
print(f"  Expected: [x, ẋ, θ, θ̇] = [0.950, 0.500, 0.100, 0.000]")

# Test action
action = np.array([3.0])
print(f"\nTesting action={action[0]:.2f}...")

# Manually call CBF
constraint = shared_cbf._evaluate_constraint(cbf_state, action)
print(f"  Constraint value: {constraint:.4f}")

certified, corrected = shared_cbf.certify_action(cbf_state, action)
print(f"  Certified action: {certified[0]:.4f}")
print(f"  Corrected: {corrected}")

# Now do a full step through wrapper
print(f"\nDoing full step through wrapper...")

# Reset the CBF state counter
shared_cbf.reset_stats()

# Set state again
env.unwrapped.set_state(test_qpos, test_qvel)
obs = env.unwrapped._get_obs()
wrapped_env._last_obs = wrapped_env._reorder_for_cbf(obs)

obs_new, reward, terminated, truncated, info = wrapped_env.step(action)

print(f"  Info['cbf_corrected']: {info.get('cbf_corrected', False)}")
print(f"  Info['certified_action']: {info.get('certified_action', [0])}")
print(f"  Info['uncertified_action']: {info.get('uncertified_action', [0])}")
print(f"  Info['constraint_value']: {info.get('constraint_value', 0):.4f}")

stats = shared_cbf.get_stats()
print(f"\nCBF Statistics:")
print(f"  Total actions: {stats['total_actions']}")
print(f"  Corrected: {stats['corrected_actions']}")
print(f"  Rate: {stats['correction_rate']:.3%}")

print("=" * 80)



