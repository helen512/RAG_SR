#!/usr/bin/env python3
"""Quick test to verify CBF statistics are tracked correctly."""

import gymnasium as gym
import numpy as np
from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF
from cartpole2_safe_rl import CBFActionWrapper, ConstraintViolationCounter

print("=" * 80)
print("Quick CBF Statistics Test")
print("=" * 80)

# Create shared CBF and counter
shared_cbf = InvertedPendulumCBF(x_max=1.0)
counter = ConstraintViolationCounter(x_threshold=1.0)

# Create environment
env = gym.make('InvertedPendulum-v4')
wrapped_env = CBFActionWrapper(env, shared_cbf, counter, steps_per_epoch=4000)

print("\nRunning 5 episodes to test statistics tracking...")

for episode in range(5):
    obs, info = wrapped_env.reset(seed=42 + episode)
    done = False
    steps = 0
    
    while not done and steps < 200:
        # Random action
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated
        steps += 1
        
        # Print when CBF corrects
        if info.get('cbf_corrected', False):
            print(f"  Episode {episode+1}, Step {steps}: CBF corrected action "
                  f"(x={obs[0]:.3f}, constraint={info.get('constraint_value', 0):.3f})")
    
    print(f"Episode {episode+1}: {steps} steps")

# Get statistics
stats = shared_cbf.get_stats()
print("\n" + "=" * 80)
print("CBF Statistics After 5 Episodes:")
print("=" * 80)
print(f"Total actions processed: {stats['total_actions']}")
print(f"Actions corrected: {stats['corrected_actions']}")
print(f"Correction rate: {stats['correction_rate']:.3%}")
print(f"Average correction magnitude: {stats['avg_correction']:.4f}")
print(f"Maximum correction magnitude: {stats['max_correction']:.4f}")
print(f"Failed actions: {stats['failed_actions']}")

counter_summary = counter.get_summary()
print("\nViolation Summary:")
print(f"Total episodes: {counter_summary['total_episodes']}")
print(f"Episodes with violations: {counter_summary['violation_episodes']}")

print("\n" + "=" * 80)
if stats['total_actions'] > 0:
    print("✅ SUCCESS: CBF statistics are being tracked!")
else:
    print("❌ FAILURE: CBF statistics are NOT being tracked!")
print("=" * 80)



