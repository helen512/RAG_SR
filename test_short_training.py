#!/usr/bin/env python3
"""Short training test to verify CBF statistics accumulation."""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
import numpy as np
import gymnasium as gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.append('/home/dmy/gymtest/safety-starter-agents')
from safe_rl import ppo

from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF
from cartpole2_safe_rl import CBFActionWrapper, ConstraintViolationCounter

SEED = 42
STEPS_PER_EPOCH = 1000
MAX_X_DISPLACEMENT = 1.0
RUN_DIR = "test_short_run"
os.makedirs(RUN_DIR, exist_ok=True)

print("=" * 80)
print("Short Training Test - 1 Epoch")
print("=" * 80)

# Create shared CBF and counter
shared_cbf = InvertedPendulumCBF(x_max=MAX_X_DISPLACEMENT)
counter = ConstraintViolationCounter(x_threshold=MAX_X_DISPLACEMENT)

def env_fn():
    env = gym.make('InvertedPendulum-v4')
    return CBFActionWrapper(
        env,
        shared_cbf,
        counter,
        use_corrected_action_for_training=True,
        reward_shaping_sigma=None,
        steps_per_epoch=STEPS_PER_EPOCH,
    )

logger_kwargs = {
    'output_dir': os.path.join(RUN_DIR, 'ppo_cbf_test'),
    'exp_name': 'ppo_cbf_short_test'
}

print("\nTraining for 1 epoch (1000 steps)...")
ppo(
    env_fn=env_fn,
    ac_kwargs=dict(hidden_sizes=(64, 64)),
    seed=SEED,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=1,  # Just 1 epoch!
    gamma=0.99,
    lam=0.95,
    target_kl=0.01,
    vf_lr=3e-4,
    vf_iters=80,
    logger_kwargs=logger_kwargs,
    save_freq=1
)

# Get statistics
stats = shared_cbf.get_stats()
counter_stats = counter.get_summary()

print("\n" + "=" * 80)
print("Results After 1 Epoch:")
print("=" * 80)
print("CBF Statistics:")
print(f"  Total actions processed: {stats['total_actions']}")
print(f"  Actions corrected: {stats['corrected_actions']}")
print(f"  Correction rate: {stats['correction_rate']:.3%}")
print(f"  Average correction: {stats['avg_correction']:.4f}")
print(f"  Max correction: {stats['max_correction']:.4f}")

print("\nViolation Statistics:")
print(f"  Total episodes: {counter_stats['total_episodes']}")
print(f"  Violation episodes: {counter_stats['violation_episodes']}")

print("\n" + "=" * 80)
if stats['total_actions'] >= 1000:
    print("✅ SUCCESS: CBF processed all ~1000 actions!")
    if stats['corrected_actions'] > 0:
        print("✅ SUCCESS: CBF made corrections!")
    else:
        print("⚠️  No corrections (might be normal if policy is conservative)")
else:
    print(f"❌ FAILURE: Only {stats['total_actions']} actions processed!")
print("=" * 80)



