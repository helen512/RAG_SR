#!/usr/bin/env python3
"""Train PPO on `InvertedPendulum-v4` with a CBF action safety filter.

This script mirrors the structure of `safe_rl_cartpole_comparison.py` but relies
exclusively on modules inside `model_base_env` for the safety logic.  It trains a
standard PPO agent (Stable-Baselines3) on the Gymnasium inverted pendulum while
wrapping the environment with a Control Barrier Function (CBF) filter that
enforces a cart-position constraint ``|x| <= x_max``.

Run with::

    conda activate safe
    python inverted_pendulum_ppo_cbf.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from model_base_env.inverted_pendulum_cbf import InvertedPendulumCBF


# ---------------------------------------------------------------------------
# Constraint tracking utilities
# ---------------------------------------------------------------------------


@dataclass
class ConstraintViolationCounter:
    x_threshold: float

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_timesteps = 0
        self.violation_timesteps = 0
        self.total_episodes = 0
        self.violation_episodes = 0
        self.current_episode_steps = 0
        self.current_episode_had_violation = False
        self.violations_per_episode = []

    def start_episode(self) -> None:
        self.current_episode_steps = 0
        self.current_episode_had_violation = False

    def step(self, obs: np.ndarray) -> bool:
        self.total_timesteps += 1
        self.current_episode_steps += 1
        violated = bool(abs(float(obs[0])) > self.x_threshold)
        if violated:
            self.violation_timesteps += 1
            self.current_episode_had_violation = True
        return violated

    def end_episode(self) -> None:
        self.total_episodes += 1
        if self.current_episode_had_violation:
            self.violation_episodes += 1
        self.violations_per_episode.append(self.current_episode_had_violation)
        self.current_episode_steps = 0
        self.current_episode_had_violation = False

    def summary(self) -> Dict[str, float]:
        if self.total_episodes == 0:
            episode_violation_rate = 0.0
        else:
            episode_violation_rate = self.violation_episodes / self.total_episodes

        if self.total_timesteps == 0:
            timestep_violation_rate = 0.0
        else:
            timestep_violation_rate = self.violation_timesteps / self.total_timesteps

        return {
            "episodes": self.total_episodes,
            "violation_episodes": self.violation_episodes,
            "episode_violation_rate": episode_violation_rate,
            "timesteps": self.total_timesteps,
            "violation_timesteps": self.violation_timesteps,
            "timestep_violation_rate": timestep_violation_rate,
        }


# ---------------------------------------------------------------------------
# Environment wrapper powering the CBF filter
# ---------------------------------------------------------------------------


class CBFActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, cbf_filter: InvertedPendulumCBF, counter: ConstraintViolationCounter):
        super().__init__(env)
        self.cbf_filter = cbf_filter
        self.counter = counter
        self._last_obs = None

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
        else:  # pragma: no cover - legacy API
            obs, info = result, {}
        self._last_obs = obs
        self.counter.start_episode()
        return obs, info

    def step(self,action):
        uncertified = np.asarray(action, dtype=np.float64)
        if self._evaluate_constraint(self._last_obs, uncertified) < 0.0:
            certified_action = action
            was_corrected = False
        else:   
            certified_action, was_corrected = self.cbf_filter.certify_action(self._last_obs, uncertified)

        step_result = self.env.step(certified_action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:  # pragma: no cover - legacy API
            obs, reward, done, info = step_result
            terminated, truncated = done, False

        violated = self.counter.step(obs)

        info = dict(info)
        info.update(
            {
                "cbf_corrected": bool(was_corrected),
                "uncertified_action": uncertified,
                "certified_action": certified_action,
                "constraint_violated": violated,
            }
        )

        if terminated or truncated:
            self.counter.end_episode()

        self._last_obs = obs
        return obs, reward, terminated, truncated, info


def make_cbf_env(x_max: float = 1.5) -> gym.Env:
    base_env = gym.make("InvertedPendulum-v4")
    cbf = InvertedPendulumCBF(x_max=x_max)
    counter = ConstraintViolationCounter(x_threshold=x_max)
    return CBFActionWrapper(base_env, cbf, counter)


# ---------------------------------------------------------------------------
# PPO training & evaluation helpers
# ---------------------------------------------------------------------------


SEED = 42
TOTAL_TIMESTEPS = 150_000
EVAL_EPISODES = 20
RUN_DIR = "runs_inverted_pendulum_cbf"


def evaluate_policy(
    model: PPO,
    env: CBFActionWrapper,
    episodes: int,
    seed: int | None = None,
) -> Tuple[float, float, Dict[str, float]]:
    returns = []
    lengths = []
    counter: ConstraintViolationCounter = env.counter

    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        obs, info = env.reset(seed=episode_seed)
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            length += 1
        returns.append(total_reward)
        lengths.append(length)

    summary = counter.summary()
    return float(np.mean(returns)), float(np.std(returns)), summary


def main() -> None:
    os.makedirs(RUN_DIR, exist_ok=True)

    # Vectorised environment setup for PPO.
    vec_env = DummyVecEnv([lambda: make_cbf_env()])
    vec_env.seed(SEED)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=SEED,
        tensorboard_log=os.path.join(RUN_DIR, "tensorboard"),
    )

    print("=" * 80)
    print("Training PPO on InvertedPendulum-v4 with CBF safety filter")
    print("=" * 80)
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Constraint: |x| <= {vec_env.envs[0].cbf_filter.x_max}")
    print()

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    cbf_stats = vec_env.envs[0].cbf_filter.get_stats()
    violation_summary = vec_env.envs[0].counter.summary()

    print("\nTraining complete!")
    print("CBF statistics:")
    for key, value in cbf_stats.items():
        print(f"  {key}: {value}")

    print("\nConstraint violation summary during training:")
    for key, value in violation_summary.items():
        print(f"  {key}: {value}")

    # Save model and statistics.
    model_path = os.path.join(RUN_DIR, "ppo_inverted_pendulum_cbf.zip")
    model.save(model_path)
    print(f"\nSaved trained model to: {model_path}")

    stats_path = os.path.join(RUN_DIR, "cbf_training_stats.npz")
    np.savez(
        stats_path,
        cbf_stats=cbf_stats,
        training_violation_summary=violation_summary,
    )
    print(f"Saved statistics to: {stats_path}")

    # Evaluate deterministic policy with safety filter enabled.
    eval_env: CBFActionWrapper = make_cbf_env()
    mean_return, std_return, eval_summary = evaluate_policy(
        model, eval_env, EVAL_EPISODES, seed=SEED + 1
    )

    print("\nEvaluation over", EVAL_EPISODES, "episodes:")
    print(f"  Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print("  Constraint summary:")
    for key, value in eval_summary.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()


