import os
import re
import math
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import juliacall

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

SEED = 42
TOTAL_STEPS = 100_000        # Increase to 300_000+ for stronger results
RUN_DIR = "runs_cartpole_test"
os.makedirs(RUN_DIR, exist_ok=True)
set_random_seed(SEED)
save_index = 4

class CurveLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.timesteps = []
        self.returns = []
    def _on_step(self) -> bool:
        info = self.locals.get('infos', [{}])[-1]
        if 'episode' in info:
            self.timesteps.append(self.num_timesteps)
            self.returns.append(info['episode']['r'])
        return True

def make_env_mon(seed=SEED):
    e = gym.make("CartPole-v1")
    e = Monitor(e)
    e.reset(seed=seed)
    return e

def build_dqn(env):
    return DQN(
        "MlpPolicy", env, seed=SEED, verbose=0,
        learning_rate=2.3e-3,             # Zoo
        buffer_size=100_000,              # Zoo
        learning_starts=1_000,            # Zoo
        batch_size=64,                    # Zoo
        gamma=0.99,                       # Zoo
        train_freq=256,                   # Zoo (collect 256 steps, then update)
        gradient_steps=128,               # Zoo (do 128 updates)
        target_update_interval=100,        # Zoo (frequent target syncs)
        exploration_fraction=0.16,        # Zoo
        exploration_final_eps=0.001,       # Zoo
        policy_kwargs=dict(net_arch=[256, 256]),  # Zoo
        replay_buffer_kwargs=dict(handle_timeout_termination=True),
        # IMPORTANT: do NOT set optimize_memory_usage=True with the above
    )

# Baseline
env_base = make_env_mon(SEED)
logger_base = CurveLogger()
agent_base = build_dqn(env_base)
print("Training baseline DQN...")
agent_base.learn(total_timesteps=100_000, callback=logger_base)  # Zoo budget
df_base = pd.DataFrame({'tag':'baseline',
                        'timesteps':logger_base.timesteps,
                        'episodic_return':logger_base.returns})
env_base.close()


def moving_avg(x, w=10):
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure(figsize=(8,5))
for tag, df in df_base.groupby('tag'):
    ts = np.array(df['timesteps'])
    rs = np.array(df['episodic_return'])
    # smooth for display
    if len(rs) > 5:
        rs_s = moving_avg(rs, w=min(25, max(5, len(rs)//10)))
        ts_s = ts[-len(rs_s):]
    else:
        rs_s = rs
        ts_s = ts
    plt.plot(ts_s, rs_s, label=tag)
plt.xlabel("Timesteps")
plt.ylabel("Episode Return")
plt.title("CartPole: Baseline vs Symbolic Reward")
plt.legend()
plt.tight_layout()

png_path = os.path.join(RUN_DIR, f'learning_curves_test_{save_index}.png')
plt.savefig(png_path, dpi=150)
print('Saved:', png_path)

plt.figure(figsize=(8,5))
for tag, df in df_base.groupby('tag'):
    t = df['timesteps'].values
    r = df['episodic_return'].values
    r_s = moving_avg(r, w=10)
    # t_s = t[-len(r_s):]
    episode_idx = np.arange(1, len(r) + 1)
    t_s = episode_idx[-len(r_s):]
    plt.plot(t_s, r_s, label=tag)
# plt.xlabel('Timesteps')
plt.xlabel('Episode')
plt.ylabel('Episodic Return (smoothed)')
plt.title('CartPole: Baseline vs Symbolic-Reward DQN')
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
png_path = os.path.join(RUN_DIR, f'learning_curves_episode_test_{save_index}.png')
plt.savefig(png_path, dpi=140)
print("Saved plot:", png_path)