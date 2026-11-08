#!/usr/bin/env python3
"""
Stability check for trained InvertedPendulum PPO policies.
==========================================================

Evaluates the two policies trained in `runs/inverted_pendulum_ppo_paper/seed_42`:

  - ppo_standard_inverted_pendulum.zip  (standard +1/timestep reward)
  - ppo_custom_inverted_pendulum.zip    (shaped reward r = 1 - alpha*E)

For each policy a single deterministic episode is rolled out on a plain
InvertedPendulum-v4 environment (dynamics are identical for both rewards) using
a fixed reset seed. At every timestep we record:

  - the control value (the policy's action), and
  - theta, the pole angle (observation index 2).

Plots (each saved as its own PDF):
  - control value vs timestep  (line plot)   -- one per policy
  - theta value vs timestep    (dot plot)     -- one per policy

We also report quantitative stability metrics per policy:
  - RMS / std of theta            : steady-state upright deviation
  - control effort (RMS, total)   : actuation energy spent
  - control Sigma a^2             : sum of squared actions (quadratic effort)
  - mean |Delta a|                : average step-to-step action change (smoothness)
  - settling behaviour            : first time theta enters and stays within a
                                    +/- band, plus the max |theta| overshoot.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO

# -------------------- Configuration --------------------
POLICY_DIR = "runs/inverted_pendulum_ppo_paper/seed_42"
OUT_DIR = "runs/inverted_pendulum_stability_check"

POLICIES = {
    "standard": "ppo_standard_inverted_pendulum",
    "custom":   "ppo_custom_inverted_pendulum",
}
LABEL = {
    "standard": "Standard reward",
    "custom":   "Shaped reward (1 - E)",
}

RESET_SEED = 42_000          # fixed reset seed -> reproducible deterministic episode
THETA_IDX = 2                # observation = [x, x_dot, theta, theta_dot]
SETTLING_BAND = 0.01         # rad; theta must stay within +/- this to be "settled"


# -------------------- Rollout --------------------
def rollout(policy_key: str):
    """One deterministic episode. Returns (theta[t], control[t]) arrays."""
    model = PPO.load(os.path.join(POLICY_DIR, POLICIES[policy_key]))
    env = gym.make("InvertedPendulum-v4")
    obs, _ = env.reset(seed=RESET_SEED)

    thetas, controls = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        thetas.append(float(obs[THETA_IDX]))
        controls.append(float(np.ravel(action)[0]))
        obs, _reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    return np.asarray(thetas), np.asarray(controls)


# -------------------- Metrics --------------------
def settling_time(theta: np.ndarray, band: float):
    """First timestep index from which |theta| stays within band for the rest of
    the episode. Returns None if it never settles."""
    within = np.abs(theta) <= band
    n = len(theta)
    # scan from the end: last index where the tail is all-within
    settle = None
    for t in range(n):
        if np.all(within[t:]):
            settle = t
            break
    return settle


def compute_metrics(theta: np.ndarray, control: np.ndarray) -> dict:
    settle = settling_time(theta, SETTLING_BAND)
    return {
        "episode_length": len(theta),
        "theta_rms": float(np.sqrt(np.mean(theta**2))),
        "theta_std": float(np.std(theta)),
        "theta_abs_max": float(np.max(np.abs(theta))),       # overshoot
        "control_rms": float(np.sqrt(np.mean(control**2))),
        "control_abs_total": float(np.sum(np.abs(control))),
        "control_sq_total": float(np.sum(control**2)),        # Sigma a^2 effort
        "control_delta_abs_mean": float(np.mean(np.abs(np.diff(control))))  # mean |Delta a| smoothness
        if len(control) > 1 else 0.0,
        "settling_step": settle,                              # None if never
    }


# -------------------- Plots --------------------
def plot_control(policy_key: str, control: np.ndarray, path: str):
    plt.figure(figsize=(10, 4))
    plt.scatter(np.arange(len(control)), control, s=6, color="tab:blue")
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Control (action)")
    plt.title(f"Control vs timestep -- {LABEL[policy_key]}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def plot_theta(policy_key: str, theta: np.ndarray, path: str):
    plt.figure(figsize=(10, 4))
    plt.scatter(np.arange(len(theta)), theta, s=6, color="tab:red")
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.axhline(SETTLING_BAND, color="green", linewidth=0.8, linestyle=":",
                label=f"+/- {SETTLING_BAND} rad band")
    plt.axhline(-SETTLING_BAND, color="green", linewidth=0.8, linestyle=":")
    plt.xlabel("Timestep")
    plt.ylabel("theta (rad)")
    plt.title(f"theta vs timestep -- {LABEL[policy_key]}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# -------------------- Main --------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metric_rows = []

    for key in POLICIES:
        print(f"Rolling out {LABEL[key]} ...")
        theta, control = rollout(key)

        plot_control(key, control, os.path.join(OUT_DIR, f"control_{key}.pdf"))
        plot_theta(key, theta,   os.path.join(OUT_DIR, f"theta_{key}.pdf"))

        m = compute_metrics(theta, control)
        m["policy"] = key
        metric_rows.append(m)

    df = pd.DataFrame(metric_rows).set_index("policy")
    csv_path = os.path.join(OUT_DIR, "stability_metrics.csv")
    df.to_csv(csv_path)

    print("\n" + "=" * 64)
    print("STABILITY METRICS")
    print("=" * 64)
    for key, row in df.iterrows():
        settle = row["settling_step"]
        settle_str = "never" if pd.isna(settle) else f"{int(settle)}"
        print(f"\n{LABEL[key]}:")
        print(f"  episode length     : {int(row['episode_length'])}")
        print(f"  theta RMS / std    : {row['theta_rms']:.5f} / {row['theta_std']:.5f} rad")
        print(f"  theta |max| (overshoot): {row['theta_abs_max']:.5f} rad")
        print(f"  control RMS        : {row['control_rms']:.5f}")
        print(f"  control |total|    : {row['control_abs_total']:.3f}")
        print(f"  control Sigma a^2  : {row['control_sq_total']:.5f}")
        print(f"  mean |Delta a|     : {row['control_delta_abs_mean']:.5f}")
        print(f"  settling step (+/- {SETTLING_BAND} rad): {settle_str}")
    print(f"\nMetrics saved to: {csv_path}")
    print(f"Plots in: {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", default=POLICY_DIR,
                        help="Directory holding the trained .zip policies.")
    args = parser.parse_args()
    POLICY_DIR = args.policy_dir
    main()
