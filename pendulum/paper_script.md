use openai inverted pendulum environment

running script with conda reg=index environment


the total energy of the inverted pendulum system is
\begin{align*}
    E = \frac{1}{2} \bigg[ (M + m)\dot{x}^2
        + m\,\dot{x}\,l\,\dot{\theta}
        + \frac{1}{3} m l^2 \dot{\theta}^2
        + m l g \big(1 - \cos(\theta)\big) \bigg],
\end{align*}
where $M$ is the mass of the cart, $m$ is the mass of the pole, and $l$ is the pole length. The potential energy is chosen to be zero when the pole is upright. To keep the pole stable and near the upright position, we want the system's total energy to remain small.

To encourage this, we introduce a physics-based reward-shaping term. The shaped reward is defined as
\begin{align}
    r(\bm{q}) = 1 -  E,
    \label{shaped reward}
\end{align}

We compare two reward designs in the inverted pendulum environment:
\begin{itemize}
    \item the standard reward (i.e., $+1$ per timestep when the pole stays within the allowed angle)
    \item the shaped reward defined in Eq.~\eqref{shaped reward}.
\end{itemize}

The task is consider success if the episode timesteps is longer than 950 timesteps.

To test the performace difference between different reward, we compare the mean success rates. After each ppo update, run evaluation over 100 episode and get the success rate. Each experiment is carried out across 5 random seeds and the shaded area represents the standard deviation. figure with success rate vs timesteps. 

after 200_000, take the policy and do evaluation across all 5 seeds, report success rate and std 

also report how many timesteps does policy takes to reate above 90% success rate. 

use ppo from stable baseline 3 for training, take the parameter from pendulum/cartpole_ppo2.py

## Implementation

Script: `pendulum/inverted_pendulum_ppo_paper.py`

- Env: `InvertedPendulum-v4` (MuJoCo, continuous action), matches `cartpole_ppo2.py`.
- Standard reward = env default (+1/timestep). Shaped reward = `1 - E` via `ShapedRewardInvertedPendulum` wrapper.
- Energy constants: `M=10.472`, `m=5.019`, `l=0.3`, `g=9.81`.
- PPO hyperparameters copied from `cartpole_ppo2.py` (`get_optimal_ppo_params`).
- Success = episode length strictly `> 950` timesteps.
- Evaluation: every `10_000` training steps, success rate over `100` episodes with fixed reset seeds. Termination dynamics are identical across both rewards, so survival length is measured on a plain `InvertedPendulum-v4` env.
- 5 seeds (42–46); plot shaded area = std across seeds.

Outputs (written to `runs/inverted_pendulum_ppo_standard_vs_energy_vs_effort_100_000/`):
- `success_rate_vs_timesteps.png` — mean success rate vs timesteps, both rewards, std shaded.
- `per_seed_success_rates.csv`, `aggregated_success_rates.csv`.
- `seed_<n>/ppo_{standard,custom}_inverted_pendulum.zip` — trained policies.
- Console: final 5-seed success rate ± std, and timesteps to first reach >90% success rate.

## Running

Use the conda `reg-index` environment.

Full experiment (5 seeds × 2 rewards, ~50 min):

```bash
python pendulum/inverted_pendulum_ppo_paper.py
```

Quick smoke test (1 seed, 4k steps, ~10 s) to verify the pipeline:

```bash
python pendulum/inverted_pendulum_ppo_paper.py --quick
```


# Stability check

create a new script, Evaluate policies in /home/dmy/gymtest/runs/inverted_pendulum_ppo_paper/seed_42, track the control and value of theta for each timestep. plot the control value vs timestpes for each policies, and theta value vs timesteps dot plot for each policies. different policies on different plot, save plot as pdf file. 

