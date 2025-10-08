# cartpole_rag_sr_ppo.py
# ------------------------------------------------------------
# End-to-end demo: Retrieval → Symbolic Regression → PPO on CartPole
# - Retrieval returns a seed reward expression + operator set
# - PySR discovers a compact symbolic expression (fallback: Ridge)
# - Train PPO (spinningup) on baseline reward vs symbolic reward; plot & save results
#
# Usage:
#   python cartpole_rag_llm_ppo.py
# ------------------------------------------------------------

import os
import re
import math
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import juliacall
from openai import OpenAI

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

SEED = 42
TOTAL_TIMESTEPS_BASE = 50_000  # PPO timesteps for training (base)
TOTAL_TIMESTEPS_SYM  = 50_000  # PPO timesteps for training (symbolic)
RUN_DIR = "runs_cartpole_llm_ppo"
os.makedirs(RUN_DIR, exist_ok=True)
set_random_seed(SEED)
save_index = 7

# ============================================================
# Haystack-powered retrieval for reward expressions
# (uses your haystack_query_pip.py hybrid engine)
# ============================================================
from typing import List, Dict, Any, Tuple
from pathlib import Path

from haystack_query_pip import (
        HybridQueryEngine,
        QuerySettings,
        build_dense_store,
        build_sparse_store,
    )


QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers")
QDRANT_PERSIST   = os.getenv("QDRANT_PERSIST", "./qdrant_papers")
BM25_CACHE_JSONL = Path(os.getenv("BM25_CACHE", "./bm25_cache.jsonl"))
EMBED_MODEL      = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL     = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

def _canon_ops_from_expr(expr: str) -> Tuple[List[str], List[str]]:
    """Heuristically infer allowed unary/binary operator sets from a string expression."""
    UN = []
    if "abs" in expr:  UN.append("abs")
    if "sin" in expr:  UN.append("sin")
    if "cos" in expr:  UN.append("cos")
    if "tanh" in expr: UN.append("tanh")
    BN = []
    for s in ["+","-","*","/"]:
        if s in expr: BN.append(s)
    if not UN: UN = ["abs","sin","cos"]  # safe defaults
    if not BN: BN = ["+","-","*"]
    return UN, BN

# Canonical variable aliases
_VAR_ALIASES = {
    "x": "x_n", "x_t": "x_n", "x_norm": "x_n",
    "xdot": "x_dot_n", "x_dot": "x_dot_n", "vx": "x_dot_n",
    "theta": "theta_n", "angle": "theta_n",
    "thetadot": "theta_dot_n", "theta_dot": "theta_dot_n", "omega": "theta_dot_n",
    "u": "u_n", "action": "u_n", "force": "u_n"
}

def _normalize_vars(expr: str) -> str:
    """Replace raw variable names with canonical forms."""
    out = expr
    for k in sorted(_VAR_ALIASES, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(k)}\b", _VAR_ALIASES[k], out, flags=re.IGNORECASE)
    return out

# Regex helpers for coefficients and powers
COEF = r"(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)"   # int, float, scientific notation
POW2 = r"(?:\*\*\s*2|\^\s*2)"               # **2 or ^2



def llm_generate_seed_expression(texts: List[str]) -> str:
    """Use OpenAI LLM to generate a seed reward expression from retrieved texts."""
    
    # Combine the retrieved texts into context
    context = "\n\n".join(texts[:5])  # Use top 5 retrieved documents
    
    prompt = f"""Based on the following research papers and documents about CartPole control, generate a physics energy-based positive mathematical expression for a reward function that encourages:
1. Keeping the pole upright (minimizing angle from vertical)
2. Keeping the cart centered (minimizing position)
3. Minimizing velocities for stability
4. The energy of the system is minimized when the pole is upright
Context from research papers:
{context}

The expression should:
- Use variables: x_n (normalized cart position), x_dot_n (normalized cart velocity), theta_n (normalized pole angle), theta_dot_n (normalized pole angular velocity), u_n (normalized action)
- Use only the variables mentioned above, no other variables like w1, w2, etc.
- Use Python syntax with ** for exponentiation
- Include functions like abs(), sin(), cos(), tanh() if appropriate
- Use wrap() function for angle wrapping if needed
- A physics energy-based positive expression
- Be a single mathematical expression that can be evaluated in Python

Return ONLY the mathematical expression, nothing else. 
"""

    try:
        # Initialize OpenAI client (using OPENAI_API_KEY environment variable)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key available")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert in control theory and reinforcement learning. Generate mathematical expressions for reward functions based on research context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=200
        )
        
        seed_expr = response.choices[0].message.content.strip()
        print("llm seed_expr:", seed_expr)
        seed_expr_path = os.path.join(RUN_DIR, f"seed_expr_{save_index}.txt")
        with open(seed_expr_path, "w") as f:
            f.write(seed_expr)
        
        # Basic validation and normalization
        if not seed_expr:
            raise ValueError("Empty response from LLM")
        

            
        # Normalize variable names using existing function
        seed_expr = _normalize_vars(seed_expr)
        
        return seed_expr
        
    except Exception as e:
        raise e
        # print(f"LLM generation failed: {e}")
        # print("Using fallback seed expression for testing...")
        # # Fallback expression for testing
        # seed_expr = "-(theta_n**2 + 0.1*theta_dot_n**2 + 0.1*x_n**2 + 0.01*x_dot_n**2 + 0.001*abs(u_n))"
        # seed_expr_path = os.path.join(RUN_DIR, f"seed_expr_{save_index}.txt")
        # with open(seed_expr_path, "w") as f:
        #     f.write(seed_expr)
        # return seed_expr 
       



def haystack_seed_reward(query_text: str):
    """
    Use Haystack hybrid retrieval to fetch CartPole reward hints, then use LLM to derive a seed expr,
    and infer operator sets for PySR.
    Returns: (seed_expr, UNARY_OPS, BINARY_OPS, hits[(source,score)])
    """

    qdrant = build_dense_store(QDRANT_COLLECTION, QDRANT_PERSIST)
    bm25   = build_sparse_store(BM25_CACHE_JSONL)
    engine = HybridQueryEngine(
        qdrant_store=qdrant,
        sparse_store=bm25,
        embedding_model=EMBED_MODEL,
        reranker_model=RERANK_MODEL,
        settings=QuerySettings(dense_k=24, sparse_k=60, final_k=12),
    )
    docs = engine.query(query_text)
    texts = []
    hits = []
    for d in docs:
        txt = (d.content or "")
        src = d.meta.get("source", "?") if getattr(d, "meta", None) else "?"
        sc  = float(getattr(d, "score", 0.0))
        hits.append((str(src), sc))
        texts.append(txt)
    
    # Use LLM to generate seed expression instead of pattern-based extraction
    seed_expr = llm_generate_seed_expression(texts)
    UN, BN = _canon_ops_from_expr(seed_expr)
    return seed_expr, UN, BN, hits



# ============================================================
# SR dataset (teacher = seed expression)
# ============================================================
def make_env(env_id='CartPole-v1', seed=SEED):
    """Create monitored environment for SB3"""
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def eval_expr(expr_str, x_n, x_dot_n, theta_n, theta_dot_n, u_n):
    safe = {"abs": abs, "sin": np.sin, "cos": np.cos, "tanh": np.tanh, "wrap": _wrap_angle}
    # Support both semantic names and PySR default feature names (x0..x4)
    locs = dict(
        x_n=x_n, x_dot_n=x_dot_n, theta_n=theta_n, theta_dot_n=theta_dot_n, u_n=u_n,
        x0=x_n, x1=x_dot_n, x2=theta_n, x3=theta_dot_n, x4=u_n,
    )
    locs.update(safe)
    return float(eval(expr_str, {"__builtins__": {}}, locs))

def sample_sr_dataset(expr_str, n=6000, seed=SEED):
    env = make_env('CartPole-v1', seed=seed)
    # thresholds from unwrapped env
    x_threshold = env.env.unwrapped.x_threshold
    theta_threshold = env.env.unwrapped.theta_threshold_radians
    obs, _ = env.reset()
    X, y = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        # random small state around stable upright
        x = rng.normal(0, 0.5)
        x_dot = rng.normal(0, 0.5)
        theta = rng.normal(0.0, 0.1)
        theta_dot = rng.normal(0.0, 0.5)
        u = rng.uniform(-1.0, 1.0)

        # normalize by thresholds to keep roughly in [-1, 1]
        x_n = x / x_threshold
        theta_n = _wrap_angle(theta) / theta_threshold
        x_dot_n = np.tanh(x_dot / 2.0)
        theta_dot_n = np.tanh(theta_dot / 3.0)
        u_n = float(u)

        X.append([x_n, x_dot_n, theta_n, theta_dot_n, u_n])
        y.append(eval_expr(expr_str, x_n, x_dot_n, theta_n, theta_dot_n, u_n))
    env.close()
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

# ============================================================
# Symbolic Regression (PySR → fallback Ridge)
# ============================================================
def fit_symbolic_reward(X, y, unary_ops=None, binary_ops=None, seed=SEED):
    unary_ops = unary_ops or ["abs", "sin", "cos"]
    binary_ops = binary_ops or ["+", "-", "*"]
    try:
        from pysr import PySRRegressor
        model = PySRRegressor(
            unary_operators=unary_ops,
            binary_operators=binary_ops,
            model_selection="best",
            niterations=60,
            populations=10,
            progress=False,
            random_state=seed,
            maxsize=20,
            elementwise_loss="L2DistLoss()",
        )
        model.fit(X, y)
        expr = model.get_best()["equation"]
        return str(expr)
    except Exception as e:
        print("PySR not available / failed, falling back to Ridge:", e)
        # Light fallback with polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xp = poly.fit_transform(X)
        reg = Ridge(alpha=1e-3, random_state=seed).fit(Xp, y)
        coeffs = reg.coef_
        terms = poly.get_feature_names_out(["x_n","x_dot_n","theta_n","theta_dot_n","u_n"])
        # Keep top 6 terms by magnitude
        idx = np.argsort(-np.abs(coeffs))[:6]
        expr_terms = [f"{coeffs[i]:+.4f}*({terms[i]})" for i in idx]
        expr = " ".join(expr_terms)
        return expr

# ============================================================
# Env wrapper to use discovered symbolic reward for DQN training
# ============================================================
class SymbolicRewardCartPole(gym.Wrapper):
    def __init__(self, env, expr_str):
        super().__init__(env)
        self.expr_str = expr_str
        self.safe = {"abs": abs, "sin": np.sin, "cos": np.cos, "tanh": np.tanh,
                     "wrap": lambda a: (a + np.pi) % (2*np.pi) - np.pi}

    def reward_from_obs(self, obs, action_value):
        x, x_dot, theta, theta_dot = obs
        x_threshold = self.env.unwrapped.x_threshold
        theta_threshold = self.env.unwrapped.theta_threshold_radians
        x_n = x / x_threshold
        theta_n = ((theta + np.pi) % (2*np.pi) - np.pi) / theta_threshold
        x_dot_n = np.tanh(x_dot / 2.0)
        theta_dot_n = np.tanh(theta_dot / 3.0)
        u_n = float(np.clip(action_value, -1.0, 1.0))
        # Support both semantic names and PySR default feature names (x0..x4)
        locs = dict(
            x_n=x_n, x_dot_n=x_dot_n, theta_n=theta_n, theta_dot_n=theta_dot_n, u_n=u_n,
            x0=x_n, x1=x_dot_n, x2=theta_n, x3=theta_dot_n, x4=u_n,
        )
        locs.update(self.safe)
        # try:
        #     r = float(eval(self.expr_str, {"__builtins__": {}}, locs))
        # except Exception:
        #     print("Error evaluating expression:", self.expr_str)
        #     r = 1.0 - (theta_n**2 + 0.1*theta_dot_n**2 + 0.1*x_n**2 + 0.05*x_dot_n**2 + 0.01*abs(u_n))
        # return r
        r = float(eval(self.expr_str, {"__builtins__": {}}, locs))
        # Better reward transformation: Hybrid approach
        # Keep the survival signal (+1 per step) and add small shaped bonus
        # This avoids reward hacking while providing useful gradients
        
        hybrid_reward = 1.0 + 0.5 * r  # Base survival + shaped bonus
        return hybrid_reward

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        # For CartPole-v1, actions are discrete {0,1}; map to [-1, 1] magnitude for shaping
        u = -1.0 if int(action) == 0 else 1.0
        r = self.reward_from_obs(obs, u)
        return obs, r, terminated, truncated, info

# ============================================================
# PPO training function using Stable Baselines3
# ============================================================
class EpisodicLoggerWithEarlyStop(BaseCallback):
    def __init__(self, early_stop_threshold=475, window_size=100):
        super().__init__()
        self.returns = []
        self.timesteps = []
        self.epoch_returns = []  # Mean return per epoch
        self.epoch_timesteps = []  # Timesteps per epoch
        self.current_epoch_returns = []  # Collect returns during current epoch
        self._ep_ret = 0.0
        self.early_stop_threshold = early_stop_threshold
        self.window_size = window_size
        self.early_stopped = False
        self.x_exceed_failures = 0  # Counter for episodes failing due to x exceeding threshold
        self.total_episodes = 0
        self.current_epoch = 0

    def _on_step(self) -> bool:
        if "episode" in self.locals.get("infos", [{}])[-1]:
            ep_info = self.locals["infos"][-1]["episode"]
            ep_return = ep_info["r"]
            self.returns.append(ep_return)
            self.timesteps.append(self.num_timesteps)
            self.current_epoch_returns.append(ep_return)
            self.total_episodes += 1
            
            # Check if episode failed due to x position exceeding threshold
            # Get the current observation to check the cart position
            obs = self.locals.get("new_obs", None)
            if obs is not None:
                # For CartPole, obs[0] is the cart position x
                x_pos = obs[0] if hasattr(obs, '__len__') and len(obs) > 0 else obs
                if hasattr(x_pos, '__len__'):
                    x_pos = x_pos[0]
                
                # Get the x_threshold from the environment
                env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
                if hasattr(env, 'env'):
                    env = env.env
                if hasattr(env, 'unwrapped'):
                    x_threshold = env.unwrapped.x_threshold
                else:
                    x_threshold = 2.4  # Default CartPole x_threshold
                
                # Check if x position exceeded the threshold
                if abs(x_pos) >= x_threshold:
                    self.x_exceed_failures += 1
            
            # Check for early stopping
            if len(self.returns) >= self.window_size:
                rolling_mean = np.mean(self.returns[-self.window_size:])
                if rolling_mean >= self.early_stop_threshold:
                    print(f"\nEarly stopping triggered! Rolling mean return ({rolling_mean:.2f}) >= {self.early_stop_threshold} over last {self.window_size} episodes")
                    self.early_stopped = True
                    return False  # Stop training
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each PPO rollout (epoch)"""
        if len(self.current_epoch_returns) > 0:
            mean_return = np.mean(self.current_epoch_returns)
            self.epoch_returns.append(mean_return)
            self.epoch_timesteps.append(self.num_timesteps)
            self.current_epoch_returns = []  # Reset for next epoch
            self.current_epoch += 1
    
    def get_x_failure_stats(self):
        """Return statistics about x-threshold failures"""
        if self.total_episodes == 0:
            return {"x_exceed_failures": 0, "total_episodes": 0, "x_failure_rate": 0.0}
        return {
            "x_exceed_failures": self.x_exceed_failures,
            "total_episodes": self.total_episodes,
            "x_failure_rate": self.x_exceed_failures / self.total_episodes
        }

def evaluate_policy_with_x_failures(model, env, n_eval_episodes=10, seed=None):
    """
    Custom evaluation function that tracks episodes failing due to x threshold exceeded.
    Returns mean_reward, std_reward, x_exceed_failures, total_episodes, x_failure_rate
    """
    if seed is not None:
        set_random_seed(seed)
    
    episode_rewards = []
    x_exceed_failures = 0
    total_episodes = 0
    
    # Get x_threshold from environment
    if hasattr(env, 'env'):
        unwrapped_env = env.env
    else:
        unwrapped_env = env
    if hasattr(unwrapped_env, 'unwrapped'):
        x_threshold = unwrapped_env.unwrapped.x_threshold
    else:
        x_threshold = 2.4  # Default CartPole x_threshold
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        total_episodes += 1
        episode_x_exceeded = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Check if x position exceeded threshold
            x_pos = obs[0]  # Cart position is first element in CartPole observation
            if abs(x_pos) >= x_threshold and not episode_x_exceeded:
                x_exceed_failures += 1
                episode_x_exceeded = True  # Count only once per episode
        
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    x_failure_rate = x_exceed_failures / total_episodes if total_episodes > 0 else 0.0
    
    return mean_reward, std_reward, x_exceed_failures, total_episodes, x_failure_rate

def build_ppo(env):
    """Build PPO model with optimal hyperparameters for CartPole"""
    return PPO(
        "MlpPolicy", 
        env, 
        seed=SEED, 
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
    )



# Main pipeline

if __name__ == "__main__":

    # ============================================================
    # Retrieval (Haystack) → seed expression + operator sets for PySR
    # ============================================================
    QUERY_TEXT = (
        "cartpole energy function: upright pole, "
        "Energy function of inverted pendulum,"
        "lyapunov function for cartpole stability"

    )
    SEED_EXPR, UNARY_OPS, BINARY_OPS, HITS = haystack_seed_reward(QUERY_TEXT)
    print("Seed expression (from Haystack or fallback):", SEED_EXPR)
    print("UNARY_OPS:", UNARY_OPS)
    print("BINARY_OPS:", BINARY_OPS)
    if HITS:
        print("Top retrieval hits (source, score):")
        for s, sc in HITS[:5]:
            print(f"  - {s} | {sc:.4f}")

    # 1) Build teacher dataset from the (retrieved) seed expression
    X, y = sample_sr_dataset(SEED_EXPR, n=6000, seed=SEED)

    # 2) Fit Symbolic Reward (PySR → fallback Ridge)
    SYM_EXPR_STR = fit_symbolic_reward(X, y, UNARY_OPS, BINARY_OPS, seed=SEED)
    print("\n=== Discovered symbolic reward ===")
    print(SYM_EXPR_STR)

    # 3) Train baseline PPO
    print("\nTraining baseline PPO...")
    env_base = make_env('CartPole-v1', SEED)
    logger_base = EpisodicLoggerWithEarlyStop()
    agent_base = build_ppo(env_base)
    agent_base.learn(total_timesteps=TOTAL_TIMESTEPS_BASE, callback=logger_base)
    if logger_base.early_stopped:
        print(f"Baseline PPO training stopped early after {len(logger_base.returns)} episodes")
    
    # Get training x-threshold failure stats
    base_train_stats = logger_base.get_x_failure_stats()
    print(f"Baseline training - X-threshold failures: {base_train_stats['x_exceed_failures']}/{base_train_stats['total_episodes']} ({base_train_stats['x_failure_rate']:.3f})")
    
    # Evaluate baseline model
    print("Evaluating baseline PPO...")
    eval_env_base = make_env('CartPole-v1', SEED + 1000)  # Different seed for evaluation
    base_eval_reward, base_eval_std, base_eval_x_failures, base_eval_episodes, base_eval_x_rate = evaluate_policy_with_x_failures(
        agent_base, eval_env_base, n_eval_episodes=100, seed=SEED + 1000
    )
    print(f"Baseline evaluation - Mean reward: {base_eval_reward:.2f} ± {base_eval_std:.2f}")
    print(f"Baseline evaluation - X-threshold failures: {base_eval_x_failures}/{base_eval_episodes} ({base_eval_x_rate:.3f})")
    eval_env_base.close()
    
    df_base = pd.DataFrame({'tag': 'baseline',
                           'timesteps': logger_base.timesteps,
                           'episodic_return': logger_base.returns})
    env_base.close()

    # 4) Train PPO on symbolic reward
    print("\nTraining symbolic PPO...")
    env_sym = make_env('CartPole-v1', SEED)
    env_sym = SymbolicRewardCartPole(env_sym, SYM_EXPR_STR)
    model_sym = build_ppo(env_sym)
    logger_sym = EpisodicLoggerWithEarlyStop()
    model_sym.learn(total_timesteps=TOTAL_TIMESTEPS_SYM, callback=logger_sym)
    if logger_sym.early_stopped:
        print(f"Symbolic PPO training stopped early after {len(logger_sym.returns)} episodes")
    
    # Get training x-threshold failure stats
    sym_train_stats = logger_sym.get_x_failure_stats()
    print(f"Symbolic training - X-threshold failures: {sym_train_stats['x_exceed_failures']}/{sym_train_stats['total_episodes']} ({sym_train_stats['x_failure_rate']:.3f})")
    
    # Evaluate symbolic model (use original CartPole reward for fair comparison)
    print("Evaluating symbolic PPO...")
    eval_env_sym = make_env('CartPole-v1', SEED + 1000)  # Different seed for evaluation
    sym_eval_reward, sym_eval_std, sym_eval_x_failures, sym_eval_episodes, sym_eval_x_rate = evaluate_policy_with_x_failures(
        model_sym, eval_env_sym, n_eval_episodes=100, seed=SEED + 1000
    )
    print(f"Symbolic evaluation - Mean reward: {sym_eval_reward:.2f} ± {sym_eval_std:.2f}")
    print(f"Symbolic evaluation - X-threshold failures: {sym_eval_x_failures}/{sym_eval_episodes} ({sym_eval_x_rate:.3f})")
    eval_env_sym.close()
    
    df_sym = pd.DataFrame({'tag': 'symbolic',
                          'timesteps': logger_sym.timesteps,
                          'episodic_return': logger_sym.returns})
    env_sym.close()

    # 5) Combine results for plotting
    df_all = pd.concat([df_base, df_sym], ignore_index=True)

    # Plot learning curves
    # ============================================================
    # Mean return per epoch (PPO rollout) - Appropriate for PPO
    plt.figure(figsize=(8,5))
    if len(logger_base.epoch_returns) > 0:
        epochs_base = np.arange(1, len(logger_base.epoch_returns) + 1)
        plt.plot(epochs_base, logger_base.epoch_returns, marker='o', label='baseline', alpha=0.7)
    if len(logger_sym.epoch_returns) > 0:
        epochs_sym = np.arange(1, len(logger_sym.epoch_returns) + 1)
        plt.plot(epochs_sym, logger_sym.epoch_returns, marker='s', label='symbolic', alpha=0.7)
    plt.xlabel("PPO Epoch (Rollout)")
    plt.ylabel("Mean Episode Return per Epoch")
    plt.title("CartPole: Baseline vs Symbolic Reward (PPO)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = os.path.join(RUN_DIR, f'learning_curves_ppo_{save_index}.png')
    plt.savefig(png_path, dpi=150)
    print('Saved:', png_path)
    plt.show()

    # ============================================================
    # Save CSV + discovered expression + summary table
    # ============================================================
    def auc(x, y):
        if len(x) < 2: return 0.0
        order = np.argsort(x)
        x, y = x[order], y[order]
        return np.trapz(y, x)

    rows = []
    for tag, df in df_all.groupby('tag'):
        ts = np.array(df['timesteps'])
        rs = np.array(df['episodic_return'])
        
        # Get the appropriate training and evaluation stats
        if tag == 'baseline':
            train_stats = base_train_stats
            eval_reward = base_eval_reward
            eval_x_failures = base_eval_x_failures
            eval_x_rate = base_eval_x_rate
        else:  # symbolic
            train_stats = sym_train_stats
            eval_reward = sym_eval_reward
            eval_x_failures = sym_eval_x_failures
            eval_x_rate = sym_eval_x_rate
        
        rows.append({
            'tag': tag,
            'episodes': int(len(rs)),
            'final_return': float(rs[-1] if len(rs)>0 else np.nan),
            'AUC': auc(ts, rs),
            'train_x_failures': train_stats['x_exceed_failures'],
            'train_x_failure_rate': train_stats['x_failure_rate'],
            'eval_mean_reward': eval_reward,
            'eval_x_failures': eval_x_failures,
            'eval_x_failure_rate': eval_x_rate
        })
    
    summary_df = pd.DataFrame(rows)
    print("\n=== Training Summary (PPO) ===")
    print(summary_df.round(4))

    csv_path = os.path.join(RUN_DIR, f'learning_curves_ppo_{save_index}.csv')
    expr_path = os.path.join(RUN_DIR, f'symbolic_expression_ppo_{save_index}.txt')
    summary_path = os.path.join(RUN_DIR, f'training_summary_ppo_{save_index}.csv')
    
    # Save episode-level data
    df_all.to_csv(csv_path, index=False)
    
    # Save epoch-level data (mean returns per PPO epoch)
    df_epochs_base = pd.DataFrame({
        'tag': 'baseline',
        'epoch': np.arange(1, len(logger_base.epoch_returns) + 1),
        'timesteps': logger_base.epoch_timesteps,
        'mean_return': logger_base.epoch_returns
    })
    df_epochs_sym = pd.DataFrame({
        'tag': 'symbolic',
        'epoch': np.arange(1, len(logger_sym.epoch_returns) + 1),
        'timesteps': logger_sym.epoch_timesteps,
        'mean_return': logger_sym.epoch_returns
    })
    df_epochs_all = pd.concat([df_epochs_base, df_epochs_sym], ignore_index=True)
    epoch_csv_path = os.path.join(RUN_DIR, f'learning_curves_ppo_epochs_{save_index}.csv')
    df_epochs_all.to_csv(epoch_csv_path, index=False)
    
    with open(expr_path, 'w') as f:
        f.write(SYM_EXPR_STR)
    summary_df.to_csv(summary_path, index=False)
    
    print('Saved episodes:', csv_path)
    print('Saved epochs:', epoch_csv_path)
    print('Saved:', expr_path)
    print('Saved:', summary_path)
    
    print("\n=== X-Threshold Failure Analysis ===")
    print("X-threshold failures occur when the cart position |x| >= 2.4 (CartPole default)")
    print("This is one of the terminal conditions that ends an episode in CartPole")
    print("Lower failure rates indicate better cart position control")
    print(f"Baseline - Training: {base_train_stats['x_failure_rate']:.3f}, Evaluation: {base_eval_x_rate:.3f}")
    print(f"Symbolic - Training: {sym_train_stats['x_failure_rate']:.3f}, Evaluation: {sym_eval_x_rate:.3f}")
        
    print("\n=== PPO Training Complete ===")
