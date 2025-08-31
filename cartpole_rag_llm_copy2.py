# cartpole_rag_sr_dqn.py
# ------------------------------------------------------------
# End-to-end demo: Retrieval → Symbolic Regression → DQN on CartPole
# - Retrieval returns a seed reward expression + operator set
# - PySR discovers a compact symbolic expression (fallback: Ridge)
# - Train DQN on baseline reward vs symbolic reward; plot & save results
#
# Usage:
#   python cartpole_rag.py
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
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

SEED = 42
TOTAL_TIMESTEPS_BASE = 50_000  # from zoo, quick demo; increase to 300_000+ for stronger results
TOTAL_TIMESTEPS_SYM  = 50_000  
RUN_DIR = "runs_cartpole_llm"
os.makedirs(RUN_DIR, exist_ok=True)
set_random_seed(SEED)
save_index = 17

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
    
    prompt = f"""Based on the following research papers and documents about CartPole control and reward functions, generate a mathematical expression for a reward function that encourages:
1. Keeping the pole upright (minimizing angle from vertical)
2. Keeping the cart centered (minimizing position)
3. Minimizing velocities for stability
4. Penalizing large control actions

Context from research papers:
{context}

The expression should:
- Use variables: x_n (normalized cart position), x_dot_n (normalized cart velocity), theta_n (normalized pole angle), theta_dot_n (normalized pole angular velocity), u_n (normalized action)
- Use only the variables mentioned above, no other variables like w1, w2, etc.
- Use Python syntax with ** for exponentiation
- Include functions like abs(), sin(), cos(), tanh() if appropriate
- Use wrap() function for angle wrapping if needed
- Be a single mathematical expression that can be evaluated in Python


Return ONLY the mathematical expression, nothing else. 
"""

    try:
        # Initialize OpenAI client (using OPENAI_API_KEY environment variable)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        
        # Ensure it starts with a sign
        if not seed_expr.startswith(("+", "-")):
            seed_expr = "-" + seed_expr
            
        # Normalize variable names using existing function
        seed_expr = _normalize_vars(seed_expr)
        
        return seed_expr
        
    except Exception as e:
        print(f"LLM generation failed: {e}")
        raise e 
       



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
            progress=True,
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
        return r

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        # For CartPole-v1, actions are discrete {0,1}; map to [-1, 1] magnitude for shaping
        u = -1.0 if int(action) == 0 else 1.0
        r = self.reward_from_obs(obs, u)
        return obs, r, terminated, truncated, info

# ============================================================
# Simple training monitors
# ============================================================
class EpisodicLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.returns = []
        self.timesteps = []
        self._ep_ret = 0.0

    def _on_step(self) -> bool:
        if "episode" in self.locals.get("infos", [{}])[-1]:
            ep_info = self.locals["infos"][-1]["episode"]
            self.returns.append(ep_info["r"])
            self.timesteps.append(self.num_timesteps)
        return True

def make_env_mon(seed=SEED):
    e = gym.make('CartPole-v1')
    e = Monitor(e)
    e.reset(seed=seed)
    return e


def build_dqn(env):
    return DQN(
        "MlpPolicy", env, seed=SEED, verbose=0,
        learning_rate=1e-3,             # Zoo
        buffer_size=50_000,              # Zoo
        learning_starts=2_000,            # Zoo
        batch_size=128,                    # Zoo
        gamma=0.99,                       # Zoo
        train_freq=256,                   # Zoo (collect 256 steps, then update)
        gradient_steps=256,               # Zoo (do 128 updates)
        target_update_interval=1000,        # Zoo (frequent target syncs)
        exploration_fraction=0.08,        # Zoo
        exploration_final_eps=0.002,       # Zoo
        policy_kwargs=dict(net_arch=[256, 256]),  # Zoo
        replay_buffer_kwargs=dict(handle_timeout_termination=True),
        # IMPORTANT: do NOT set optimize_memory_usage=True with the above
    )

# ============================================================
# Main pipeline
# ============================================================
if __name__ == "__main__":

    # Baseline
    env_base = make_env_mon(SEED)
    logger_base = EpisodicLogger()
    agent_base = build_dqn(env_base)
    agent_base.learn(total_timesteps=TOTAL_TIMESTEPS_BASE, callback=logger_base)
    df_base = pd.DataFrame({'tag':'baseline',
                            'timesteps':logger_base.timesteps,
                            'episodic_return':logger_base.returns})
    env_base.close()

    # ============================================================
    # Retrieval (Haystack) → seed expression + operator sets for PySR
    # ============================================================
    QUERY_TEXT = (
        "cartpole reward function: upright pole, "
        "centered cart, equations or loss terms"
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



    # 4) Train DQN on symbolic reward
    env_sym = make_env_mon(SEED)
    env_sym = SymbolicRewardCartPole(env_sym, SYM_EXPR_STR)
    model_sym = build_dqn(env_sym)
    logger_sym = EpisodicLogger()
    print("\nTraining symbolic DQN...")
    model_sym.learn(total_timesteps=TOTAL_TIMESTEPS_SYM, callback=logger_sym)
    df_sym  = pd.DataFrame({'tag':'symbolic',
                            'timesteps':logger_sym.timesteps,
                            'episodic_return':logger_sym.returns})
    env_sym.close()


    df_all = pd.concat([df_base, df_sym], ignore_index=True)

    # ============================================================
    # Plot learning curves
    # ============================================================
    def moving_avg(x, w=10):
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w)/w, mode='valid')

    plt.figure(figsize=(8,5))
    for tag, df in df_all.groupby('tag'):
        ts = np.array(df['timesteps'])
        rs = np.array(df['episodic_return'])
        # smooth for display
        if len(rs) > 5:
            rs_s = moving_avg(rs, w=10)
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

    png_path = os.path.join(RUN_DIR, f'learning_curves_llm_{save_index}.png')
    plt.savefig(png_path, dpi=150)
    print('Saved:', png_path)

    plt.figure(figsize=(8,5))
    for tag, df in df_all.groupby('tag'):
        t = df['timesteps'].values
        r = df['episodic_return'].values
        # smooth for display
        if len(r) > 5:
            rs_s = moving_avg(r, w=10)
            
        else:
            rs_s = r
        episode_idx = np.arange(1, len(r) + 1)
        t_s = episode_idx[-len(rs_s):]
        plt.plot(t_s, rs_s, label=tag)
    # plt.xlabel('Timesteps')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Return (smoothed)')
    plt.title('CartPole: Baseline vs Symbolic-Reward DQN')
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    png_path = os.path.join(RUN_DIR, f'learning_curves_episode_llm_{save_index}.png')
    plt.savefig(png_path, dpi=140)
    print("Saved plot:", png_path)


    # plot learning curves, no smoothing
    plt.figure(figsize=(8,5))
    for tag, df in df_all.groupby('tag'):
        t = df['timesteps'].values
        r = df['episodic_return'].values
        plt.plot(t, r, label=tag)
   
    plt.xlabel('Timesteps')
    plt.ylabel('Episodic Return')
    plt.title('CartPole: Baseline vs Symbolic-Reward DQN')
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    png_path = os.path.join(RUN_DIR, f'learning_curves_episode_no_smooth_{save_index}.png')
    plt.savefig(png_path, dpi=140)
    print("Saved plot:", png_path)

    
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
        rows.append({'tag': tag,
                     'episodes': int(len(rs)),
                     'final_return': float(rs[-1] if len(rs)>0 else np.nan),
                     'AUC': auc(ts, rs)})
    summary_df = pd.DataFrame(rows)
    print(summary_df)

    csv_path = os.path.join(RUN_DIR, f'learning_curves_llm_{save_index}.csv')
    expr_path = os.path.join(RUN_DIR, f'symbolic_expression_llm_{save_index}.txt')
    df_all.to_csv(csv_path, index=False)
    with open(expr_path, 'w') as f:
        f.write(SYM_EXPR_STR)
    print('Saved:', csv_path)
    print('Saved:', expr_path)
