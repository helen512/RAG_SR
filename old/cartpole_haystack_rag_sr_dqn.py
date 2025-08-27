# cartpole_haystack_rag_sr_dqn.py
# ------------------------------------------------------------
# End-to-end demo: Haystack RAG → (seed reward) → Symbolic Regression → DQN on CartPole
# - Uses your haystack_pipeline.py to retrieve control/RL snippets
# - Heuristically proposes a seed reward from top RAG hits (fallback to defaults)
# - PySR discovers a compact symbolic expression (fallback: Ridge)
# - Train DQN on baseline reward vs symbolic reward; plot & save results
#
# Usage:
#   python cartpole_haystack_rag_sr_dqn.py
#   (Optional) Set env vars to point at your Marker JSONs:
#       RAG_BLOCKS=rag_seed/marker/murray_lyapunov/blocks.json
#       RAG_CONTENT=rag_seed/marker/murray_lyapunov/murray_lyapunov.json
#       RAG_META=rag_seed/marker/murray_lyapunov/murray_lyapunov_meta.json
# ------------------------------------------------------------

import os
import re
import math
import json
import time
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# ============== Config ==============
SEED = int(os.environ.get("SEED", 42))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 150_000))
RUN_DIR = os.environ.get("RUN_DIR", "runs_cartpole_haystack")
os.makedirs(RUN_DIR, exist_ok=True)
set_random_seed(SEED)

# Try optional sympy (used for minor tokenization aid only)
try:
    import sympy as sp  # type: ignore
except Exception:
    sp = None

# ============================================================
# Utilities
# ============================================================
def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def safe_eval_expr(expr_str, x_n, x_dot_n, theta_n, theta_dot_n, u_n):
    safe = {"abs": abs, "sin": np.sin, "cos": np.cos, "tanh": np.tanh, "wrap": _wrap_angle}
    locs = dict(
        x_n=x_n, x_dot_n=x_dot_n, theta_n=theta_n, theta_dot_n=theta_dot_n, u_n=u_n,
        # PySR default variable names mapping
        x0=x_n, x1=x_dot_n, x2=theta_n, x3=theta_dot_n, x4=u_n
    )
    locs.update(safe)
    return float(eval(expr_str, {"__builtins__": {}}, locs))

# ============================================================
# 1) RAG hookup (imports your haystack_pipeline.py)
# ============================================================
class RAGSeedFinder:
    """
    Builds your Haystack hybrid pipeline from haystack_pipeline.py and queries it
    for reward/Lyapunov candidate hints. Produces a seed reward expression and
    a set of operators for SR.
    """
    def __init__(self,
                 blocks: Optional[Path] = None,
                 content: Optional[Path] = None,
                 meta: Optional[Path] = None):
        self.blocks = blocks
        self.content = content
        self.meta = meta
        self.pipe = None
        self.available = False
        try:
            import haystack_pipeline as hp  # your file
            self.hp = hp
        except Exception as e:
            print("[RAG] Could not import haystack_pipeline.py:", e)
            self.hp = None

    def _default_paths(self):
        # Same defaults as in your haystack_pipeline.py __main__
        base = Path("rag_seed/marker/murray_lyapunov")
        return (
            self.blocks or Path(os.environ.get("RAG_BLOCKS", base / "blocks.json")),
            self.content or Path(os.environ.get("RAG_CONTENT", base / "murray_lyapunov.json")),
            self.meta or Path(os.environ.get("RAG_META", base / "murray_lyapunov_meta.json")),
        )

    def initialize(self) -> bool:
        if self.hp is None:
            return False
        try:
            blocks, content, meta = self._default_paths()
            print(f"[RAG] Loading Marker JSONs:\n  blocks={blocks}\n  content={content}\n  meta={meta}")
            docs = self.hp.load_marker_as_documents(blocks, content, meta, source_name=str(blocks.parent.name))
            if not docs:
                print("[RAG] No documents were created from Marker JSONs.")
                return False
            qdrant_store, sparse_store = self.hp.build_stores(qdrant_collection="rl_control_papers")
            self.hp.index_documents(qdrant_store, sparse_store, docs)
            self.pipe = self.hp.build_query_pipeline(qdrant_store, sparse_store)
            self.available = True
            print("[RAG] Pipeline built and warmed up.")
            return True
        except Exception as e:
            print("[RAG] Initialization failed:", e)
            traceback.print_exc()
            return False

    def query(self, text: str, top_k: int = 10):
        if not self.available or self.pipe is None:
            return []
        try:
            docs = self.pipe.fuse_and_rerank(self.pipe, text)
            return docs[:top_k]
        except Exception as e:
            print("[RAG] Query failed:", e)
            traceback.print_exc()
            return []

def _extract_math_candidates(html: str) -> List[str]:
    """
    Very lightweight math sniffer from HTML-ish blocks: looks for inline math
    in $...$, \\(...\\), display math \\[...\\], or 'V(x) =', 'J =', 'cost' lines.
    Returns small list of strings we can then tokenize for hints.
    """
    cands = []
    # $...$
    cands += re.findall(r"\$(.+?)\$", html)
    # \( ... \)
    cands += re.findall(r"\\\((.+?)\\\)", html)
    # \[ ... \]
    cands += re.findall(r"\\\[(.+?)\\\]", html)
    # V(x)= ... or J= ... or cost = ...
    for m in re.finditer(r"(V\s*\(.+?\)\s*=\s*[^\n<]+)", html):
        cands.append(m.group(1))
    for m in re.finditer(r"(J\s*=\s*[^\n<]+)", html):
        cands.append(m.group(1))
    for m in re.finditer(r"(cost\s*=\s*[^\n<]+)", html, flags=re.IGNORECASE):
        cands.append(m.group(1))
    # Keep shortish
    cands = [s.strip() for s in cands if 1 <= len(s) <= 300]
    return list(dict.fromkeys(cands))  # dedupe

def propose_seed_from_rag(docs) -> Tuple[str, List[str]]:
    """
    Turn top-ranked RAG docs into a CartPole-style quadratic seed with
    periodic angle, plus operator hints. If extraction fails, use a good default.
    """
    # Default (solid baseline)
    default_expr = "- (wrap(theta_n))**2 - 0.1*(theta_dot_n)**2 - 0.1*(x_n)**2 - 0.05*(x_dot_n)**2 - 0.01*abs(u_n)"
    default_ops = ["abs", "sin", "cos", "+", "-", "*"]

    if not docs:
        return default_expr, default_ops

    text = " ".join([(d.content or "") for d in docs if getattr(d, 'content', None)])
    math_bits = []
    for d in docs:
        html = getattr(d, 'content', '') or ''
        math_bits.extend(_extract_math_candidates(html))

    # Heuristic counts
    def hcount(patterns):
        s = text.lower()
        return sum(s.count(p) for p in patterns)

    include_theta = hcount(["theta", "angle", "pendulum", "upright"]) > 0
    include_x = hcount(["cart", "position", "center", "origin"]) > 0
    include_vel = hcount(["velocity", "rate", "derivative", "dot"]) > 0
    include_action = hcount(["control effort", "torque", "input", "action", "u"]) > 0
    trig_hint = hcount(["sin", "cos", "periodic", "angle wrapping", "modulo"]) > 0

    # Operator hints
    ops = set(["+", "-", "*"])
    if trig_hint or include_theta:
        ops.update(["sin", "cos"])
    if include_action:
        ops.add("abs")

    # Weights tuned gently; boost terms that appear strongly
    w_th = 1.0 if include_theta else 0.3
    w_thd = 0.1 if include_vel else 0.05
    w_x = 0.1 if include_x else 0.05
    w_xd = 0.05 if include_vel else 0.02
    w_u = 0.01 if include_action else 0.0

    expr_parts = []
    if include_theta or trig_hint:
        expr_parts.append(f"- {w_th}*(wrap(theta_n))**2")
        expr_parts.append(f"- {w_thd}*(theta_dot_n)**2")
    if include_x:
        expr_parts.append(f"- {w_x}*(x_n)**2")
    if include_vel:
        expr_parts.append(f"- {w_xd}*(x_dot_n)**2")
    if include_action:
        expr_parts.append(f"- {w_u}*abs(u_n)")

    seed = " - ".join([p.replace("- ", "") for p in expr_parts]) if expr_parts else default_expr
    # Clamp: ensure at least angle penalty is present
    if "theta_n" not in seed:
        seed = default_expr

    return seed, sorted(list(ops.union(default_ops)))

def build_seed_via_rag() -> Tuple[str, List[str], List[Tuple[str, float, str]]]:
    """
    Initializes RAG, sends a query tailored to CartPole reward/Lyapunov hints,
    and returns (seed_expr, op_set, debug_hits)
    """
    rag = RAGSeedFinder()
    ok = rag.initialize()
    if not ok:
        print("[RAG] Using default seed and operators (RAG unavailable).")
        seed, ops = propose_seed_from_rag([])
        return seed, ops, []
    query = (
        "Lyapunov or reward/cost shaping for inverted pendulum/cart-pole; "
        "periodic angle; stabilize upright with small angle, small angular and cart velocity; "
        "prefer minimal control effort"
    )
    hits = rag.query(query, top_k=8)
    seed, ops = propose_seed_from_rag(hits)
    debug = []
    for h in hits[:5]:
        meta = getattr(h, "meta", {}) or {}
        debug.append((float(getattr(h, "score", 0.0) or 0.0), meta.get("source", ""), (h.content or "")[:160].replace("\\n", " ")))
    print("[RAG] Proposed seed expression:", seed)
    print("[RAG] Operators:", ops)
    print("[RAG] Top hits (score, source, snippet):")
    for s, src, snip in debug:
        print(f"  - {s:.4f} | {src} | {snip}")
    return seed, ops, debug

# ============================================================
# 2) SR dataset (teacher = seed expression from RAG or default)
# ============================================================
def make_env(env_id='CartPole-v1', seed=SEED):
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

def sample_sr_dataset(expr_str, n=6000, seed=SEED):
    env = make_env('CartPole-v1', seed=seed)
    # thresholds from unwrapped env
    x_threshold = env.env.unwrapped.x_threshold
    theta_threshold = env.env.unwrapped.theta_threshold_radians
    obs, _ = env.reset()
    X, y = [], []
    for _ in range(n):
        a = env.action_space.sample()
        force = 1.0 if a==1 else -1.0
        next_obs, _, term, trunc, _ = env.step(a)
        x_, xdot_, th_, thdot_ = obs
        x_n = x_ / (x_threshold + 1e-8)
        x_dot_n = xdot_ / 2.0
        theta_n = _wrap_angle(th_) / (theta_threshold + 1e-8)
        theta_dot_n = thdot_ / 2.0
        u_n = force
        target = safe_eval_expr(expr_str, x_n, x_dot_n, theta_n, theta_dot_n, u_n)
        X.append([x_n, x_dot_n, theta_n, theta_dot_n, u_n])
        y.append(target)
        obs = next_obs if not (term or trunc) else env.reset()[0]
    env.close()
    return np.array(X), np.array(y)

# ============================================================
# 3) Fit PySR (fallback to Ridge if PySR not available)
# ============================================================
def fit_symbolic_reward(X_sr, y_sr, unary_ops: List[str], binary_ops: List[str]) -> Tuple[str, bool]:
    SYM_EXPR_STR = None
    SR_FALLBACK = False
    try:
        from pysr import PySRRegressor  # type: ignore
        model = PySRRegressor(
            niterations=120,
            unary_operators=[op for op in unary_ops if op in ["abs","sin","cos","tanh"]],
            binary_operators=[op for op in binary_ops if op in ["+","-","*","/"]],
            maxsize=18,
            batching=True,
            procs=0,
            progress=True,
        )
        model.fit(X_sr, y_sr)
        best = model.get_best()
        SYM_EXPR_STR = str(best.get("sympy_format", best.get("equation")))
        print("[SR] PySR best expression:", SYM_EXPR_STR)
    except Exception as e:
        print("[SR] PySR failed or unavailable; fallback to linear model:", e)
        SR_FALLBACK = True
        from sklearn.linear_model import Ridge  # type: ignore
        def basis(X):
            x_n, xdot_n, theta_n, thetadot_n, u_n = X.T
            return np.vstack([
                x_n, xdot_n, theta_n, thetadot_n, u_n,
                x_n**2, xdot_n**2, theta_n**2, thetadot_n**2, np.abs(u_n),
                np.ones_like(x_n)
            ]).T
        Phi = basis(X_sr)
        ridge = Ridge(alpha=1e-2).fit(Phi, y_sr)
        names = ["x_n","x_dot_n","theta_n","theta_dot_n","u_n",
                 "x_n**2","x_dot_n**2","theta_n**2","theta_dot_n**2","abs(u_n)","1"]
        terms = []
        for w, name in zip(ridge.coef_, names):
            if abs(w) > 1e-6:
                terms.append(f"({w:.4g})*{name}")
        SYM_EXPR_STR = " + ".join(terms) + (f" + {ridge.intercept_:.4g}" if abs(ridge.intercept_)>1e-6 else "")
        print("[SR] Fallback expression:", SYM_EXPR_STR)
    return SYM_EXPR_STR, SR_FALLBACK

# ============================================================
# 4) Reward wrapper that uses the discovered expression
# ============================================================
class SymbolicRewardCartPole(gym.Wrapper):
    def __init__(self, env, expr_str):
        super().__init__(env)
        self.expr_str = expr_str
        self.safe = {"abs": abs, "sin": np.sin, "cos": np.cos, "tanh": np.tanh,
                     "wrap": lambda a: (a + np.pi) % (2*np.pi) - np.pi}

    def _compute_reward(self, obs, action):
        x_, xdot_, th_, thdot_ = obs
        base = self.env.unwrapped
        x_threshold = base.x_threshold
        theta_threshold = base.theta_threshold_radians
        x_n = x_ / (x_threshold + 1e-8)
        x_dot_n = xdot_ / 2.0
        theta_n = ((th_ + np.pi) % (2*np.pi) - np.pi) / (theta_threshold + 1e-8)
        theta_dot_n = thdot_ / 2.0
        u_n = 1.0 if int(action)==1 else -1.0
        
        # Create variable mapping for both PySR default names and semantic names
        locs = dict(
            x_n=x_n, x_dot_n=x_dot_n, theta_n=theta_n, theta_dot_n=theta_dot_n, u_n=u_n,
            # PySR default variable names mapping
            x0=x_n, x1=x_dot_n, x2=theta_n, x3=theta_dot_n, x4=u_n
        )
        locs.update(self.safe)
        try:
            return float(eval(self.expr_str, {"__builtins__": {}}, locs))
        except Exception as e:
            print(f"[WARNING] Expression evaluation failed: {e}")
            print(f"[WARNING] Expression: {self.expr_str}")
            print(f"[WARNING] Available variables: {list(locs.keys())}")
            # Fallback to basic penalty
            return -(theta_n**2 + 0.1*theta_dot_n**2 + 0.1*x_n**2 + 0.05*x_dot_n**2)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        r = self._compute_reward(obs, action)
        return obs, r, terminated, truncated, info

# ============================================================
# 5) DQN training (baseline vs symbolic reward)
# ============================================================
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
    e = gym.make('CartPole-v1')
    e = Monitor(e)
    e.reset(seed=seed)
    return e

def build_dqn(env):
    return DQN(
        'MlpPolicy', env, seed=SEED, verbose=0,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=500,
        batch_size=128,
        gamma=0.99,
        train_freq=(1, 'step'),
        gradient_steps=1,
        target_update_interval=250,
        exploration_fraction=0.05,
        exploration_final_eps=0.02,
        policy_kwargs={'net_arch': [64, 64]},
    )

def moving_avg(x, w=10):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')

def auc(ts, rs):
    return float(np.trapz(rs, ts)) if len(ts)>1 else float(rs[-1] if len(rs)>0 else 0)

def main():
    # ---- RAG → seed ----
    seed_expr, op_set, rag_hits = build_seed_via_rag()
    UNARY_OPS = [op for op in op_set if op in ["abs","sin","cos","tanh"]]
    BINARY_OPS = [op for op in op_set if op in ["+","-","*","/"]]

    # ---- SR dataset ----
    print("[SR] Sampling dataset from seed expression...")
    X_sr, y_sr = sample_sr_dataset(seed_expr, n=6000, seed=SEED)
    print("[SR] Dataset shapes:", X_sr.shape, y_sr.shape)

    # ---- Fit SR ----
    sym_expr, used_fallback = fit_symbolic_reward(X_sr, y_sr, UNARY_OPS, BINARY_OPS)
    expr_to_use = sym_expr or seed_expr

    # ---- Baseline training ----
    env_base = make_env_mon(SEED)
    logger_base = CurveLogger()
    agent_base = build_dqn(env_base)
    agent_base.learn(total_timesteps=TOTAL_STEPS, callback=logger_base)
    df_base = pd.DataFrame({'tag':'baseline',
                            'timesteps':logger_base.timesteps,
                            'episodic_return':logger_base.returns})
    env_base.close()

    # ---- Symbolic-reward training ----
    env_sym = make_env_mon(SEED)
    env_sym = SymbolicRewardCartPole(env_sym, expr_to_use)
    logger_sym = CurveLogger()
    agent_sym = build_dqn(env_sym)
    agent_sym.learn(total_timesteps=TOTAL_STEPS, callback=logger_sym)
    df_sym = pd.DataFrame({'tag':'symbolic',
                           'timesteps':logger_sym.timesteps,
                           'episodic_return':logger_sym.returns})
    env_sym.close()

    df_all = pd.concat([df_base, df_sym], ignore_index=True)

    # ---- Plot learning curves ----
    plt.figure(figsize=(8,5))
    for tag, df in df_all.groupby('tag'):
        t = df['timesteps'].values
        r = df['episodic_return'].values
        r_s = moving_avg(r, w=10)
        episode_idx = np.arange(1, len(r) + 1)
        t_s = episode_idx[-len(r_s):]
        plt.plot(t_s, r_s, label=tag)
    plt.xlabel('Episode')
    plt.ylabel('Episodic Return (smoothed)')
    plt.title('CartPole: Baseline vs Symbolic-Reward DQN')
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    png_path = os.path.join(RUN_DIR, 'learning_curves.png')
    plt.savefig(png_path, dpi=140)
    print("Saved plot:", png_path)

    # ---- Save results ----
    rows = []
    for tag, df in df_all.groupby('tag'):
        ts = df['timesteps'].values
        rs = df['episodic_return'].values
        rows.append({'run': tag,
                     'episodes': int(len(rs)),
                     'final_return': float(rs[-1] if len(rs)>0 else np.nan),
                     'AUC': auc(ts, rs)})
    summary_df = pd.DataFrame(rows)
    print(summary_df)

    csv_path = os.path.join(RUN_DIR, 'learning_curves.csv')
    expr_path = os.path.join(RUN_DIR, 'symbolic_expression.txt')
    df_all.to_csv(csv_path, index=False)
    with open(expr_path, 'w') as f:
        f.write(expr_to_use)
    print('Saved:', csv_path)
    print('Saved:', expr_path)
    print('[INFO] Expression used for reward:\n', expr_to_use)

if __name__ == "__main__":
    main()
