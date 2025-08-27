# cartpole_rag_sr_dqn.py
# ------------------------------------------------------------
# End-to-end demo: Retrieval → Symbolic Regression → DQN on CartPole
# - Retrieval returns a seed reward expression + operator set
# - PySR discovers a compact symbolic expression (fallback: Ridge)
# - Train DQN on baseline reward vs symbolic reward; plot & save results
#
# Usage:
#   python cartpole_rag_sr_dqn.py
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

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# ============== Config ==============
SEED = 42
TOTAL_STEPS = 150_000        # Increase to 300_000+ for stronger results
RUN_DIR = "runs_cartpole_clean"
os.makedirs(RUN_DIR, exist_ok=True)
set_random_seed(SEED)

# ============================================================
# Retrieval module (minimal, inline)
# ============================================================
try:
    import sympy as sp
    print("Sympy loaded")
except Exception:
    print("Sympy not loaded")
    sp = None

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Template:
    tid: str
    text: str
    expr_template: str
    operators: List[str]
    tags: List[str]

@dataclass
class ResultEntry:
    tid: str
    context: Dict[str, Any]
    outcomes: Dict[str, float]
    expr_final: str

_OP_TOKENS = {"+":"op_add","-":"op_sub","*":"op_mul","/":"op_div",
              "sin":"op_sin","cos":"op_cos","abs":"op_abs","tanh":"op_tanh",
              "wrap":"op_wrap","**2":"op_square"}
_VAR_TOKENS = ["x_n","x_dot_n","theta_n","theta_dot_n","u_n"]

def expr_fingerprint(expr_str: str) -> List[str]:
    toks = []
    for v in _VAR_TOKENS:
        if v in expr_str:
            toks.append(f"var_{v}")
    for k, tk in _OP_TOKENS.items():
        if k == "**2":
            if re.search(r"\*\*2\b", expr_str):
                toks.append(tk)
        else:
            if k in expr_str:
                toks.append(tk)
    if sp is not None:
        try:
            sy = sp.sympify(expr_str, locals={"wrap": sp.Function('wrap'), "abs": sp.Abs})
            for node in sy.atoms(sp.Function):
                name = getattr(node, "name", str(node))
                if name in _OP_TOKENS:
                    toks.append(_OP_TOKENS[name])
        except Exception:
            pass
    return toks

def pack_for_index(t: Template) -> str:
    return " ".join([t.text] + t.tags + expr_fingerprint(t.expr_template))

class SemanticLibrary:
    def __init__(self, store_dir: str = "semantic_lib_cartpole"):
        self.store_dir = Path(store_dir); self.store_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, Template] = {}
        self.results: List[ResultEntry] = []
        self._vectorizer = None; self._matrix = None; self._ids: List[str] = []

    def add_template(self, t: Template, overwrite=False):
        if overwrite or t.tid not in self.templates:
            self.templates[t.tid] = t

    def build_index(self):
        if not self.templates:
            raise ValueError("No templates to index.")
        docs, ids = [], []
        for tid, t in self.templates.items():
            docs.append(pack_for_index(t)); ids.append(tid)
        self._vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self._matrix = self._vectorizer.fit_transform(docs)
        self._ids = ids
        return self

    def _query_to_text(self, query: Dict[str,Any]) -> str:
        parts = []
        parts += query.get("goals", [])
        for k,v in query.get("context", {}).items():
            parts.append(f"{k}:{v}")
        parts += query.get("constraints", [])
        parts += [f"op_{o}" for o in query.get("operator_hints", [])]
        return " ".join(parts)

    def retrieve(self, query: Dict[str,Any], topk=3) -> List[Tuple[str,float,Template]]:
        if self._vectorizer is None:
            self.build_index()
        qtext = self._query_to_text(query)
        qvec = self._vectorizer.transform([qtext])
        sims = cosine_similarity(qvec, self._matrix)[0]
        order = np.argsort(-sims)[:topk]
        return [(self._ids[i], float(sims[i]), self.templates[self._ids[i]]) for i in order]

    def suggest_seed(self, query: Dict[str,Any], topk=3) -> Dict[str,Any]:
        hits = self.retrieve(query, topk=topk)
        if not hits:
            raise ValueError("No hits found.")
        seed_expr = hits[0][2].expr_template
        op_set = set()
        for _,_,t in hits:
            op_set.update(t.operators)
        return {"seed_expression": seed_expr,
                "operators": sorted(op_set),
                "hits": [(tid,score) for tid,score,_ in hits]}

def default_cartpole_library() -> SemanticLibrary:
    lib = SemanticLibrary()
    lib.add_template(Template(
        tid="upright_low_vel",
        text="Keep the pole upright with small angular velocity; minimal control effort.",
        expr_template="- (wrap(theta_n))**2 - 0.1*(theta_dot_n)**2 - 0.01*abs(u_n)",
        operators=["abs","sin","cos","+","-","*"],
        tags=["upright","stability","low_action","angle_periodic"]
    ))
    lib.add_template(Template(
        tid="center_cart",
        text="Keep the cart near the origin and reduce cart velocity.",
        expr_template="- 0.1*(x_n)**2 - 0.05*(x_dot_n)**2",
        operators=["+","-","*"],
        tags=["centering","position_control"]
    ))
    lib.add_template(Template(
        tid="upright_center_lowaction",
        text="Upright pole, centered cart, modest action magnitude.",
        expr_template="- (wrap(theta_n))**2 - 0.1*(theta_dot_n)**2 - 0.1*(x_n)**2 - 0.05*(x_dot_n)**2 - 0.01*abs(u_n)",
        operators=["abs","sin","cos","+","-","*"],
        tags=["upright","centering","low_action","angle_periodic"]
    ))
    lib.add_template(Template(
        tid="smooth_actions",
        text="Discourage large actions; prefer smoother control.",
        expr_template="- 0.02*abs(u_n) - 0.02*(x_dot_n**2 + theta_dot_n**2)",
        operators=["abs","+","-","*"],
        tags=["low_action","smooth","effort"]
    ))
    lib.add_template(Template(
        tid="angle_sparse",
        text="Sparse term focusing on angle error only.",
        expr_template="- (wrap(theta_n))**2",
        operators=["sin","cos","+","-","*"],
        tags=["upright","sparse","angle_periodic"]
    ))
    lib.build_index()
    return lib

# ============================================================
# Retrieval → seed expression + operator sets for PySR
# ============================================================
lib = default_cartpole_library()
query = {
    "goals": ["upright", "centering", "low_action"],
    "context": {"env_id": "CartPole-v1", "algo": "DQN"},
    "constraints": ["angle_periodic"],
    "operator_hints": ["abs", "sin", "cos"],
}
suggestion = lib.suggest_seed(query, topk=3)
SEED_EXPR = suggestion["seed_expression"]
OP_SET = suggestion["operators"]
UNARY_OPS = [op for op in OP_SET if op in ["abs","sin","cos","tanh"]]
BINARY_OPS = [op for op in OP_SET if op in ["+","-","*","/"]]
print("Seed expression:", SEED_EXPR)
print("UNARY_OPS:", UNARY_OPS)
print("BINARY_OPS:", BINARY_OPS)

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
    locs = dict(x_n=x_n, x_dot_n=x_dot_n, theta_n=theta_n, theta_dot_n=theta_dot_n, u_n=u_n)
    locs.update(safe)
    return float(eval(expr_str, {"__builtins__": {}}, locs))

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
        target = eval_expr(expr_str, x_n, x_dot_n, theta_n, theta_dot_n, u_n)
        X.append([x_n, x_dot_n, theta_n, theta_dot_n, u_n])
        y.append(target)
        obs = next_obs if not (term or trunc) else env.reset()[0]
    env.close()
    return np.array(X), np.array(y)

print("Sampling SR dataset...")
X_sr, y_sr = sample_sr_dataset(SEED_EXPR, n=6000, seed=SEED)
print("SR dataset shapes:", X_sr.shape, y_sr.shape)

# ============================================================
# Fit PySR (fallback to Ridge if PySR not available)
# ============================================================
SYM_EXPR_STR = None
SR_FALLBACK = False

try:
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=120,             # increase for stronger fits
        unary_operators=UNARY_OPS,   # guided by retrieval
        binary_operators=BINARY_OPS,
        maxsize=18,
        batching=True,
        procs=0,
        progress=True,
    )
    model.fit(X_sr, y_sr)
    best = model.get_best()
    SYM_EXPR_STR = str(best.get("sympy_format", best.get("equation")))
    print("PySR best expression:", SYM_EXPR_STR)
except Exception as e:
    print("PySR failed or unavailable; fallback to linear model:", e)
    SR_FALLBACK = True
    from sklearn.linear_model import Ridge
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
    print("Fallback expression:", SYM_EXPR_STR)

# ============================================================
# Reward wrapper that uses the discovered expression
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
        locs = dict(x_n=x_n, x_dot_n=x_dot_n, theta_n=theta_n, theta_dot_n=theta_dot_n, u_n=u_n)
        locs.update(self.safe)
        try:
            return float(eval(self.expr_str, {"__builtins__": {}}, locs))
            print("Evaled expression, sympy:", self.expr_str)
        except Exception:
            return float(eval(SEED_EXPR, {"__builtins__": {}}, locs))
            print("Evaled expression, seed:", SEED_EXPR)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        r = self._compute_reward(obs, action)
        return obs, r, terminated, truncated, info

# ============================================================
# DQN training (baseline vs symbolic reward)
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
        learning_rate=2.5e-4,           # was 2.5e-4
        buffer_size=50_000,           # was 100_000
        learning_starts=500,          # was 1_000
        batch_size=128,               # was 64
        gamma=0.99,
        train_freq=(4, 'step'),       # was (4, 'step')
        gradient_steps=1,             # do 1 update per env step; can try 2–4
        target_update_interval=500,   # was 1_000 → faster propagation
        exploration_fraction=0.07,    # was 0.1 → faster epsilon decay
        exploration_final_eps=0.02,   # was 0.01 (slightly higher is fine for CartPole)
        policy_kwargs={'net_arch': [64, 64]},  # was [256, 256]
        
    )

# Baseline
env_base = make_env_mon(SEED)
logger_base = CurveLogger()
agent_base = build_dqn(env_base)
print("Training baseline DQN...")
agent_base.learn(total_timesteps=TOTAL_STEPS, callback=logger_base)
df_base = pd.DataFrame({'tag':'baseline',
                        'timesteps':logger_base.timesteps,
                        'episodic_return':logger_base.returns})
env_base.close()

# Symbolic reward
env_sym = make_env_mon(SEED)
env_sym = SymbolicRewardCartPole(env_sym, SYM_EXPR_STR)
logger_sym = CurveLogger()
agent_sym = build_dqn(env_sym)
print("Training symbolic DQN...")
agent_sym.learn(total_timesteps=TOTAL_STEPS, callback=logger_sym)
df_sym = pd.DataFrame({'tag':'symbolic',
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
png_path = os.path.join(RUN_DIR, 'learning_curves_fake.png')
plt.savefig(png_path, dpi=140)
print("Saved plot:", png_path)

# ============================================================
# Results table + save artifacts
# ============================================================
def auc(ts, rs):
    return float(np.trapz(rs, ts)) if len(ts)>1 else float(rs[-1] if len(rs)>0 else 0)

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

csv_path = os.path.join(RUN_DIR, 'learning_curves_fake.csv')
expr_path = os.path.join(RUN_DIR, 'symbolic_expression_fake.txt')
df_all.to_csv(csv_path, index=False)
with open(expr_path, 'w') as f:
    f.write(SYM_EXPR_STR)
print('Saved:', csv_path)
print('Saved:', expr_path)

