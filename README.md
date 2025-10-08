# CartPole RAG-LLM-PPO Project

An end-to-end Retrieval-Augmented Generation (RAG) system integrated with Large Language Models (LLMs) and Reinforcement Learning (RL) for CartPole control. This project demonstrates how to leverage academic knowledge from research papers to discover symbolic reward functions and train effective RL policies.

## Overview

This project combines:
- **RAG (Retrieval-Augmented Generation)**: Retrieves relevant knowledge from academic papers about CartPole dynamics, control theory, and Lyapunov stability
- **LLM (Large Language Model)**: Generates seed reward expressions based on retrieved research context
- **Symbolic Regression**: Discovers compact, interpretable reward functions using PySR
- **PPO (Proximal Policy Optimization)**: Trains RL agents on both baseline and symbolic rewards

## 1. RAG Source Documents

### Academic Papers (in `rag_source/`)

The RAG system is built on the following research papers about CartPole control, inverted pendulum dynamics, and control theory:

1. **`Cart_and_Pendulum_(Lagrange).pdf`**
   - Lagrangian mechanics for cart-pole systems
   - Equations of motion derivation

2. **`cartPoleEqns.pdf`**
   - Mathematical equations for CartPole dynamics
   - State space representations

3. **`Energy_shaping_control_revisited.pdf`**
   - Energy-based control methods
   - Energy shaping for stabilization

4. **`MIT_chapter3.pdf`**
   - MIT lecture notes on control systems
   - Foundational control theory concepts

5. **`murray_lyapunov.pdf`**
   - Lyapunov stability analysis
   - Candidate Lyapunov functions for control
   - Stability proofs for nonlinear systems

6. **`phys239_2016_lec05.pdf`**
   - Physics lecture on classical mechanics
   - Pendulum dynamics and control

These papers provide the theoretical foundation for:
- Understanding CartPole dynamics (position, velocity, angle, angular velocity)
- Energy-based reward functions
- Lyapunov candidate functions for stability
- Control objectives (keeping pole upright, cart centered, minimizing velocities)

### Parsed Documents

The parsed documents are stored in `rag_parsed/marker/` with structured JSON exports from Marker (PDF parser).

## 2. RAG Pipeline Tools & Libraries

### Core RAG Infrastructure

#### Document Processing
- **[Marker](https://github.com/VikParuchuri/marker)**: PDF-to-structured-JSON parser
  - Extracts layout blocks, text, equations, and metadata from PDFs
  - Outputs `blocks.json`, content JSON, and metadata JSON

#### Vector Database & Retrieval
- **[Haystack 2.x](https://haystack.deepset.ai/)**: RAG orchestration framework
  - Document stores, retrievers, and pipeline components
  - Version: `haystack-ai>=2.1.0`

- **[Qdrant](https://qdrant.tech/)**: Vector database for dense retrieval
  - Persistent local storage in `./qdrant_papers`
  - Stores document embeddings (384-dimensional vectors)
  - Package: `qdrant-client`, `qdrant-haystack`

- **BM25 (Sparse Retrieval)**: Keyword-based retrieval
  - In-memory document store (`InMemoryDocumentStore`)
  - Cache stored as JSONL in `bm25_cache.jsonl` (12MB)
  - Component: `InMemoryBM25Retriever`

#### Embeddings & Reranking
- **[SentenceTransformers](https://www.sbert.net/)**: Neural text embeddings
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
  - Reranking model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Components:
    - `SentenceTransformersDocumentEmbedder` (for indexing)
    - `SentenceTransformersTextEmbedder` (for queries)
    - `SentenceTransformersReranker` (cross-encoder reranking)

#### Fusion Strategy
- **Reciprocal Rank Fusion (RRF)**: Combines dense + sparse retrieval results
  - Configurable weights for dense vs. sparse
  - Default: `dense_k=20`, `sparse_k=40`, `final_k=10`

### LLM Integration
- **[OpenAI API](https://openai.com/)**: GPT-4 for reward function generation
  - Model: `gpt-4.1`
  - Generates Python-evaluable mathematical expressions
  - Uses retrieved research context to inform reward design

### Symbolic Regression
- **[PySR](https://github.com/MilesCranmer/PySR)**: Symbolic regression via genetic programming
  - Discovers compact symbolic reward functions
  - Configurable operator sets (unary: `abs`, `sin`, `cos`, `tanh`; binary: `+`, `-`, `*`, `/`)
  - Fallback: Ridge regression with polynomial features if PySR unavailable

### Reinforcement Learning
- **[Stable Baselines3](https://stable-baselines3.readthedocs.io/)**: RL algorithms
  - PPO (Proximal Policy Optimization) implementation
  - Policy: MLP with [64, 64] hidden layers
  - Hyperparameters: lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10

- **[Gymnasium](https://gymnasium.farama.org/)**: RL environment interface
  - Environment: `CartPole-v1`
  - Custom wrapper: `SymbolicRewardCartPole` for shaped rewards

### Supporting Libraries
- **NumPy, Pandas**: Data manipulation
- **Matplotlib**: Visualization (learning curves)
- **JuliaCall**: Interface for PySR (Julia-based)

## 3. Structure of `cartpole_rag_llm_ppo.py`

### High-Level Pipeline

```
RAG Retrieval → LLM Seed Generation → Symbolic Regression → PPO Training → Evaluation
```

### Detailed Architecture

#### 1. **RAG Retrieval Module** (Lines 40-201)

**Function**: `haystack_seed_reward(query_text)`

- **Input**: Natural language query about CartPole control
  ```python
  query = "cartpole energy function: upright pole, potential energy of inverted pendulum, lyapunov function for cartpole stability"
  ```

- **Process**:
  1. Initialize Qdrant (dense) and BM25 (sparse) stores
  2. Build `HybridQueryEngine` with embedding and reranking models
  3. Execute hybrid retrieval (dense + sparse + RRF fusion)
  4. Return top-k reranked documents (default: 12 documents)

- **Output**: 
  - Retrieved text chunks from research papers
  - Source metadata (paper name, block type)
  - Relevance scores

#### 2. **LLM Seed Generation** (Lines 97-167)

**Function**: `llm_generate_seed_expression(texts)`

- **Input**: Top 5 retrieved document chunks

- **Process**:
  1. Constructs prompt with research context
  2. Instructs LLM to generate reward expression with:
     - Canonical variables: `x_n`, `x_dot_n`, `theta_n`, `theta_dot_n`, `u_n`
     - Allowed functions: `abs()`, `sin()`, `cos()`, `tanh()`, `wrap()`
     - Python-evaluable syntax
  3. Calls OpenAI API (GPT-4.1, temp=0.6)
  4. Normalizes variable names and validates syntax

- **Output**: 
  - Seed reward expression (string)
  - Example: `"-(theta_n**2 + 0.1*theta_dot_n**2 + 0.1*x_n**2 + 0.01*x_dot_n**2 + 0.001*abs(u_n))"`
  - Saved to: `runs_cartpole_llm_ppo/seed_expr_{i}.txt`

#### 3. **Operator Set Inference** (Lines 61-73)

**Function**: `_canon_ops_from_expr(expr)`

- **Input**: Seed expression string

- **Process**: Heuristically detects operators present in expression

- **Output**: 
  - `UNARY_OPS`: List of unary operators (e.g., `["abs", "sin", "cos"]`)
  - `BINARY_OPS`: List of binary operators (e.g., `["+", "-", "*"]`)
  - Used to constrain PySR search space

#### 4. **SR Dataset Generation** (Lines 206-255)

**Function**: `sample_sr_dataset(expr_str, n=6000)`

- **Input**: Seed expression, number of samples

- **Process**:
  1. Samples random CartPole states around stable upright equilibrium
     - Position: N(0, 0.5)
     - Angle: N(0, 0.1)
     - Velocities: N(0, 0.5)
  2. Normalizes states by thresholds (`x_threshold=2.4`, `theta_threshold_radians`)
  3. Evaluates seed expression for each state
  4. Creates training dataset (X, y)

- **Output**: 
  - `X`: (6000, 5) array of normalized states + actions
  - `y`: (6000,) array of reward values (teacher signal)

#### 5. **Symbolic Regression** (Lines 258-293)

**Function**: `fit_symbolic_reward(X, y, unary_ops, binary_ops)`

- **Input**: Training data, operator constraints

- **Process**:
  1. **Primary**: PySR genetic programming
     - `niterations=60`, `populations=10`
     - `maxsize=20` (expression complexity limit)
     - `elementwise_loss="L2DistLoss()"`
     - Returns best equation from Pareto frontier
  
  2. **Fallback**: Ridge regression with polynomial features (degree=2)
     - Keeps top 6 terms by coefficient magnitude
     - More interpretable than deep polynomial

- **Output**: 
  - Symbolic reward expression (string)
  - Example: `"0.3*x_n**2 + 0.5*theta_n**2 - 0.1*sin(theta_n)"`
  - Saved to: `runs_cartpole_llm_ppo/symbolic_expression_ppo_{i}.txt`

#### 6. **Reward Wrapper** (Lines 296-339)

**Class**: `SymbolicRewardCartPole(gym.Wrapper)`

- **Purpose**: Replaces CartPole's default reward (+1 per step) with symbolic reward

- **Method**: `reward_from_obs(obs, action_value)`
  1. Extracts state: `x`, `x_dot`, `theta`, `theta_dot`
  2. Normalizes to canonical variables: `x_n`, `x_dot_n`, `theta_n`, `theta_dot_n`, `u_n`
  3. Evaluates symbolic expression safely (restricted `eval()` with whitelisted functions)
  4. Returns hybrid reward: `1.0 + 0.2 * symbolic_reward`
     - Keeps survival signal (+1) to prevent reward hacking
     - Adds shaped bonus from symbolic expression

- **Safety**: Uses restricted namespace to prevent arbitrary code execution

#### 7. **PPO Training** (Lines 342-472)

**Class**: `EpisodicLoggerWithEarlyStop(BaseCallback)`

- **Features**:
  - Logs episode returns and timesteps
  - Early stopping when rolling mean return ≥ 475 (over 100 episodes)
  - Tracks x-threshold failures (cart exceeds position ±2.4)

**Function**: `build_ppo(env)`

- **Hyperparameters**:
  ```python
  learning_rate=3e-4
  n_steps=2048         # steps per rollout
  batch_size=64
  n_epochs=10          # gradient updates per rollout
  gamma=0.99           # discount factor
  gae_lambda=0.95      # GAE advantage estimation
  clip_range=0.2       # PPO clip parameter
  policy: MLP[64, 64]  # 2-layer network
  ```

- **Training Process**:
  1. **Baseline PPO**: Train on standard CartPole (+1 per step reward)
  2. **Symbolic PPO**: Train on wrapped environment with symbolic reward
  3. Both train for `TOTAL_TIMESTEPS=50,000` (or until early stop)

#### 8. **Evaluation** (Lines 404-451)

**Function**: `evaluate_policy_with_x_failures(model, env, n_eval_episodes=100)`

- **Metrics**:
  - Mean episode reward ± std
  - X-threshold failure count and rate
  - Evaluation on separate seed (seed+1000) for unbiased assessment

- **Purpose**: 
  - Compare baseline vs. symbolic reward performance
  - Diagnose failure modes (cart position control vs. pole angle control)

#### 9. **Results & Visualization** (Lines 477-658)

**Main Execution Block**:

1. **RAG Query**: Retrieve CartPole control knowledge
   ```python
   QUERY = "cartpole energy function: upright pole, potential energy of inverted pendulum, lyapunov function for cartpole stability"
   ```

2. **Generate & Discover**:
   - LLM generates seed expression
   - Sample 6000 state-reward pairs
   - PySR discovers compact symbolic function

3. **Train Both Agents**:
   - Baseline PPO (standard CartPole reward)
   - Symbolic PPO (discovered reward)

4. **Compare & Save**:
   - Learning curves plot (`learning_curves_ppo_{i}.png`)
   - Summary CSV with AUC, final returns, failure rates
   - Expression files (`seed_expr_{i}.txt`, `symbolic_expression_ppo_{i}.txt`)

**Output Files** (in `runs_cartpole_llm_ppo/`):
```
learning_curves_ppo_1.png         # Training curves
learning_curves_ppo_1.csv         # Episode-level data
training_summary_ppo_1.csv        # Aggregate metrics
seed_expr_1.txt                   # LLM-generated seed
symbolic_expression_ppo_1.txt     # PySR-discovered expression
```

### Key Design Choices

1. **Hybrid Reward**: Combines survival signal (+1) with symbolic shaping (0.2×)
   - Prevents reward hacking (agent learns both survival and optimization)
   
2. **Early Stopping**: Stops at 475 rolling average return
   - CartPole-v1 max = 500; early stop saves compute

3. **Normalized Variables**: All states normalized to [-1, 1] range
   - `x_n = x / 2.4` (position threshold)
   - `theta_n = wrap(theta) / theta_threshold`
   - `x_dot_n = tanh(x_dot / 2.0)` (velocity squashing)
   - Improves PySR search and LLM understanding

4. **Safe Expression Evaluation**: Restricted `eval()` with whitelist
   - Only allows math functions: `abs`, `sin`, `cos`, `tanh`, `wrap`
   - No access to `__builtins__` or dangerous functions

5. **Failure Analysis**: Tracks x-threshold violations separately
   - Distinguishes pole balance failures from cart position control failures
   - Helps diagnose if symbolic reward over-optimizes one objective

## Installation

### Prerequisites
- Python 3.10+
- Julia 1.9+ (for PySR symbolic regression)

### Core Dependencies
```bash
# RAG & Retrieval
pip install "haystack-ai>=2.1.0"
pip install qdrant-client qdrant-haystack
pip install "sentence-transformers>=2.7.0"

# LLM
pip install openai

# Symbolic Regression
pip install pysr
python -c "import pysr; pysr.install()"

# Reinforcement Learning
pip install "stable-baselines3[extra]"
pip install gymnasium

# Utilities
pip install numpy pandas matplotlib
pip install torch  # CPU version is sufficient
```

## Usage

### 1. Ingest RAG Documents

Parse PDFs and index into Qdrant + BM25:

```bash
python haystack_ingestion.py \
  --marker-root rag_parsed/marker \
  --qdrant-collection papers \
  --qdrant-persist ./qdrant_papers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128 \
  --bm25-cache ./bm25_cache.jsonl
```

### 2. Query RAG System (Optional)

Test retrieval before running full pipeline:

```bash
python haystack_query_pip.py \
  --qdrant-collection papers \
  --qdrant-persist ./qdrant_papers \
  --bm25-cache ./bm25_cache.jsonl \
  --query "lyapunov stability for inverted pendulum"
```

### 3. Run Full RAG-LLM-PPO Pipeline

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run end-to-end pipeline
python cartpole_rag_llm_ppo.py
```

**What happens**:
1. Retrieves relevant papers about CartPole control
2. LLM generates seed reward expression
3. PySR discovers compact symbolic function
4. Trains baseline and symbolic PPO agents
5. Evaluates and compares performance
6. Saves results to `runs_cartpole_llm_ppo/`

### 4. Analyze Results

Check output directory:
```bash
ls runs_cartpole_llm_ppo/
# learning_curves_ppo_1.png         # Visual comparison
# training_summary_ppo_1.csv        # Metrics table
# symbolic_expression_ppo_1.txt     # Discovered reward function
```

View summary:
```bash
cat runs_cartpole_llm_ppo/training_summary_ppo_1.csv
```

## Project Structure

```
gymtest/
├── rag_source/                      # Research papers (PDFs)
│   ├── Cart_and_Pendulum_(Lagrange).pdf
│   ├── cartPoleEqns.pdf
│   ├── Energy_shaping_control_revisited.pdf
│   ├── MIT_chapter3.pdf
│   ├── murray_lyapunov.pdf
│   └── phys239_2016_lec05.pdf
│
├── rag_parsed/                      # Parsed PDFs (Marker JSON)
│   └── marker/
│       └── [paper_name]/
│           ├── blocks.json
│           ├── content.json
│           └── meta.json
│
├── qdrant_papers/                   # Qdrant vector DB (persistent)
├── bm25_cache.jsonl                 # BM25 document cache (12MB)
│
├── haystack_ingestion.py            # RAG indexing pipeline
├── haystack_query_pip.py            # RAG query interface
├── cartpole_rag_llm_ppo.py          # Main end-to-end pipeline
│
├── runs_cartpole_llm_ppo/           # Experiment outputs
│   ├── learning_curves_ppo_1.png
│   ├── learning_curves_ppo_1.csv
│   ├── training_summary_ppo_1.csv
│   ├── seed_expr_1.txt
│   └── symbolic_expression_ppo_1.txt
│
└── [other experiments and baselines]
```

## Key Features

✅ **Hybrid Retrieval**: Combines dense (semantic) and sparse (keyword) search with RRF fusion  
✅ **Cross-Encoder Reranking**: Improves relevance of top results  
✅ **LLM-Guided Reward Design**: Uses research context to generate reward functions  
✅ **Symbolic Regression**: Discovers interpretable, compact reward expressions  
✅ **Stable RL Training**: PPO with early stopping and failure analysis  
✅ **Reproducible**: Fixed random seeds (SEED=42) throughout pipeline  
✅ **Persistent Storage**: Qdrant DB and BM25 cache avoid re-indexing  

## Performance Monitoring

The pipeline tracks:
- **Episode Returns**: Smoothed learning curves over timesteps
- **AUC (Area Under Curve)**: Integral of returns over training
- **Final Return**: Last episode performance
- **X-Threshold Failures**: Rate of cart position limit violations
- **Early Stop**: Whether agent reached 475 average return threshold

## Related Files

- **`cartpole_ppo.py`**: Baseline PPO without RAG or symbolic rewards
- **`safe_rl_cartpole_comparison.py`**: Safe RL comparison experiments
- **`official_ppo_comparison.py`**: PPO hyperparameter comparison
- **`baseline_test.py`**: Simple CartPole baseline

## References

- [Haystack Documentation](https://haystack.deepset.ai/)
- [Qdrant Vector Database](https://qdrant.tech/)
- [PySR Symbolic Regression](https://github.com/MilesCranmer/PySR)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Environments](https://gymnasium.farama.org/)

## Citation

If you use this work, please cite:
```bibtex
@software{cartpole_rag_llm_ppo,
  title={CartPole RAG-LLM-PPO: Retrieval-Augmented Symbolic Reward Discovery},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gymtest}
}
```

## License

MIT License (or specify your license)

---

**Questions or Issues?** Open an issue on GitHub or contact the maintainer.

