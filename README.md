# Safe RL & RAG+LLM Automate Reward Shaping

This repository contains implementations of Safe Reinforcement Learning algorithms (PPO, CPO, PPO-Lagrangian, CBF-based methods) and RAG automate energy based reward shaping for CartPole, Inverted Pendulum, and Reacher environments.

## 1. RAG (Retrieval-Augmented Generation)

This section implements a RAG pipeline to retrieve control theory concepts from academic papers to inform RL agents.

### Installation
```bash
pip install haystack-ai qdrant-client qdrant-haystack sentence-transformers torch accelerate
# Marker is used for PDF parsing (follow marker-pdf installation guide)
```

### Components
*   **Documents:** PDF sources located in `rag_source/`.
*   **Parser:** Maker ([Marker](https://github.com/datalab-to/marker)) is used to parse documents into markdown/JSON.
*   **Orchestration:** [Haystack](https://haystack.deepset.ai/) framework.
*   **Vector Database:** [Qdrant](https://qdrant.tech/) for dense retrieval.
*   **Retrieval:** Hybrid approach using BM25 (sparse, keyword-based) and Qdrant (dense, semantic).
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Reranking Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Usage
To build the RAG knowledge base (ingest documents):
```bash
python haystack_ingestion_md.py --marker-root rag_parsed/marker --qdrant-collection qdrant_papers
```

To run the query pipeline:
```bash
python haystack_query_pip.py --query "your control theory question"
```

## 2. RAG_LLM_Cartpole

This section combines RAG with Large Language Models to generate symbolic reward functions for the CartPole environment, which are then used to train a PPO agent.

### Installation
```bash
pip install gymnasium stable-baselines3 openai juliacall pysr
```
*Note: Julia must be installed for `juliacall` and `pysr`.*

### Usage
Run the end-to-end pipeline (Retrieval -> Symbolic Regression -> PPO):
```bash
python cartpole_rag_llm_ppo.py
```
This script retrieves relevant equations, uses an LLM to propose a seed reward expression, optimizes it using Symbolic Regression (PySR), and trains a PPO agent.

## 3. CartPole + Inverted Pendulum

This section implements various Safe RL algorithms on CartPole and Inverted Pendulum tasks.

### Installation
```bash
pip install gymnasium cvxpy scipy tensorflow==1.15  # Legacy TF required for safety-starter-agents
```
*Note: This section relies on the local `safety-starter-agents` package located in this repository.*

### Algorithms Implemented
*   **PPO:** Proximal Policy Optimization
*   **CPO:** Constrained Policy Optimization
*   **PPO-Lagrangian:** PPO with Lagrangian relaxation for constraints
*   **PPO+CBF:** PPO with Control Barrier Functions for safety filtering
*   **CBF Reward Shaping:** Augmenting reward with CBF-based potential

### Usage

**CartPole Comparison (PPO vs PPO-Lagrangian vs CPO vs PPO+CBF):**
```bash
python safe_rl_cartpole_comparison.py
```

**Inverted Pendulum (PPO, PPO+CBF, PPO+CBF+RewardShaping):**
```bash
python cartpole2_safe_rl_multi_seed.py
```
*Note: Despite the filename `cartpole2...`, this script is configured for the `InvertedPendulum-v4` environment.*

*To reporduce the the figures in the report,  *

## 4. Reacher

This section applies Safe RL to the Reacher environment, focusing on multi-joint constraints.

### Installation
```bash
pip install gymnasium gymnasium-robotics
```

### Algorithms Implemented
*   **PPO**
*   **PPO + CBF** (Control Barrier Functions)
*   **PPO + CBF + Reward Shaping** (Gaussian shaping based on correction magnitude)
*   **Potential-based Reward Shaping**

### Usage

**Run PPO, PPO+CBF, and CBF Reward Shaping:**
```bash
python reacher_script/reacher_ppo.py
python reacher_script/reacher_cbf.py
python reacher_script/reacher_cbf_reward.py
```

**Run Potential-based Reward Shaping:**
```bash
python reacher_script/reacher_custom_potentialbased.py
```

