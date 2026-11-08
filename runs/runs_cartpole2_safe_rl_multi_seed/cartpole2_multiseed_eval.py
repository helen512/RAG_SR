#!/usr/bin/env python3
"""
Quick evaluation script for cartpole2 safe RL policies.
Run this from the runs_cartpole2_safe_rl_multi_seed directory.
"""

import os
import sys

# Evaluation script lives in safe_rl/ at the repo root.
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
safe_rl_dir = os.path.join(repo_root, "safe_rl")
sys.path.insert(0, safe_rl_dir)

# Import and run the main evaluation
try:
    print("Loading evaluation script...")
    exec(open(os.path.join(safe_rl_dir, 'cartpole2_multiseed_evaluation.py')).read())
except Exception as e:
    print(f"Error running evaluation: {e}")
    print("\nMake sure to:")
    print("1. Activate the conda 'safe' environment: conda activate safe")
    print("2. Run from the correct directory")
    print("3. Ensure all required libraries are installed")

