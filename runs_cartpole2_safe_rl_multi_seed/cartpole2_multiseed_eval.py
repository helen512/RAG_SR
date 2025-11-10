#!/usr/bin/env python3
"""
Quick evaluation script for cartpole2 safe RL policies.
Run this from the runs_cartpole2_safe_rl_multi_seed directory.
"""

import os
import sys

# Add parent directory to path to import the main evaluation script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import and run the main evaluation
try:
    print("Loading evaluation script...")
    exec(open(os.path.join(parent_dir, 'cartpole2_multiseed_evaluation.py')).read())
except Exception as e:
    print(f"Error running evaluation: {e}")
    print("\nMake sure to:")
    print("1. Activate the conda 'safe' environment: conda activate safe")
    print("2. Run from the correct directory")
    print("3. Ensure all required libraries are installed")

