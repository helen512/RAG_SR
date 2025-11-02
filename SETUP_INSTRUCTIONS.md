# Setup Instructions for safety-starter-agents with Current Environment

## Summary
The safety-starter-agents repository has been successfully modified to work with your current environment:
- Python 3.10.19
- TensorFlow 2.20.0
- Gymnasium 0.28.1
- NumPy 2.2.6

## What Was Fixed

### 1. TensorFlow 1.x → 2.x Compatibility
- Changed all TensorFlow imports to use compatibility mode
- Added `tf.disable_v2_behavior()` to enable TF1 behavior
- Updated 7 files in safety-starter-agents

### 2. Keras 3 → Legacy Keras
- Installed `tf_keras` package
- Set `TF_USE_LEGACY_KERAS=1` environment variable
- Updated all `tf.layers.dense()` calls to `tf.compat.v1.layers.dense()`

### 3. Gym → Gymnasium Compatibility
- Added runtime detection for both APIs
- Created compatibility helpers for reset() and step()
- Works with both old gym and new gymnasium environments

### 4. Dependencies
- Installed `mpi4py` for MPI support
- Installed `tf_keras` for Keras 2 compatibility

## Files Modified

### In safety-starter-agents directory:
```
safe_rl/
├── pg/
│   ├── network.py          (TF compat + gym compat + layers fix)
│   ├── run_agent.py        (TF compat + gym compat helpers)
│   └── trust_region.py     (TF compat)
├── sac/
│   └── sac.py             (TF compat + gym compat + layers fix)
└── utils/
    ├── load_utils.py       (TF compat)
    ├── logx.py            (TF compat)
    └── mpi_tf.py          (TF compat)
```

### In your project:
```
safe_rl_cartpole_comparison.py  (Keras env var + TF compat + wrapper fixes)
```

## How to Run

```bash
# Activate conda environment
conda activate safe

# Run your script
cd /home/dmy/gymtest
python safe_rl_cartpole_comparison.py
```

The code will now run without compatibility errors!

## Verification

You can verify the setup works with:

```bash
conda activate safe
python -c "
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gymnasium as gym
from mpi4py import MPI
print('✅ All dependencies loaded successfully!')
"
```

## What's Still Compatible

The modified code maintains backward compatibility:
- Works with both gym and gymnasium
- Works with TensorFlow 1.x (if you want to downgrade)
- Works with TensorFlow 2.x (current setup)

## Next Steps

Your `safe_rl_cartpole_comparison.py` script should now run successfully. The training will:
1. Train PPO (standard)
2. Train PPO-Lagrangian (constrained)
3. Train CPO (constrained)
4. Train PPO with CBF safety filter
5. Generate comparison plots

## Troubleshooting

If you encounter any issues:

1. **Check conda environment is activated**: `conda activate safe`
2. **Verify packages installed**: `pip list | grep -E "(tensorflow|gym|mpi4py|tf-keras)"`
3. **Check environment variable**: Echo should show "1": `echo $TF_USE_LEGACY_KERAS`

## Additional Documentation

- `COMPATIBILITY_FIXES_SUMMARY.md` - Detailed technical explanation
- `QUICK_REFERENCE.md` - Code patterns and common fixes
- This file - Setup and usage instructions
