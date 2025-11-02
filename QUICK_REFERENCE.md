# Quick Reference: Compatibility Changes

## Key Changes at a Glance

### 1. TensorFlow Import Pattern
**Every file importing TensorFlow needs this change:**

**Before:**
```python
import tensorflow as tf
```

**After:**
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

---

### 2. Gym/Gymnasium Import Pattern

**Before:**
```python
import gym
from gym.spaces import Box, Discrete
```

**After:**
```python
try:
    import gymnasium as gym
except ImportError:
    import gym

try:
    from gymnasium.spaces import Box, Discrete
except ImportError:
    from gym.spaces import Box, Discrete
```

---

### 3. Keras Layers API

**Before:**
```python
x = tf.layers.dense(x, units=h, activation=activation)
```

**After:**
```python
x = tf.compat.v1.layers.dense(x, units=h, activation=activation)
```

---

### 4. Environment Reset/Step Compatibility

**For reset():**
```python
def safe_reset(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]  # Gymnasium: (obs, info)
    else:
        return result      # Gym: obs
```

**For step():**
```python
def safe_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        # Gymnasium: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info
    else:
        # Gym: (obs, reward, done, info)
        return result
```

---

### 5. Environment Variable for Legacy Keras

**At the very top of your main script (before any TensorFlow imports):**
```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
```

---

## File-by-File Checklist

### safety-starter-agents/safe_rl/

- [x] `utils/mpi_tf.py` - TF compat import
- [x] `pg/run_agent.py` - TF compat import + gym compat helpers
- [x] `pg/network.py` - TF compat import + gym compat + layers fix
- [x] `pg/trust_region.py` - TF compat import
- [x] `utils/load_utils.py` - TF compat import
- [x] `utils/logx.py` - TF compat import
- [x] `sac/sac.py` - TF compat import + gym compat + layers fix

### Your main script

- [x] `safe_rl_cartpole_comparison.py` - Keras env var + TF compat + wrapper fixes

---

## Dependencies to Install

```bash
# Activate your conda environment
conda activate safe

# Install required packages
pip install mpi4py      # For MPI support
pip install tf_keras    # For legacy Keras with TensorFlow 2.20
```

---

## Testing

Run a quick test:
```bash
conda activate safe
cd /home/dmy/gymtest
python -c "
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gymnasium as gym
print('✓ All imports successful!')
print(f'TensorFlow version: {tf.__version__}')
print(f'Gymnasium version: {gym.__version__}')
"
```

Expected output:
```
✓ All imports successful!
TensorFlow version: 2.20.0
Gymnasium version: 0.28.1
```

---

## Common Errors and Solutions

### Error: `AttributeError: module 'tensorflow._api.v2.train' has no attribute 'AdamOptimizer'`
**Solution:** Use `import tensorflow.compat.v1 as tf` and `tf.disable_v2_behavior()`

### Error: `AttributeError: 'dense' is not available with Keras 3`
**Solution:** 
1. Install tf_keras: `pip install tf_keras`
2. Set env var: `os.environ['TF_USE_LEGACY_KERAS'] = '1'`
3. Use `tf.compat.v1.layers.dense()` instead of `tf.layers.dense()`

### Error: `ModuleNotFoundError: No module named 'mpi4py'`
**Solution:** `pip install mpi4py`

### Error: `TypeError: tuple indices must be integers or slices, not NoneType`
**Solution:** Use gymnasium-compatible reset/step helpers

### Error: `ValueError: too many values to unpack (expected 4)`
**Solution:** Handle both gym (4 returns) and gymnasium (5 returns) in step()

---

## Verification Script

Save this as `test_compatibility.py`:

```python
#!/usr/bin/env python3
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

try:
    import gymnasium as gym
except ImportError:
    import gym

# Test TensorFlow
print("✓ TensorFlow v1 compatibility mode enabled")

# Test environment creation
env = gym.make('CartPole-v1')
obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]
print(f"✓ Environment created, observation shape: {obs.shape}")

# Test step
result = env.step(env.action_space.sample())
if len(result) == 5:
    obs, reward, terminated, truncated, info = result
    print("✓ Gymnasium API detected")
else:
    obs, reward, done, info = result
    print("✓ Gym API detected")

print("\n✅ All compatibility checks passed!")
```

Run with:
```bash
conda activate safe
python test_compatibility.py
```




