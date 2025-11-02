# PPO Training on Standard Gymnasium Environments with safe-control-gym

## Overview
Successfully trained a PPO agent from **safe-control-gym** on the standard **InvertedPendulum-v4** environment from Gymnasium (MuJoCo).

## Key Challenge: Discrete vs Continuous Action Spaces
- **Initial Attempt**: CartPole-v1 (discrete action space) - Failed due to incompatibility
  - safe-control-gym's PPO buffer expects actions shaped as `(batch_size, action_dim)`
  - For Discrete(2), this expects shape `(N, 2)`, but Categorical distribution outputs `(N,)` integers
  - This is a limitation in safe-control-gym's current PPO implementation for discrete actions

- **Solution**: Switched to InvertedPendulum-v4 (continuous action space)
  - Box action space with 1 continuous action dimension
  - Works seamlessly with safe-control-gym's PPO implementation

## Implementation Details

### Custom Wrapper: `SafeGymWrapper`
Created a wrapper to bridge standard Gymnasium environments with safe-control-gym's expectations:

```python
class SafeGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.CTRL_FREQ = 1              # Required by BaseExperiment
        self.EPISODE_LEN_SEC = 1000     # Required by BaseExperiment
        self._state = None              # State tracking for BaseExperiment
    
    @property
    def state(self):
        """BaseExperiment expects env.state attribute"""
        return self._state if self._state is not None else np.zeros(self.observation_space.shape)
    
    def step(self, action):
        # Ensure action is properly shaped
        action = np.atleast_1d(action).flatten()
        # Convert Gymnasium API (5 values) to Gym API (4 values)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._state = obs
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._state = obs
        return obs, info
```

### Training Configuration
- **Environment**: InvertedPendulum-v4 (MuJoCo)
- **Algorithm**: PPO from safe-control-gym
- **Total Training Steps**: 100,000
- **Workers**: 1
- **Rollout Batch Size**: 4
- **Rollout Steps**: 100
- **Hidden Dimensions**: 64
- **Learning Rates**:
  - Actor: 0.0003
  - Critic: 0.001
- **PPO Parameters**:
  - Clip param: 0.2
  - Target KL: 0.01
  - Entropy coefficient: 0.01
  - GAE lambda: 0.95
  - Gamma: 0.99

## Training Results

### Performance Metrics
- **Initial Average Return**: 29.9
- **Final Average Return**: 1000.0 (maximum possible)
- **Total Improvement**: +970.1 (3244.5% increase)
- **Steps to Max Performance**: 34,000 steps
- **Training Time**: ~90 seconds

### Learning Curve
The agent achieved the following progression:
- 2k steps: 29.9 return
- 4k steps: 62.3 return
- 6k steps: 73.3 return
- 8k steps: 129 return
- 10k steps: 204 return
- 34k steps: 1000 return (maxed out)
- 100k steps: Maintained 1000 return

The agent learned to balance the inverted pendulum perfectly and maintain it for the maximum episode length of 1000 steps.

## Files Generated
1. **ppo_standard_cartpole_example.py** - Main training script
2. **plot_training_results.py** - Log parser and plotting utility
3. **ppo_training_plot.png** - Training progress visualization
4. **temp_ppo_inverted_pendulum/model_checkpoint.pt** - Trained model checkpoint
5. **training_full.log** - Complete training log with all metrics

## How to Use

### Training
```bash
conda activate safe
python ppo_standard_cartpole_example.py
```

### Plotting Results
```bash
python plot_training_results.py
```

### Loading Trained Model
```python
from safe_control_gym.utils.registration import make

# Create environment
env_func = lambda: SafeGymWrapper(gym.make('InvertedPendulum-v4'))

# Load PPO controller
ppo_controller = make('ppo', env_func, training=False,
                     checkpoint_path='./temp_ppo_inverted_pendulum/model_checkpoint.pt')

# Run evaluation
results, metrics = ppo_controller.run(env=env_func(), n_episodes=10)
```

## Key Takeaways

1. **safe-control-gym PPO works with standard Gymnasium environments** with proper wrapping
2. **Continuous action spaces** (Box) work seamlessly
3. **Discrete action spaces** (Discrete) require buffer modifications in safe-control-gym
4. **API Compatibility**: Need to bridge Gymnasium (5-value step return) with safe-control-gym (4-value expected)
5. **Required Attributes**: `CTRL_FREQ`, `EPISODE_LEN_SEC`, and `state` must be added to standard envs
6. **Fast Training**: PPO converges quickly on InvertedPendulum-v4 (~34k steps to optimal)

## Recommendations for Future Work

1. **For Discrete Action Spaces**: Modify `PPOBuffer` in `ppo_utils.py` to handle discrete actions properly:
   - Change action buffer shape from `(T, N, act_dim)` to `(T, N, 1)` for discrete spaces
   - Ensure Categorical distribution outputs are properly handled

2. **For Other Continuous Control Tasks**: This same approach should work for:
   - Hopper-v4
   - Walker2d-v4
   - Ant-v4
   - Humanoid-v4
   - Any other MuJoCo or continuous control environment

3. **Enable Tensorboard**: Set `tensorboard: True` in config for real-time training visualization

4. **Hyperparameter Tuning**: The default parameters worked well, but could be optimized for:
   - Faster convergence
   - More stable learning
   - Better sample efficiency

## Dependencies
- safe-control-gym
- gymnasium >= 0.28.1
- mujoco >= 2.3.2 (install via `pip install gymnasium[mujoco]`)
- numpy
- matplotlib
- torch

## Conclusion
Successfully demonstrated that safe-control-gym's PPO implementation can be used to train agents on standard Gymnasium environments with continuous action spaces. The training was efficient and the agent achieved optimal performance.

