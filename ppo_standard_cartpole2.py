"""
PPO Training on Standard InvertedPendulum-v4 (MuJoCo) using safe-control-gym PPO implementation
"""

import gymnasium as gym
from functools import partial
import numpy as np

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make

def train_ppo_controller(save_model=True):
    print("=" * 60)
    print("Training PPO on InvertedPendulum-v4 (MuJoCo)")
    print("=" * 60)

    # Custom wrapper to add required attributes for safe-control-gym
    class SafeGymWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.CTRL_FREQ = 1  # Dummy for step-based env
            self.EPISODE_LEN_SEC = 1000  # Matches InvertedPendulum-v4 default max steps
            self._state = None  # Initialize state tracking
        
        @property
        def state(self):
            """Return the current state (same as observation for standard gym envs)"""
            return self._state if self._state is not None else np.zeros(self.observation_space.shape)
        
        def step(self, action):
            # Ensure action is properly shaped for continuous action space
            action = np.atleast_1d(action).flatten()
            # Call step and convert from Gymnasium API (5 values) to Gym API (4 values)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self._state = obs  # Track current state
            return obs, reward, done, info
        
        def reset(self, **kwargs):
            # Keep Gymnasium API for reset (obs, info)
            obs, info = self.env.reset(**kwargs)
            self._state = obs  # Track initial state
            return obs, info
    
    # Environment function for standard Gym InvertedPendulum
    def env_func(seed=None):
        env = gym.make('InvertedPendulum-v4')
        env = SafeGymWrapper(env)
        return env

    # PPO configuration
    ppo_config = {
        'training': True,
        'hidden_dim': 64,
        'activation': 'tanh',
        'norm_obs': False,
        'norm_reward': False,
        'clip_obs': 10,
        'clip_reward': 10,
        'gamma': 0.99,
        'use_gae': True,
        'gae_lambda': 0.95,
        'use_clipped_value': False,
        'clip_param': 0.2,
        'target_kl': 0.01,
        'entropy_coef': 0.01,
        'opt_epochs': 10,
        'mini_batch_size': 64,
        'actor_lr': 0.0003,
        'critic_lr': 0.001,
        'max_grad_norm': 0.5,
        'max_env_steps': 100000,  # Training steps for InvertedPendulum
        'num_workers': 1,
        'rollout_batch_size': 4,
        'rollout_steps': 100,
        'deque_size': 10,
        'eval_batch_size': 10,
        'log_interval': 1000,
        'save_interval': 0,
        'num_checkpoints': 0,
        'eval_interval': 0,
        'eval_save_best': False,
        'tensorboard': False
    }

    # Create training and evaluation environments
    train_env = env_func()
    eval_env = env_func()

    # Create PPO controller
    print("\nCreating PPO controller...")
    ppo_controller = make('ppo', env_func, **ppo_config, 
                         output_dir='./temp_ppo_inverted_pendulum', 
                         checkpoint_path='./temp_ppo_inverted_pendulum/model_checkpoint.pt')

    # Train the controller
    print("\nTraining PPO...")
    experiment = BaseExperiment(env=eval_env, ctrl=ppo_controller, train_env=train_env)
    experiment.launch_training()

    # Note: Training metrics are logged to console and can be parsed from log files
    # or viewed via tensorboard if enabled. See plot_training_results.py for 
    # an example of parsing and plotting training progress.

    # Save trained model
    if save_model:
        ppo_controller.save('./models/ppo_inverted_pendulum.pt')
        print("\nPPO model saved to './models/ppo_inverted_pendulum.pt'")

    train_env.close()
    eval_env.close()
    return ppo_controller, eval_env, env_func

if __name__ == '__main__':
    train_ppo_controller()
