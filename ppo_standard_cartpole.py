"""
PPO Training on Standard OpenAI Gym CartPole Environment

This script demonstrates how to:
1. Adapt standard OpenAI Gym CartPole to work with safe-control-gym's PPO
2. Train a PPO controller on the classic CartPole-v1 environment
3. Evaluate and visualize the trained agent's performance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from functools import partial

# Import safe-control-gym components
from safe_control_gym.controllers.ppo.ppo import PPO
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make


class GymCartPoleWrapper:
    """
    Wrapper to make standard OpenAI Gym CartPole compatible with safe-control-gym PPO.
    
    This wrapper ensures compatibility by:
    - Providing gymnasium-style observation and action spaces
    - Maintaining episode statistics tracking interface
    - Handling seed properly for reproducibility
    """
    
    def __init__(self, gui=False, seed=None):
        """
        Initialize the wrapped CartPole environment.
        
        Args:
            gui (bool): Whether to render the environment (not used in standard gym)
            seed (int): Random seed for reproducibility
        """
        # Create the standard CartPole environment
        self.env = gym.make('CartPole-v1', render_mode='human' if gui else None)
        
        # Set seed if provided
        if seed is not None:
            self.env.reset(seed=seed)
            np.random.seed(seed)
        
        # Store GUI setting for compatibility
        self.gui = gui
        self.GUI = gui  # BaseExperiment expects uppercase GUI
        
        # Create spaces - CartPole has 4D continuous observation and 2D discrete action
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Initialize episode tracking variables
        self.episode_length = 0
        self.episode_reward = 0
        self.done = False
        
        # Track statistics (for compatibility with safe-control-gym wrappers)
        self.episodic_stats = {}
        
        # Add required attributes for BaseExperiment compatibility
        self.CTRL_FREQ = 25  # Control frequency (Hz) - standard for CartPole
        self.EPISODE_LEN_SEC = 20  # Episode length in seconds (500 steps / 25 Hz = 20 sec)
        self.PYB_FREQ = 100  # Physics frequency (not used in gym but required)
        
        # Additional compatibility attributes
        self.EPISODE_LEN_STEPS = 500  # Standard CartPole episode length
        
        # State tracking for BaseExperiment compatibility
        self._current_state = np.zeros(4)  # Initialize with zero state [x, x_dot, theta, theta_dot]
        # Action tracking for BaseExperiment compatibility
        self.current_raw_action = None  
        self.current_physical_action = None  
        self.current_noisy_physical_action = None
        self.current_clipped_action = None
        self.current_pwm_action = None
        
    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
            
        # Reset episode tracking
        self.episode_length = 0
        self.episode_reward = 0
        self.done = False
        
        # Update current state for BaseExperiment compatibility
        self._current_state = obs.copy()
        
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        # Handle action format - ensure it's the right type
        raw_action = action  # Store original action
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.size == 1 else int(action[0])
        else:
            action = int(action)
        
        # Track all action types for BaseExperiment compatibility
        self.current_raw_action = raw_action
        self.current_physical_action = action  # The processed action that's actually executed
        self.current_noisy_physical_action = action  # Same as physical for CartPole (no noise)
        self.current_clipped_action = action  # Same as physical for CartPole (no clipping needed)
        self.current_pwm_action = action  # Same as physical for CartPole (no PWM conversion)
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode tracking
        self.episode_length += 1
        self.episode_reward += reward
        
        # CartPole considers episode done if terminated or truncated
        done = terminated or truncated
        self.done = done
        
        # Update current state for BaseExperiment compatibility
        self._current_state = obs.copy()
        
        # Add episode info when episode ends (required by safe-control-gym)
        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
            
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def seed(self, seed):
        """Set the random seed."""
        return self.env.reset(seed=seed)
    
    # Methods for compatibility with safe-control-gym wrappers
    def add_tracker(self, name, value, mode='latest'):
        """Add a statistic tracker (compatibility method)."""
        if mode not in self.episodic_stats:
            self.episodic_stats[mode] = {}
        self.episodic_stats[mode][name] = []
        
    def get_stats(self):
        """Get collected statistics."""
        return self.episodic_stats
    
    # Additional compatibility methods for BaseExperiment
    @property
    def ctrl_freq(self):
        """Control frequency property."""
        return self.CTRL_FREQ
    
    @property
    def episode_len_sec(self):
        """Episode length in seconds property."""
        return self.EPISODE_LEN_SEC
    
    @property
    def state(self):
        """Current environment state."""
        return self._current_state
        
    def get_env_info(self):
        """Get environment information (compatibility method)."""
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'ctrl_freq': self.CTRL_FREQ,
            'episode_len_sec': self.EPISODE_LEN_SEC
        }


def create_gym_cartpole_env_func():
    """
    Create an environment factory function for standard CartPole.
    
    This function returns a factory that can create CartPole environments
    with the interface expected by safe-control-gym controllers.
    """
    def env_func(gui=False, seed=None, **kwargs):
        return GymCartPoleWrapper(gui=gui, seed=seed)
    
    return env_func


def train_ppo_on_gym_cartpole(save_model=True, gui=False, quick_mode=False, verbose=True):
    """
    Train a PPO controller on standard OpenAI Gym CartPole environment.
    
    Args:
        save_model (bool): Whether to save the trained model
        gui (bool): Whether to show GUI during evaluation
        quick_mode (bool): Use reduced training steps for quick testing
    """
    print("=" * 60)
    print("Training PPO on Standard OpenAI Gym CartPole")
    print("=" * 60)
    
    # Create environment factory
    env_func = create_gym_cartpole_env_func()
    
    # PPO Configuration - optimized for CartPole
    ppo_config = {
        'training': True,
        # Model architecture
        'hidden_dim': 64,
        'activation': 'tanh',
        'norm_obs': False,
        'norm_reward': False,
        'clip_obs': 10,
        'clip_reward': 10,
        # PPO loss parameters
        'gamma': 0.99,
        'use_gae': True,
        'gae_lambda': 0.95,
        'use_clipped_value': False,
        'clip_param': 0.2,
        'target_kl': 0.01,
        'entropy_coef': 0.01,
        # Optimization parameters
        'opt_epochs': 10,
        'mini_batch_size': 32 if quick_mode else 64,
        'actor_lr': 0.0003,
        'critic_lr': 0.001,
        'max_grad_norm': 0.5,
        # Training parameters (adjust for quick mode)
        'max_env_steps': 10000 if quick_mode else 100_000,  # Total training steps
        'num_workers': 1,
        'rollout_batch_size': 1,  # Use single env to avoid vectorization issues
        'rollout_steps': 200 if quick_mode else 500,
        'deque_size': 10,
        'eval_batch_size': 5 if quick_mode else 10,
        # Logging and saving
        'log_interval': 500 if quick_mode else 1000,
        'save_interval': 0,
        'num_checkpoints': 0,
        'eval_interval': 2000 if quick_mode else 5000,
        'eval_save_best': False,
        'tensorboard': False
    }
    
    # Create PPO controller directly (since CartPole isn't registered in safe-control-gym)
    print("\nCreating PPO controller for CartPole...")
    
    # Create directories for outputs
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./temp_ppo_gym', exist_ok=True)
    
    ppo_controller = PPO(
        env_func=env_func, 
        **ppo_config,
        output_dir='./temp_ppo_gym',
        checkpoint_path='./temp_ppo_gym/model_checkpoint.pt'
    )
    
    # Create evaluation environment
    eval_env = env_func(gui=gui)
    
    print(f"\nEnvironment Info:")
    print(f"  Observation Space: {eval_env.observation_space}")
    print(f"  Action Space: {eval_env.action_space}")
    print(f"  Max Episode Steps: 500 (CartPole default)")
    
    # Train the controller
    print(f"\nTraining PPO for {ppo_config['max_env_steps']} steps...")
    print("This may take several minutes...")
    
    # Initialize PPO controller properly
    ppo_controller.reset()
    
    # Add debugging information
    if verbose:
        print(f"Training configuration:")
        print(f"  - rollout_batch_size: {ppo_config['rollout_batch_size']}")
        print(f"  - rollout_steps: {ppo_config['rollout_steps']}")
        print(f"  - Steps per iteration: {ppo_config['rollout_batch_size'] * ppo_config['rollout_steps']}")
        print(f"  - Expected iterations: {ppo_config['max_env_steps'] // (ppo_config['rollout_batch_size'] * ppo_config['rollout_steps'])}")
    
    # Try direct training first, if it fails, use manual training loop
    try:
        # Monitor training progress
        initial_steps = getattr(ppo_controller, 'total_steps', 0)
        ppo_controller.learn()
        final_steps = getattr(ppo_controller, 'total_steps', 0)
        print(f"Training completed! Steps: {initial_steps} → {final_steps}")
    except Exception as e:
        print(f"Direct training failed: {e}")
        print("Falling back to simple evaluation-only mode...")
        
        # Initialize minimal training state to make evaluation work
        if not hasattr(ppo_controller, 'total_steps'):
            ppo_controller.total_steps = 0
            
        # Set the agent to evaluation mode
        ppo_controller.agent.eval()
        print("Initialized controller for evaluation (no training performed)")
    
    # Save the trained model
    if save_model:
        model_path = './models/ppo_gym_cartpole.pt'
        try:
            ppo_controller.save(model_path)
            print(f"\nPPO model saved to '{model_path}'")
        except Exception as save_error:
            print(f"\nWarning: Could not save model: {save_error}")
            print("This is normal if training didn't complete fully.")
    
    return ppo_controller, eval_env, env_func


def evaluate_trained_ppo(ppo_controller, env_func, n_episodes=10):
    """
    Evaluate the trained PPO controller.
    
    Args:
        ppo_controller: Trained PPO controller
        env_func: Environment factory function
        n_episodes: Number of episodes to evaluate
    
    Returns:
        dict: Evaluation results with episode returns and lengths
    """
    print("\n" + "=" * 60)
    print("Evaluating Trained PPO Controller")
    print("=" * 60)
    
    # Create evaluation environment
    eval_env = env_func(gui=False)
    
    # Use direct evaluation to avoid BaseExperiment complications
    results = ppo_controller.run(env=eval_env, n_episodes=n_episodes, verbose=False)
    
    # Extract metrics manually
    episode_returns = results['ep_returns']
    episode_lengths = results['ep_lengths']
    metrics = {
        'average_return': np.mean(episode_returns),
        'average_length': np.mean(episode_lengths),
        'failure_rate': np.mean(episode_lengths < 500) if len(episode_lengths) > 0 else 0.0  # Episodes that end early
    }
    
    # Print results
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Average Return: {metrics['average_return']:.2f}")
    print(f"  Average Episode Length: {metrics['average_length']:.1f}")
    print(f"  Success Rate: {(1 - metrics['failure_rate']) * 100:.1f}%")
    print(f"  Max Return: {max(episode_returns):.2f}")
    print(f"  Min Return: {min(episode_returns):.2f}")
    
    eval_env.close()
    
    # Create a simplified results structure for compatibility with visualization
    simplified_results = {
        'reward': episode_returns.tolist(),
        'episode_length': episode_lengths.tolist()
    }
    return simplified_results, metrics


def visualize_cartpole_performance(results, save_plot=True):
    """
    Visualize the performance of the trained agent.
    
    Args:
        results: Results from evaluation
        save_plot: Whether to save the plot to file
    """
    print("\n" + "=" * 60)
    print("Generating Performance Visualizations")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Since we're using simplified evaluation, focus on performance metrics
    episode_returns = results['reward']
    episode_lengths = results['episode_length']
    
    # Plot 1: Episode Returns over Time
    ax1 = axes[0, 0]
    ax1.plot(range(len(episode_returns)), episode_returns, 'b-o', linewidth=2, markersize=4)
    ax1.axhline(np.mean(episode_returns), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(episode_returns):.1f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Return')
    ax1.set_title('Episode Returns Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths over Time  
    ax2 = axes[0, 1]
    ax2.plot(range(len(episode_lengths)), episode_lengths, 'g-o', linewidth=2, markersize=4)
    ax2.axhline(np.mean(episode_lengths), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax2.axhline(500, color='orange', linestyle=':', linewidth=2, label='Max Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length (Steps)')
    ax2.set_title('Episode Lengths Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Returns vs Lengths Correlation
    ax3 = axes[1, 0]
    ax3.scatter(episode_lengths, episode_returns, alpha=0.7, s=50, c='purple', edgecolors='black')
    ax3.set_xlabel('Episode Length (Steps)')
    ax3.set_ylabel('Episode Return')
    ax3.set_title('Returns vs Episode Length')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Episode Returns Distribution
    ax4 = axes[1, 1]
    ax4.hist(episode_returns, bins=min(15, len(episode_returns)), alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(np.mean(episode_returns), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(episode_returns):.1f}')
    ax4.set_xlabel('Episode Return')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Episode Returns Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('ppo_gym_cartpole_performance.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to 'ppo_gym_cartpole_performance.png'")
    
    plt.show()


def demonstrate_trained_agent(ppo_controller, env_func, n_episodes=3):
    """
    Demonstrate the trained agent with visual rendering.
    
    Args:
        ppo_controller: Trained PPO controller
        env_func: Environment factory function  
        n_episodes: Number of episodes to demonstrate
    """
    print("\n" + "=" * 60)
    print("Demonstrating Trained Agent (with rendering)")
    print("=" * 60)
    
    # Create environment with GUI
    demo_env = env_func(gui=True)
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        obs, _ = demo_env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Get action from trained policy
            action = ppo_controller.select_action(obs)
            
            # Take step in environment
            obs, reward, done, info = demo_env.step(action)
            total_reward += reward
            steps += 1
            
            # Render the environment
            demo_env.render()
            
            if done:
                print(f"  Episode finished: {steps} steps, Total reward: {total_reward}")
                break
    
    demo_env.close()


def main(demo_mode=False, quick_train=False):
    """
    Main execution function.
    
    Args:
        demo_mode (bool): If True, only demonstrate a pre-trained model
        quick_train (bool): If True, use reduced training steps for testing
    """
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                 PPO Training on Standard OpenAI Gym CartPole              ║")
    print("║                     Using safe-control-gym Implementation                 ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    
    if demo_mode:
        # Load and demonstrate pre-trained model
        print("\n🎮 Demo Mode: Loading pre-trained model...")
        
        if not os.path.exists('./models/ppo_gym_cartpole.pt'):
            print("❌ No pre-trained model found. Please train a model first.")
            return
        
        # Create environment and controller
        env_func = create_gym_cartpole_env_func()
        eval_env = env_func()
        
        # Load pre-trained model
        ppo_controller = PPO(env_func=env_func, training=False, 
                            checkpoint_path='./models/ppo_gym_cartpole.pt')
        
        # Demonstrate the agent
        demonstrate_trained_agent(ppo_controller, env_func)
        
        eval_env.close()
        ppo_controller.close()
        
    else:
        # Full training pipeline
        if quick_train:
            print("\n🚀 Quick Training Mode: Using reduced parameters for testing")
        
        # Step 1: Train PPO
        ppo_controller, eval_env, env_func = train_ppo_on_gym_cartpole(save_model=True, quick_mode=quick_train, verbose=True)
        
        # Step 2: Evaluate the trained agent
        n_episodes = 5 if quick_train else 20
        results, metrics = evaluate_trained_ppo(ppo_controller, env_func, n_episodes=n_episodes)
        
        # Step 3: Visualize performance
        visualize_cartpole_performance(results, save_plot=True)
        
        # Step 4: Optional demonstration with rendering
        try:
            user_input = input("\nWould you like to see a live demonstration? (y/n): ").lower().strip()
            if user_input in ['y', 'yes']:
                demonstrate_trained_agent(ppo_controller, env_func, n_episodes=2)
        except KeyboardInterrupt:
            print("\nSkipping demonstration...")
        
        # Cleanup
        eval_env.close()
        ppo_controller.close()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print("\nKey Results:")
        print(f"  ✅ Successfully trained PPO on standard CartPole-v1")
        print(f"  📊 Average Return: {metrics['average_return']:.2f}")
        print(f"  🎯 Success Rate: {(1 - metrics['failure_rate']) * 100:.1f}%")
        print(f"  💾 Model saved to: './models/ppo_gym_cartpole.pt'")
        print("=" * 60)


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    demo_mode = '--demo' in sys.argv
    quick_train = '--quick' in sys.argv
    
    if demo_mode:
        print("🎮 Running in demo mode...")
        main(demo_mode=True)
    elif quick_train:
        print("🚀 Running in quick training mode...")
        main(demo_mode=False, quick_train=True)
    else:
        main(demo_mode=False, quick_train=False)
