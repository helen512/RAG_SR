from cartpole_ppo2_no_termination import ModifiedCartPoleWrapper
import gymnasium as gym
import numpy as np

def test_wrapper():
    env = gym.make('InvertedPendulum-v4')
    # Test standard
    wrapper_std = ModifiedCartPoleWrapper(env, use_custom_reward=False)
    obs, info = wrapper_std.reset()
    assert len(obs) == 4
    obs, reward, term, trunc, info = wrapper_std.step(np.array([0.0]))
    print(f"Standard reward: {reward}")

    # Test custom PBRS
    wrapper_custom = ModifiedCartPoleWrapper(env, use_custom_reward=True)
    obs, info = wrapper_custom.reset()
    prev_pot = wrapper_custom.prev_potential
    print(f"Initial potential: {prev_pot}")
    
    action = np.array([0.0])
    obs, reward, term, trunc, info = wrapper_custom.step(action)
    curr_pot = wrapper_custom.prev_potential
    print(f"New potential: {curr_pot}")
    
    # Calculate expected reward
    # base is 1 if |theta| <= 0.2 else 0
    theta = obs[1]
    base = 1.0 if abs(theta) <= 0.2 else 0.0
    gamma = 0.99
    # Note: In step(), self.prev_potential is updated to current_potential after calculation.
    # So wrapper_custom.prev_potential is now current_potential (Phi(s')).
    # We need the OLD prev_pot (Phi(s)) which we stored in 'prev_pot' variable.
    
    expected_shaping = gamma * curr_pot - prev_pot
    expected_reward = base + expected_shaping
    
    print(f"Reward: {reward}, Expected: {expected_reward}")
    assert np.isclose(reward, expected_reward), f"Reward mismatch: {reward} != {expected_reward}"
    print("Test passed!")

if __name__ == "__main__":
    test_wrapper()

