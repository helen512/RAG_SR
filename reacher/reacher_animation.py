import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Create environment with rgb_array mode
env = gym.make("Reacher-v5", render_mode="rgb_array")

# Wrap to record video
env = RecordVideo(env, video_folder="reacher_videos", episode_trigger=lambda x: True)

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()