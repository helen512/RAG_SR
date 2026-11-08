import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Define the policies to evaluate
# User should update the paths below
POLICIES = {
    "baseline": "ppo_reacher_baselinefinal_logs/ppo_reacher_final.zip",
    "baseline_constraint": "reacher_custom2_final/ppo_reacher_final.zip", 
    "energy": "reacher_custom2_final/ppo_reacher_final.zip",
    "energy_potential": "reacher_custom2potential_final/ppo_reacher_potentialbased_final.zip",
    "cbf": "reacher_cbf/ppo_cbf_final_c1_15_c2_70/final_model.zip",
    "cbf_reward": "runs_reacher_cbfreward/ppo_cbf_final_c1_15_c2_70/final_model.zip",
}

def evaluate_policy(name, model_path, num_episodes=100):
    print(f"\nEvaluating policy: {name}")
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Create environment
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    
    # Wrap in DummyVecEnv for SB3 compatibility
    env = DummyVecEnv([lambda: env])

    # Check for VecNormalize statistics
    model_dir = os.path.dirname(model_path)
    vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        print(f"Loading VecNormalize stats from: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False # Disable updates
        env.norm_reward = False # Don't normalize reward for evaluation (we want true return)
    else:
        print("Warning: VecNormalize stats not found. Policy might perform poorly if it expects normalized obs.")

    # Load model
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Metrics
    episode_returns = []
    episode_violations = [] # 1 if violation occurred in episode, 0 otherwise
    total_violation_steps = 0
    total_steps = 0
    last_10_distances = [] # Store average of last 10 distances for each episode

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        curr_return = 0
        violation_occurred = False
        
        # Track distances for this episode
        ep_distances = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Note: env is VecEnv, so return values are arrays
            # We assume num_envs=1
            # VecEnv returns: obs, rewards, dones, infos (4 values)
            reward = reward[0]
            done = done[0]
            info = info[0] # info is a list of dicts
            
            curr_return += reward
            total_steps += 1

            # Check constraint: -2.55 < theta2 < 2.55
            # Access underlying MuJoCo data
            try:
                # env is VecNormalize -> DummyVecEnv -> TimeLimit -> ReacherEnv
                # We need to unwrap to reach the base env
                base_env = env.envs[0].unwrapped
                theta2 = base_env.data.qpos[1]
                
                if not (-2.55 < theta2 < 2.55):
                    violation_occurred = True
                    total_violation_steps += 1
            except (AttributeError, Exception) as e:
                # Fallback if unwrapping fails or structure is different
                print(f"Error unwrapping environment: {e}")
                pass

            # Calculate Euclidean distance
            # Reacher-v5 obs (unnormalized): [cos(t1), cos(t2), sin(t1), sin(t2), target_x, target_y, av1, av2, tip_x, tip_y, ...]
            # However, 'obs' here might be normalized.
            # We should get physical coordinates from the unwrapped env to be safe and accurate, 
            # or use info if available. But easiest is to use the unwrapped env data.
            try:
                # qpos: [theta1, theta2, target_x, target_y]
                # qvel: [ang_vel1, ang_vel2, 0, 0]
                # But we need tip position.
                # base_env.data.site_xpos gives site positions. 
                # Reacher tip is usually a site.
                
                # Using the observation from the step is safer if normalization is handled correctly, 
                # but if obs is normalized, we can't easily invert it without the VecNormalize object details.
                # Actually, VecNormalize normalizes the OUTPUT of step.
                # To calculate distance, we can rely on the environment's internal state.
                
                tip_pos = base_env.data.body("fingertip").xpos[:2]
                target_pos = base_env.data.body("target").xpos[:2]
                dist = np.linalg.norm(target_pos - tip_pos)
                ep_distances.append(dist)
                
            except Exception as e:
                # Fallback to observation indices if possible, but normalization makes this risky.
                # Let's try to get unnormalized obs if possible.
                # For now, just logging error if strictly needed, but let's assume body lookup works.
                print(f"Error calculating Euclidean distance: {e}")
                raise e
            
            # Check for done
            if done:
                break
        
        episode_returns.append(curr_return)
        episode_violations.append(1 if violation_occurred else 0)
        
        # Last 10 distances
        if ep_distances:
            last_10 = ep_distances[-10:]
            last_10_distances.append(np.mean(last_10))
        else:
            last_10_distances.append(0.0)

    env.close()

    # Calculate statistics
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    num_violations = np.sum(episode_violations)
    violation_rate = num_violations / num_episodes
    mean_last_10_dist = np.mean(last_10_distances)
    std_last_10_dist = np.std(last_10_distances)

    print(f"Results for {name}:")
    print(f"  Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    print(f"  Number of Episodes with Violation: {num_violations}/{num_episodes}")
    print(f"  Violation Rate (Episodes): {violation_rate:.2%}")
    print(f"  Mean Euclidean Distance (Last 10 steps): {mean_last_10_dist:.4f} +/- {std_last_10_dist:.4f}")
    print("-" * 50)

    # Save detailed results to CSV
    import pandas as pd
    df = pd.DataFrame({
        "episode": range(num_episodes),
        "return": episode_returns,
        "violation": episode_violations,
        "last_10_dist": last_10_distances
    })
    csv_path = f"eval_results_{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to {csv_path}")

if __name__ == "__main__":
    # You can verify/update the paths in the POLICIES dictionary above
    for name, path in POLICIES.items():
        if "path/to" in path:
            print(f"Skipping {name} (path not set)")
            continue
            
        evaluate_policy(name, path)

