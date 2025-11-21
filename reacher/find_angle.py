"""
Script to visualize OpenAI Gym Reacher environment with custom initial angles.

This script allows you to:
1. Define initial joint angles for the Reacher arm
2. Display the animation/picture corresponding to those initial angles
"""

import os
# Set environment variables for headless rendering BEFORE importing mujoco/gymnasium
# Try glfw or egl (they work better than osmesa in many systems)
if 'MUJOCO_GL' not in os.environ:
    # Default to glfw (better compatibility), fall back to egl if needed
    os.environ['MUJOCO_GL'] = 'glfw'

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import math

try:
    import mujoco
    MUJOCO_AVAILABLE = True
    print("MuJoCo imported successfully")
except Exception as e:
    MUJOCO_AVAILABLE = False
    print(f"Warning: MuJoCo import failed: {e}")
    


def _plot_reacher_2d(joint_angles, model, data, save_image=False, image_path='reacher_initial_state.png'):
    """
    Create a 2D matplotlib visualization of the reacher arm based on joint angles.
    This is a fallback when OpenGL rendering is not available.
    """
    # Reacher arm parameters (approximate)
    # Link 1: from base to elbow (approximately 0.1 m)
    # Link 2: from elbow to fingertip (approximately 0.1 m)
    link1_length = 0.1
    link2_length = 0.1
    
    # Base position (reacher origin)
    base_x, base_y = 0.0, 0.0
    
    # Calculate elbow position
    shoulder_angle = joint_angles[0]
    elbow_x = base_x + link1_length * math.cos(shoulder_angle)
    elbow_y = base_y + link1_length * math.sin(shoulder_angle)
    
    # Calculate fingertip position (relative to elbow, then add elbow position)
    elbow_angle = joint_angles[1]
    # Total angle for second link (relative to horizontal)
    total_angle = shoulder_angle + elbow_angle
    fingertip_x = elbow_x + link2_length * math.cos(total_angle)
    fingertip_y = elbow_y + link2_length * math.sin(total_angle)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot links
    ax.plot([base_x, elbow_x], [base_y, elbow_y], 'b-', linewidth=4, label='Link 1')
    ax.plot([elbow_x, fingertip_x], [elbow_y, fingertip_y], 'r-', linewidth=4, label='Link 2')
    
    # Plot joints
    ax.plot(base_x, base_y, 'ko', markersize=12, label='Base')
    ax.plot(elbow_x, elbow_y, 'go', markersize=10, label='Elbow')
    ax.plot(fingertip_x, fingertip_y, 'ro', markersize=10, label='Fingertip')
    
    # Plot target (if available from data)
    try:
        # Try to get target position from observation space
        target_pos = data.site_xpos[0] if model.nsite > 0 else None
        if target_pos is not None:
            ax.plot(target_pos[0], target_pos[1], 'y*', markersize=15, label='Target')
    except:
        pass
    
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Reacher Arm Visualization\nShoulder: {joint_angles[0]:.3f} rad ({math.degrees(joint_angles[0]):.1f}°), '
                 f'Elbow: {joint_angles[1]:.3f} rad ({math.degrees(joint_angles[1]):.1f}°)')
    ax.legend()
    
    if save_image:
        plt.tight_layout()
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        print(f"2D visualization saved to {image_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        
    return fig, ax


def visualize_reacher_with_angles(joint_angles, render_mode='human', save_image=False, image_path='reacher_initial_state.png'):
    """
    Visualize the Reacher environment with specified initial joint angles.
    
    Args:
        joint_angles: Array-like with 2 elements [shoulder_angle, elbow_angle] in radians
        render_mode: 'human' for interactive window, 'rgb_array' for frame capture, None for no rendering
        save_image: If True, save a static image of the initial state
        image_path: Path to save the image if save_image is True
    """
    # Create the Reacher environment
    if render_mode == 'rgb_array' or save_image:
        env = gym.make('Reacher-v5', render_mode='rgb_array')
    else:
        env = gym.make('Reacher-v5', render_mode=render_mode)
    
    # Convert to numpy array and ensure we have exactly 2 angles
    joint_angles = np.array(joint_angles, dtype=np.float32)
    if joint_angles.shape != (2,):
        raise ValueError(f"joint_angles must have shape (2,), got {joint_angles.shape}")
    
    # Reset the environment
    observation, info = env.reset()
    
    # Access the MuJoCo model and data
    # In gymnasium MuJoCo environments, use model and data instead of sim
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Set the joint positions (first 2 elements are the arm joints)
    data.qpos[:2] = joint_angles
    
    # Set the joint velocities to zero
    data.qvel[:2] = 0
    
    # Forward the simulation to apply changes
    if MUJOCO_AVAILABLE:
        mujoco.mj_forward(model, data)
    else:
        # If mujoco is not available, we can't forward, but angles are still set
        pass
    
    # Get the end effector position after setting angles
    # The reacher has two links, we can compute end effector position from angles
    try:
        if model.ngeom > 0:
            end_effector_pos = data.geom_xpos[model.ngeom - 1]  # Usually the fingertip is the last geom
            print(f"Joint angles set: shoulder={joint_angles[0]:.3f} rad, elbow={joint_angles[1]:.3f} rad")
            print(f"End effector position: x={end_effector_pos[0]:.3f}, y={end_effector_pos[1]:.3f}, z={end_effector_pos[2]:.3f}")
        else:
            print(f"Joint angles set: shoulder={joint_angles[0]:.3f} rad, elbow={joint_angles[1]:.3f} rad")
    except:
        print(f"Joint angles set: shoulder={joint_angles[0]:.3f} rad, elbow={joint_angles[1]:.3f} rad")
    
    if render_mode == 'human':
        # Render and keep window open
        env.render()
        print(f"Displaying Reacher with joint angles: shoulder={joint_angles[0]:.3f}, elbow={joint_angles[1]:.3f}")
        print("Close the render window when done viewing.")
        time.sleep(0.1)  # Small delay to ensure rendering
    elif render_mode == 'rgb_array' or save_image:
        # Try to get the rendered frame
        try:
            frame = env.render()
            if frame is not None:
                if save_image:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(frame)
                    plt.axis('off')
                    plt.title(f'Reacher Initial State\nShoulder: {joint_angles[0]:.3f} rad, Elbow: {joint_angles[1]:.3f} rad')
                    plt.tight_layout()
                    plt.savefig(image_path, dpi=150, bbox_inches='tight')
                    print(f"Image saved to {image_path}")
                    plt.close()
                else:
                    # Display using matplotlib
                    plt.figure(figsize=(10, 8))
                    plt.imshow(frame)
                    plt.axis('off')
                    plt.title(f'Reacher Initial State\nShoulder: {joint_angles[0]:.3f} rad, Elbow: {joint_angles[1]:.3f} rad')
                    plt.tight_layout()
                    plt.show()
            else:
                print(f"Warning: Could not render frame. Joint angles set: shoulder={joint_angles[0]:.3f}, elbow={joint_angles[1]:.3f}")
        except Exception as e:
            print(f"Rendering failed (likely due to headless environment): {e}")
            print(f"Joint angles successfully set: shoulder={joint_angles[0]:.3f} rad, elbow={joint_angles[1]:.3f} rad")
            print(f"Current joint positions: {data.qpos[:2]}")
            print("Falling back to matplotlib visualization...")
            # Create a simple 2D visualization using matplotlib
            _plot_reacher_2d(joint_angles, model, data, save_image, image_path)
    
    # Close the environment
    env.close()


def main():
    """
    Main function demonstrating how to use the visualization script.
    """
    # Define initial joint angles (in radians)
    # [shoulder_angle, elbow_angle]
    # Example angles:
    initial_angles = np.array([0.5, -0.5])  # 45° shoulder, -45° elbow
    
    # You can modify these angles as needed:
    # initial_angles = np.array([0.0, 0.0])      # Both joints at 0
    # initial_angles = np.array([np.pi/2, 0.0])  # Shoulder at 90°, elbow at 0°
    # initial_angles = np.array([-0.3, 0.7])     # Custom angles
    
    # Option 1: Display interactive animation window (try this first)
    print("Attempting OpenGL rendering with human mode...")
    try:
        visualize_reacher_with_angles(
            joint_angles=initial_angles,
            render_mode='human'
        )
    except Exception as e:
        print(f"Human mode failed: {e}")
        print("Falling back to rgb_array mode...")
        
        # Option 2: Display static image using matplotlib
        visualize_reacher_with_angles(
            joint_angles=initial_angles,
            render_mode='rgb_array',
            save_image=False
        )
    
    # Option 3: Save static image to file
    # visualize_reacher_with_angles(
    #     joint_angles=initial_angles,
    #     render_mode='rgb_array',
    #     save_image=True,
    #     image_path='reacher_initial_state.png'
    # )


if __name__ == '__main__':
    main()

