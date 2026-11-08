"""
Parse training log and plot training progress
"""
import re
import matplotlib.pyplot as plt
import numpy as np

# Parse the log file
log_file = 'training_full.log'
steps = []
returns = []
losses = []

with open(log_file, 'r') as f:
    content = f.read()
    
    # Find all log entries
    # Pattern to match step and ep_return
    step_pattern = r'\|\s+step\s+\|\s+([\d.e+]+)\s+\|'
    return_pattern = r'\|\s+ep_return\s+\|\s+([\d.e+]+)\s+\|'
    loss_pattern = r'\|\s+policy_loss\s+\|\s+([-\d.e+]+)\s+\|'
    
    # Split into blocks
    blocks = content.split('--------------------------------------')
    
    for block in blocks:
        step_match = re.search(step_pattern, block)
        return_match = re.search(return_pattern, block)
        loss_match = re.search(loss_pattern, block)
        
        if step_match and return_match:
            step_val = float(step_match.group(1))
            return_val = float(return_match.group(1))
            steps.append(step_val)
            returns.append(return_val)
            
            if loss_match:
                loss_val = float(loss_match.group(1))
                losses.append(loss_val)

print(f"Parsed {len(steps)} training steps")
print(f"Steps range: {min(steps)} to {max(steps)}")
print(f"Returns range: {min(returns)} to {max(returns)}")

# Create plots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Episode Return vs Training Steps
axes[0].plot(steps, returns, linewidth=2, color='blue')
axes[0].set_xlabel('Training Steps', fontsize=12)
axes[0].set_ylabel('Average Episode Return', fontsize=12)
axes[0].set_title('PPO Training on InvertedPendulum-v4: Episode Return vs Steps', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, max(returns) * 1.1])

# Add horizontal line at 1000 (max possible return)
axes[0].axhline(y=1000, color='red', linestyle='--', linewidth=2, label='Max Episode Length')
axes[0].legend(fontsize=10)

# Plot 2: Policy Loss vs Training Steps (if available)
if losses:
    axes[1].plot(steps[:len(losses)], losses, linewidth=2, color='orange')
    axes[1].set_xlabel('Training Steps', fontsize=12)
    axes[1].set_ylabel('Policy Loss', fontsize=12)
    axes[1].set_title('Policy Loss During Training', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'Policy loss data not available', 
                 ha='center', va='center', fontsize=14, transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig('ppo_training_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to 'ppo_training_plot.png'")

# Print summary statistics
print("\n" + "="*60)
print("Training Summary")
print("="*60)
print(f"Total training steps: {int(max(steps))}")
print(f"Initial average return: {returns[0]:.1f}")
print(f"Final average return: {returns[-1]:.1f}")
print(f"Improvement: {returns[-1] - returns[0]:.1f} (+{((returns[-1]/returns[0])-1)*100:.1f}%)")
print(f"Steps to reach 1000 return: {steps[next(i for i, r in enumerate(returns) if r >= 999)]:.0f}")
print("="*60)

plt.show()

