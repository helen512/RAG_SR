import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
file_name = "energy_log_Custom_0.2.csv"
energy_df = pd.read_csv(os.path.join(HERE, file_name))

def process_data(data, name):
    print(f"\nProcessing {name}...")
    
    # Histogram data
    counts, bin_edges = np.histogram(data, bins=200)
    
    # Create DataFrame for export
    hist_df = pd.DataFrame({
        'bin_start': bin_edges[:-1],
        'bin_end': bin_edges[1:],
        'count': counts
    })
    
    csv_name = os.path.join(HERE, f"{name.lower().replace(' ', '_')}_histogram_data.csv")
    hist_df.to_csv(csv_name, index=False)
    print(f"Exported histogram data to {csv_name}")
    
    # Calculate percentages
    total_count = len(data)
    
    count_10 = ((data > -10) & (data < 10)).sum()
    pct_10 = (count_10 / total_count) * 100
    print(f"Percentage within -10 < data < 10: {pct_10:.2f}%")
    
    count_5 = ((data > -5) & (data < 5)).sum()
    pct_5 = (count_5 / total_count) * 100
    print(f"Percentage within -5 < data < 5: {pct_5:.2f}%")

    return counts, bin_edges

# Process Shaping Reward
y = energy_df['shaping_reward']
process_data(y, "Shaping Reward")

plt.hist(y, bins=200, density=True)
plt.xlabel('Shaping Reward')
plt.ylabel('Density')
plt.title('Shaping Reward Distribution')
plt.savefig(os.path.join(HERE, 'shaping_reward_distribution_0.2.png'))
plt.show(block=True)

# Process Total Energy
total_energy = energy_df['total_energy']
process_data(total_energy, "Total Energy")

plt.hist(total_energy, bins=200, density=True)
plt.xlabel('Total Energy')
plt.ylabel('Density')
plt.title('Total Energy Distribution')
plt.savefig(os.path.join(HERE, 'total_energy_distribution_0.2.png'))
plt.show(block=True)
