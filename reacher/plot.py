import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(log_folder, title='Reacher-v5 PPO Training Reward'):
    """
    Plot the results using pandas loading
    """
    try:
        # Find monitor file
        monitor_files = [f for f in os.listdir(log_folder) if f.endswith("monitor.csv")]
        if not monitor_files:
            print("No monitor file found.")
            return
            
        # Assuming single env or picking the first one
        # Note: If running multiple times, there might be multiple files. 
        # We usually pick the most recent one or all. 
        # Here we pick the most recent one based on modification time.
        latest_file = max([os.path.join(log_folder, f) for f in monitor_files], key=os.path.getmtime)
        print(f"Plotting results from: {latest_file}")
            
        results_df = pd.read_csv(latest_file, comment='#')
        
        # 'r' is modified reward
        # 'original_reward' is standard reward
        
        # Calculate rolling window mean for smoother plot
        window = 50
        if len(results_df) > window:
            rolling_mean_mod = results_df['r'].rolling(window=window).mean()
            rolling_mean_orig = results_df['original_reward'].rolling(window=window).mean()
            
        else:
            rolling_mean_mod = results_df['r']
            rolling_mean_orig = results_df['original_reward']
        

        plt.figure(figsize=(12, 8))
        
        # Plot Modified Reward
        plt.plot(results_df['l'].cumsum(), results_df['r'], alpha=0.3, label='Modified Reward (with penalty)', color='blue')
        plt.plot(results_df['l'].cumsum(), rolling_mean_mod, color='blue', linewidth=2, label=f'Modified Mean ({window} eps)')
        
        # Plot Standard Reward
        if 'original_reward' in results_df.columns:
            plt.plot(results_df['l'].cumsum(), results_df['original_reward'], alpha=0.3, label='Standard Reward', color='green')
            plt.plot(results_df['l'].cumsum(), rolling_mean_orig, color='green', linewidth=2, label=f'Standard Mean ({window} eps)')
        
        plt.ylim(-30, 0)
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(log_folder, "plot.png")
        plt.savefig(save_path)
        print(f"Reward plot saved to {save_path}")
        plt.show(block=True)
        plt.close()
        
        # Track violations
        total_episodes = len(results_df)
        if 'violation' in results_df.columns:
            # violation column might be boolean or string 'True'/'False'
            # Convert to boolean if necessary
            if results_df['violation'].dtype == object:
                 violation_count = (results_df['violation'] == 'True').sum()
            else:
                 violation_count = results_df['violation'].sum()
                 
            print(f"Total Episodes: {total_episodes}")
            print(f"Episodes Terminated due to Violation: {violation_count}")
            print(f"Violation Rate: {violation_count/total_episodes:.2%}")
        else:
            print("No violation tracking data found.")
        
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    log_folder = "test"
    plot_results(log_folder)