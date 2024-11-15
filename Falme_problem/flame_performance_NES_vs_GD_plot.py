import matplotlib.pyplot as plt
import pandas as pd

# Load the data from CSV
df = pd.read_csv("output_acoustic.csv")

print(df)
# Separate GD and NES data
gd_data = df[df["Type"] == "GD"]
nes_data = df[df["Type"] == "NES"]

def plot_flame_performance(data_gd, data_nes):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define colors and line styles for the plot
    colors = {'GD': 'blue', 'NES': 'green'}
    styles = {'GD': '-', 'NES': '--'}

    # Plot objective value convergence
    axes[0, 0].plot(data_gd["Iteration"], data_gd["Value"],
                    color=colors['GD'], linestyle=styles['GD'], label='GD')
    axes[0, 0].plot(data_nes["Iteration"], data_nes["Value"],
                    color=colors['NES'], linestyle=styles['NES'], label='NES (pop=50)')
    axes[0, 0].set_ylabel('Objective Value')
    axes[0, 0].set_xlabel('Iterations')
    axes[0, 0].set_title('Convergence - Objective Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot Parameter 1 evolution
    axes[0, 1].plot(data_gd["Iteration"], data_gd["Parameter1"],
                    color=colors['GD'], linestyle=styles['GD'], label='GD')
    axes[0, 1].plot(data_nes["Iteration"], data_nes["Parameter1"],
                    color=colors['NES'], linestyle=styles['NES'], label='NES (pop=50)')
    axes[0, 1].set_ylabel('Parameter 1')
    axes[0, 1].set_xlabel('Iterations')
    axes[0, 1].set_title('Parameter 1 Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Plot Parameter 2 evolution
    axes[1, 0].plot(data_gd["Iteration"], data_gd["Parameter2"],
                    color=colors['GD'], linestyle=styles['GD'], label='GD')
    axes[1, 0].plot(data_nes["Iteration"], data_nes["Parameter2"],
                    color=colors['NES'], linestyle=styles['NES'], label='NES (pop=50)')
    axes[1, 0].set_ylabel('Parameter 2')
    axes[1, 0].set_xlabel('Iterations')
    axes[1, 0].set_title('Parameter 2 Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    # Configure empty plot for better layout (optional)
    axes[1, 1].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('ice_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as ice_performance_comparison.png")

# Call the function with the GD and NES data
plot_flame_performance(gd_data, nes_data)
