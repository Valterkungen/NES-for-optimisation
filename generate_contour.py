import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def create_contour_plot():
    # Read the CSV file
    df = pd.DataFrame(pd.read_csv('Ass 2 fr/dev_samples.csv'))
    
    # Create grid for contour plot
    a_unique = sorted(df['a'].unique())
    b_unique = sorted(df['b'].unique())
    
    # Create meshgrid
    A, B = np.meshgrid(a_unique, b_unique)
    
    # Reshape the dev values to match the grid
    Z = df.pivot_table(index='b', columns='a', values='dev').values
    
    # Create the figure with a specific size
    plt.figure(figsize=(12, 8))
    
    # Create contour plot
    # Using more levels for smoother visualization
    levels = 50
    contours = plt.contour(A, B, Z, levels=levels, colors='black', linewidths=0.5)
    contourf = plt.contourf(A, B, Z, levels=levels, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(contourf)
    cbar.set_label('DEV Value', rotation=270, labelpad=15)
    
    # Add labels and title
    plt.xlabel('a (Ice Thickness Amplitude)')
    plt.ylabel('b (Displacement of Minimum)')
    plt.title('Contour Plot of DEV Function')
    
    # Add contour labels
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('dev_contour_plot.png', dpi=300, bbox_inches='tight')
    print("Contour plot saved as 'dev_contour_plot.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    create_contour_plot()