import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_flame_contour_plot():
    # Read the CSV file
    df = pd.DataFrame(pd.read_csv('flame_performance_samples.csv'))
    
    # Create grid for contour plot
    nozc_unique = sorted(df['nozc'].unique())
    nozw_unique = sorted(df['nozw'].unique())
    
    # Create meshgrid
    NOZC, NOZW = np.meshgrid(nozc_unique, nozw_unique)
    
    # Reshape the performance values to match the grid
    Z = df.pivot_table(index='nozw', columns='nozc', values='performance').values
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create contour plot
    levels = 50
    contours = plt.contour(NOZC, NOZW, Z, levels=15, colors='black', linewidths=0.5)
    contourf = plt.contourf(NOZC, NOZW, Z, levels=levels, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(contourf)
    cbar.set_label('Flame Performance', rotation=270, labelpad=15)
    
    # Add labels and title
    plt.xlabel('nozc (Nozzle Center Position)')
    plt.ylabel('nozw (Nozzle Width)')
    plt.title('Contour Plot of Flame Performance')
    
    # Add contour labels
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('flame_performance_contour.png', dpi=300, bbox_inches='tight')
    print("Contour plot saved as 'flame_performance_contour.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    create_flame_contour_plot()