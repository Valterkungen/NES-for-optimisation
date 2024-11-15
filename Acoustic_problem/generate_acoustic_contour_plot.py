import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import find_peaks

def find_local_minima(Z):
    # Convert 2D array to 1D peaks
    minima = []
    rows, cols = Z.shape
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # Check if point is smaller than all its neighbors
            current = Z[i, j]
            neighbors = [
                Z[i-1, j], Z[i+1, j],  # Above and below
                Z[i, j-1], Z[i, j+1],  # Left and right
                Z[i-1, j-1], Z[i-1, j+1],  # Diagonal
                Z[i+1, j-1], Z[i+1, j+1]
            ]
            if current < min(neighbors):
                minima.append((i, j))
    
    return minima

def create_acoustic_contour_plot():
    # Read the CSV file
    df = pd.DataFrame(pd.read_csv('acoustic_performance_samples.csv'))
    
    # Filter the data for the specified range
    df = df[
        (df['c_amp'] >= -0.5) & (df['c_amp'] <= 0.5) &
        (df['c_pse'] >= 0.0) & (df['c_pse'] <= 2.0)
    ]
    
    # Create grid for contour plot
    c_pse_unique = sorted(df['c_pse'].unique())
    c_amp_unique = sorted(df['c_amp'].unique())
    
    # Create meshgrid
    PSE, AMP = np.meshgrid(c_pse_unique, c_amp_unique)
    
    # Reshape the performance values to match the grid
    Z = df.pivot_table(index='c_amp', columns='c_pse', values='performance').values
    
    # Find local minima
    local_minima_indices = find_local_minima(Z)
    
    # Convert indices to coordinates
    local_minima = []
    for i, j in local_minima_indices:
        pse = c_pse_unique[j]
        amp = c_amp_unique[i]
        perf = Z[i, j]
        local_minima.append((pse, amp, perf))
    
    # Sort minima by performance value
    local_minima.sort(key=lambda x: x[2])
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create levels
    min_val = np.min(Z)
    max_val = np.max(Z)
    n_levels = 50
    levels = np.linspace(min_val, max_val, n_levels)
    levels = np.unique(levels)
    
    # Create custom colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)-1))
    custom_colormap = ListedColormap(colors)
    
    # Create contour plot
    contours = plt.contour(PSE, AMP, Z, levels=15, colors='black', linewidths=0.5)
    contourf = plt.contourf(PSE, AMP, Z, levels=levels, cmap=custom_colormap)
    
    # Add contour labels
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.4f')
    
    # Add colorbar
    cbar = plt.colorbar(contourf)
    cbar.set_label('RMS Pressure', rotation=270, labelpad=15)
    
    # Plot all local minima
    markers = ['*', 'o', 's', 'd']  # Different markers for different minima
    colors = ['red', 'orange', 'yellow', 'green']  # Different colors for different minima
    
    for i, (pse, amp, perf) in enumerate(local_minima[:4]):  # Plot top 4 minima
        plt.plot(pse, amp, color=colors[i], marker=markers[i], 
                markersize=15, label=f'Minimum {i+1}: {perf:.6f}')
    
    plt.xlabel('Phase (c_pse)')
    plt.ylabel('Amplitude (c_amp)')
    plt.title('Acoustic Performance Contour Plot\nPhase (0 to 2), Amplitude (-0.5 to 0.5)')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend( loc='upper right')
    
    # Add text box with minima details
    stats_text = "Local Minima:\n"
    for i, (pse, amp, perf) in enumerate(local_minima[:4]):
        stats_text += f"{i+1}: ({pse:.3f}, {amp:.3f}) = {perf:.6f}\n"
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set axis limits
    plt.xlim(0, 2.0)
    plt.ylim(-0.5, 0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('acoustic_performance_contour_all_minima.png', dpi=300, bbox_inches='tight')
    print("Contour plot with all minima saved as 'acoustic_performance_contour_all_minima.png'")
    
    # Show the plot
    plt.show()
    
    # Print minima coordinates
    print("\nLocal Minima Coordinates:")
    for i, (pse, amp, perf) in enumerate(local_minima[:4]):
        print(f"Minimum {i+1}: Phase = {pse:.3f}, Amplitude = {amp:.3f}, Performance = {perf:.6f}")

if __name__ == "__main__":
    create_acoustic_contour_plot()