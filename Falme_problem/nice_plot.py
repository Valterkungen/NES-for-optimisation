import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_flame_contour_plot_with_generation(generation_number, contour_data_file='flame_performance_samples.csv'):
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Read the contour data
    df = pd.DataFrame(pd.read_csv(contour_data_file))
    
    # Create grid for contour plot
    nozc_unique = sorted(df['nozc'].unique())
    nozw_unique = sorted(df['nozw'].unique())
    NOZC, NOZW = np.meshgrid(nozc_unique, nozw_unique)
    Z = df.pivot_table(index='nozw', columns='nozc', values='performance').values
    
    # Create base contour plot
    levels = 50
    contours = plt.contour(NOZC, NOZW, Z, levels=15, colors='black', linewidths=0.5)
    contourf = plt.contourf(NOZC, NOZW, Z, levels=levels, cmap='viridis')
    
    # Load generation data
    gen_file = f'nes_generations/generation_{generation_number:03d}.csv'
    try:
        gen_data = pd.read_csv(gen_file)
        
        # Plot population points
        population = gen_data[~gen_data['is_mean']]
        mean_point = gen_data[gen_data['is_mean']]
        
        # Plot population points with color based on fitness
        scatter = plt.scatter(population['nozc'], population['nozw'], 
                            c=population['fitness'], 
                            cmap='Reds',
                            s=50, alpha=0.6, 
                            edgecolors='white', linewidth=1)
        
        # Plot mean point as a star
        plt.scatter(mean_point['nozc'], mean_point['nozw'], 
                   color='yellow', marker='*', s=200, 
                   label='Population Mean', 
                   edgecolors='black', linewidth=1)
        
        # Add a legend for the mean point
        plt.legend()
        
        # Add title with generation number
        plt.title(f'Flame Performance Contour with Generation {generation_number} Population')
    except FileNotFoundError:
        print(f"Generation {generation_number} data not found")
        plt.title('Flame Performance Contour')
    
    # Add colorbar for the contour plot
    cbar = plt.colorbar(contourf)
    cbar.set_label('Flame Performance', rotation=270, labelpad=15)
    
    # Add labels
    plt.xlabel('nozc (Nozzle Center Position)')
    plt.ylabel('nozw (Nozzle Width)')
    
    # Add contour labels
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'nes_generations/flame_performance_sigma_cma_contour_gen_{generation_number:03d}.png', 
                dpi=300, bbox_inches='tight')
    print(f"Contour plot with generation {generation_number} saved")
    
    plt.close()

def plot_all_generations(start_gen=0, end_gen=None):
    """Plot contours for a range of generations"""
    import glob
    import os
    
    # Get list of all generation files
    gen_files = glob.glob('nes_generations/generation_*.csv')
    if not gen_files:
        print("No generation files found!")
        return
    
    if end_gen is None:
        end_gen = len(gen_files) - 1
    
    for gen in range(start_gen, end_gen + 1):
        create_flame_contour_plot_with_generation(gen)

if __name__ == "__main__":
    # Plot a specific generation
    for i in range(90):
        create_flame_contour_plot_with_generation(i)
    
    # Or plot all generations
    # plot_all_generations()