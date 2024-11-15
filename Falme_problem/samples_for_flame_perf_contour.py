import numpy as np
import pandas as pd
import time
from compute_flame_performance import compute_flame_performance

def sample_flame_space():
    # Define the sampling grid
    # Using a coarse grid initially - adjust step sizes as needed
    nozc_values = np.linspace(-1.5, 0, 21)  # 21 points for 0.25 step size
    nozw_values = np.linspace(0, 4, 21)    # 21 points for 0.5 step size
    
    # Initialize results list
    results = []
    
    # Calculate total number of points
    total_points = len(nozc_values) * len(nozw_values)
    current_point = 0
    
    # Start timing
    start_time = time.time()
    
    print(f"Sampling {total_points} points...")
    
    for nozc in nozc_values:
        for nozw in nozw_values:
            current_point += 1
            print(f"Processing point {current_point}/{total_points}: nozc={nozc:.2f}, nozw={nozw:.2f}")
            
            try:
                # Use a lower refinement for faster sampling
                performance = compute_flame_performance(nozc, nozw, refine=1, plot=False)
                results.append({
                    'nozc': nozc,
                    'nozw': nozw,
                    'performance': performance
                })
            except Exception as e:
                print(f"Error at point nozc={nozc}, nozw={nozw}: {str(e)}")
                results.append({
                    'nozc': nozc,
                    'nozw': nozw,
                    'performance': np.nan
                })
            
            # Print estimated time remaining
            if current_point % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_point = elapsed_time / current_point
                remaining_points = total_points - current_point
                est_time_remaining = remaining_points * avg_time_per_point
                print(f"Estimated time remaining: {est_time_remaining/60:.1f} minutes")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_filename = 'flame_performance_samples.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time/60:.1f} minutes")
    
    return df

# Run the sampling if this script is run directly
if __name__ == "__main__":
    results_df = sample_flame_space()