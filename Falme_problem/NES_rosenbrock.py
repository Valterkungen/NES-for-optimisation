import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import seaborn as sns
import time
#import lognorm
from matplotlib.colors import LogNorm
def rosenbrock(x):
    """
    Rosenbrock function (banana function)
    Global minimum at (1,1)
    f(x,y) = (1-x)^2 + 100(y-x^2)^2
    """
    return (((1 - x[0])**2) + 100 * ((x[1] - x[0]**2)**2))

class NES:
    def __init__(self, population_size=50, learning_rate=0.01, sigma=0.1, seed=None):
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate
        self.sigma_max = sigma
        self.sigma_min = 1e-4
        self.sigma = sigma
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # Initialize RNG
        self.rng = np.random.default_rng(seed)
        
        # Initialize parameters
        self.theta = np.array([0.0, 0.0])  # Starting point
        
        # Initialize momentum and second moment estimates
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0
        
        # Track best solution
        self.best_solution = self.theta.copy()
        self.best_fitness = float('-inf')
        self.history = []

    def save_generation_data(self, generation, samples, fitness_values, output_dir='nes_generations'):
        """Save data for a single generation to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame with generation data
        gen_data = pd.DataFrame({
            'x': samples[:, 0],
            'y': samples[:, 1],
            'fitness': fitness_values,
            'is_mean': False
        })
        
        # Add the mean (theta) point
        mean_point = pd.DataFrame({
            'x': [self.theta[0]],
            'y': [self.theta[1]],
            'fitness': [rosenbrock(self.theta)],
            'is_mean': [True]
        })
        
        # Combine population and mean
        gen_data = pd.concat([gen_data, mean_point], ignore_index=True)
        
        # Save to CSV
        filename = f'{output_dir}/generation_{generation:03d}.csv'  # Fixed filename format
        gen_data.to_csv(filename, index=False)
        return filename
    
    def optimize(self, max_generations=100, tolerance=1e-6, patience=20):
        start_time = time.time()
        stagnation_counter = 0
        prev_best = float('-inf')
        
        print("\nStarting NES optimization on Rosenbrock function")
        print(f"Population size: {self.population_size}")
        print(f"Initial parameters: x={self.theta[0]:.6f}, y={self.theta[1]:.6f}")
        print("-" * 50)
        
        for generation in range(max_generations):
            gen_start_time = time.time()
            
            # Generate population
            epsilon = self.rng.normal(0, 1, (self.population_size, 2))
            solutions = self.theta + self.sigma * epsilon
            
            # Evaluate fitness
            fitness_values = np.array([rosenbrock(solution) for solution in solutions])
            ranks = np.argsort(np.argsort(fitness_values))
            utilities = np.maximum(0, np.log(self.population_size/2 + 1) - np.log(ranks + 1))
            utilities = (utilities - np.mean(utilities)) / (np.std(utilities) + self.epsilon)
            
            # Save generation data
            self.save_generation_data(generation, solutions, fitness_values)
            
            # Update best solution
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_solution = solutions[best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Compute gradients
            grad_theta = np.sum(utilities.reshape(-1, 1) * epsilon, axis=0) / self.sigma
            grad_sigma = np.sum(utilities * (np.sum(epsilon**2, axis=1) - 2)) / self.sigma
            
            # Update parameters with Adam
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_theta
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_theta**2)
            
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            
            self.theta += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Update sigma with decay
            self.sigma *= np.exp(self.learning_rate * 0.1 * grad_sigma)
            self.sigma = np.clip(self.sigma, 
                               self.sigma_min,
                               self.sigma_max * np.exp(-generation/max_generations))
            
            # Store generation statistics
            gen_time = time.time() - gen_start_time
            gen_stats = {
                'generation': generation,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'learning_rate': self.learning_rate,
                'sigma': self.sigma,
                'theta': self.theta.copy(),
                'time': gen_time
            }
            self.history.append(gen_stats)
            
            # Print progress
            if (generation + 1) % 5 == 0:
                print(f"\nGeneration {generation + 1}/{max_generations}")
                print(f"Best fitness: {self.best_fitness:.10f}")
                print(f"At x={self.best_solution[0]:.6f}, y={self.best_solution[1]:.6f}")
                print(f"Current σ: {self.sigma:.6f}, η: {self.learning_rate:.6f}")
                print(f"Generation time: {gen_time:.2f}s")
            
            # Check convergence
            if abs(self.best_fitness - prev_best) < tolerance:
                if stagnation_counter >= patience:
                    print("\nConverged! No improvement in fitness.")
                    break
            
            prev_best = self.best_fitness
        
        self.total_time = time.time() - start_time
        return self.best_solution, self.best_fitness

def create_rosenbrock_contour_plot_with_generation(generation_number):
    """Create contour plot of Rosenbrock function with generation data overlay"""
    # Create grid for contour plot
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-1, 3, 1000)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Rosenbrock function values
    Z = ((1 - X)**2 + 100 * (Y - X**2)**2)  # Negated to match our maximization
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create logarithmically spaced levels
    # First, shift Z to be positive
    Z_shift = Z + 1e-6
    
    # Create logarithmic levels
    min_level = np.log10(Z_shift.min())
    max_level = np.log10(Z_shift.max())
    levels = 500
    
    # Create contour plot with logarithmic levels
    contours = plt.contour(X, Y, Z_shift, levels=15, colors='black', linewidths=0.5)
    contourf = plt.contourf(X, Y, Z_shift, levels=levels, cmap='viridis', norm=LogNorm(vmin=Z_shift.min(), vmax=Z_shift.max()))
    
    # Load and plot generation data
    gen_file = f'nes_generations/generation_{generation_number:03d}.csv'  # Fixed filename format
    try:
        gen_data = pd.read_csv(gen_file)
        
        # Plot population points
        population = gen_data[~gen_data['is_mean']]
        mean_point = gen_data[gen_data['is_mean']]
        
        # Plot population points with color based on fitness
        scatter = plt.scatter(population['x'], population['y'], 
                            c=population['fitness'], 
                            cmap='Reds',
                            s=50, alpha=0.6, 
                            edgecolors='white', linewidth=1)
        
        # Plot mean point as a star
        plt.scatter(mean_point['x'], mean_point['y'], 
                   color='yellow', marker='*', s=200, 
                   label='Population Mean', 
                   edgecolors='black', linewidth=1)
        
        # Add optimal point
        plt.scatter(1, 1, color='green', marker='x', s=100, 
                   label='Global Optimum', linewidth=2)
        
        plt.legend()
        
    except FileNotFoundError:
        print(f"Generation {generation_number} data not found")
    
    # Add colorbar with logarithmic scale
    cbar = plt.colorbar(contourf)
    cbar.set_label('Rosenbrock Function Value (log scale)', rotation=270, labelpad=15)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Rosenbrock Function Contour with Generation {generation_number}')
    
    # Add grid and adjust layout
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'rosenbrock_contour_gen_{generation_number:03d}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('nes_generations', exist_ok=True)
    
    # Run optimization
    nes = NES(population_size=50, learning_rate=0.01, sigma=0.1, seed=42)
    best_solution, best_fitness = nes.optimize(max_generations=50)
    
    # Create plots for all generations
    for gen in range(50):
        create_rosenbrock_contour_plot_with_generation(gen)
    
    print("\nOptimization complete!")
    print(f"Best solution found: x={best_solution[0]:.6f}, y={best_solution[1]:.6f}")
    print(f"Best fitness: {best_fitness:.6f}")
    print("\nGeneration plots have been saved!")