import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap, grad
import time
import json
import os

# Define the Sphere function (f1)
def fitness_function(samples, use_sphere=True):
    def sphere(x):
        return -jnp.sum(x**2)  # Negative because we're maximizing
    
    if use_sphere:
        return sphere(samples)
    
class NES:
    def __init__(self, dim=10, population_size=100, learning_rate=0.01, sigma=0.1, rng_key=None):
        self.dim = dim
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.sigma_max, self.sigma_min = jnp.array(sigma), jnp.array(1e-4)
        self.sigma = self.sigma_max
        if rng_key is None:
            rng_key = random.PRNGKey(np.random.randint(0, 1000))
        self.rng_key, subkey = random.split(rng_key)
        self.theta = random.normal(subkey, (dim,))

    def optimize(self, max_generations, track_convergence=True, track_parameters=False, error_tol=1e-6):
        best_solution = self.theta
        best_fitness = fitness_function(best_solution)
        
        convergence = [] if track_convergence else None
        parameter_history = [] if track_parameters else None
        errors = []
        
        for generation in range(max_generations):
            # Sample solutions
            self.rng_key, subkey = random.split(self.rng_key)
            samples = self.theta + self.sigma * random.normal(subkey, (self.population_size, self.dim))
            
            # Evaluate fitness
            fitness_values = vmap(fitness_function)(samples)
            fitness_values_normalized = (fitness_values - jnp.mean(fitness_values)) / (jnp.std(fitness_values) + 1e-8)
            
            # Calculate gradient and update
            log_derivatives = (samples - self.theta) / (self.sigma**2)
            grad_estimate = jnp.sum(fitness_values_normalized[:, None] * log_derivatives, axis=0)
            fisher_estimate = jnp.dot(log_derivatives.T, log_derivatives) / self.population_size
            natural_gradient = jnp.linalg.solve(fisher_estimate + 1e-8 * jnp.eye(self.dim), grad_estimate)
            
            # Adapt learning rate
            self.adaptation_sampling(natural_gradient, fitness_values)
            
            # Update parameters
            previous_theta = self.theta
            self.theta += self.learning_rate * natural_gradient
            self.sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * jnp.exp(-0.001 * generation)
            
            # Track best solution
            current_fitness = fitness_function(self.theta)
            if current_fitness > best_fitness:
                best_solution = self.theta
                best_fitness = current_fitness
            
            # Track progress
            if track_convergence:
                convergence.append(float(current_fitness))
            if track_parameters:
                parameter_history.append(self.theta.copy())
            
            # Calculate error (distance from optimal solution at origin)
            error = float(jnp.linalg.norm(best_solution))
            errors.append(error)
            
            if error < error_tol:
                print(f"Converged at generation {generation} with error {error:.2e}")
                break
            
            if generation % 1000 == 0:
                print(f"Generation {generation}: Error = {error:.2e}")
        
        results = {
            'solution': best_solution,
            'fitness': float(best_fitness),
            'errors': errors,
            'generation': generation
        }
        if track_convergence:
            results['convergence'] = convergence
        if track_parameters:
            results['parameters'] = parameter_history
            
        return results

    def adaptation_sampling(self, update, fitness_values, c_prime=0.1, scaling_factor=1.1, max_lr=1.0):
        eta_prime = jnp.minimum(scaling_factor * self.learning_rate, max_lr)
        self.rng_key, subkey = random.split(self.rng_key)
        theta_prime = self.theta + eta_prime * update
        samples_prime = theta_prime + self.sigma * random.normal(subkey, (self.population_size, self.dim))
        fitness_values_prime = vmap(fitness_function)(samples_prime)
        
        if jnp.isnan(fitness_values_prime).any():
            self.learning_rate = self.learning_rate_initial
            return
            
        avg_fitness = jnp.mean(fitness_values)
        avg_fitness_prime = jnp.mean(fitness_values_prime)
        self.learning_rate = eta_prime if avg_fitness_prime > avg_fitness else (1 - c_prime) * self.learning_rate + c_prime * self.learning_rate_initial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime

# Your existing NES class and fitness_function remain the same

def run_experiments():
    # Experiment parameters
    dimensions = [2, 5, 10, 25]
    population_sizes = [50, 500]
    max_generations = 10000
    learning_rate = 0.01
    sigma = 0.1
    
    # Store results for each configuration
    results_data = []
    
    for pop_size in population_sizes:
        for dim in dimensions:
            print(f"\nRunning: Dimension {dim}, Population {pop_size}")
            
            # Initialize NES
            key = random.PRNGKey(42)  # Fixed seed for reproducibility
            nes = NES(dim=dim, population_size=pop_size, learning_rate=learning_rate, 
                     sigma=sigma, rng_key=key)
            
            # Run optimization
            start_time = time.time()
            results = nes.optimize(max_generations=max_generations, 
                                 track_convergence=True,
                                 error_tol=1e-6)
            
            # Store results
            run_data = {
                'dimension': dim,
                'population': pop_size,
                'errors': results['errors'],
                'objective_values': results['convergence'],
                'generations': len(results['errors']),
                'time': time.time() - start_time
            }
            results_data.append(run_data)
            
            # Save to CSV
            df = pd.DataFrame({
                'generation': range(len(results['errors'])),
                'relative_error': np.array(results['errors']) * 100,  # Convert to percentage
                'objective_value': np.abs(results['convergence']),
                'dimension': dim,
                'population': pop_size
            })
            
            # Save each run
            df.to_csv(f'nes_results_dim{dim}_pop{pop_size}.csv', index=False)
            
            print(f"Completed in {run_data['time']:.2f} seconds")
            print(f"Final error: {results['errors'][-1]:.2e}")
    
    return results_data

def plot_results(results_data):
    # Set up the figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Color mapping for dimensions
    colors = {2: 'black', 5: 'blue', 10: 'green', 25: 'cyan'}
    
    # Plot for each population size
    for pop_idx, pop_size in enumerate([50, 500]):
        pop_results = [r for r in results_data if r['population'] == pop_size]
        
        for result in pop_results:
            dim = result['dimension']
            generations = np.arange(1, len(result['errors']) + 1)
            
            # Plot relative error (top row)
            axes[0, pop_idx].plot(generations, 
                                np.array(result['errors']) * 100,  # Convert to percentage
                                color=colors[dim],
                                label=f'Dimension {dim}')
            
            # Plot objective value (bottom row)
            axes[1, pop_idx].plot(generations,
                                np.abs(result['objective_values']),
                                color=colors[dim])
    
    # Configure plots
    for i in range(2):
        for j in range(2):
            axes[i, j].set_xscale('log')
            axes[i, j].grid(True, which='both', linestyle='--', alpha=0.7)
            axes[i, j].set_xlabel('Iterations (log scale)')
    
    # Set specific settings for each row
    for j in range(2):
        # Relative error plots (top row)
        axes[0, j].set_ylabel('Relative error (%)')
        axes[0, j].set_title(f'Population = {[50, 500][j]}')
        
        # Objective value plots (bottom row)
        axes[1, j].set_yscale('log')
        axes[1, j].set_ylabel('Objective value (log scale)')
    
    # Add legend to the first plot only
    axes[0, 0].legend(title='Dimension')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'nes_results_{datetime.now().strftime("%Y%m%d_%H%M")}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Run experiments
    print("Starting experiments...")
    results = run_experiments()
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(results)
    print("Done! Check the generated files for results.")