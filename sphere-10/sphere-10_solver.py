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
from jax import grad, vmap
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd

# Sphere function for both optimizers
def sphere_function(x):
    return jnp.sum(x**2)

# Gradient-based optimizer
class GradientDescent:
    def __init__(self, dim, learning_rate=0.01, rng_key=None):
        self.dim = dim
        self.learning_rate = learning_rate
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        self.rng_key = rng_key
        
        # Initialize starting point randomly
        self.theta = random.normal(self.rng_key, (dim,))
        
        # Get gradient function
        self.grad_fn = grad(sphere_function)
    
    def optimize(self, max_iterations, error_tol=1e-6):
        errors = []
        objective_values = []
        
        for iteration in range(max_iterations):
            # Calculate gradient
            gradient = self.grad_fn(self.theta)
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            # Calculate error and objective
            error = float(jnp.linalg.norm(self.theta))
            obj_value = float(sphere_function(self.theta))
            
            errors.append(error)
            objective_values.append(obj_value)
            
            if error < error_tol:
                print(f"GD Converged at iteration {iteration} with error {error:.2e}")
                break
                
            if iteration % 1000 == 0:
                print(f"GD Iteration {iteration}: Error = {error:.2e}")
        
        return {
            'solution': self.theta,
            'errors': errors,
            'objective_values': objective_values,
            'iterations': iteration + 1
        }

def run_comparison():
    # Parameters
    dimensions = [2, 5, 10, 25]
    max_iterations = 5000
    results_data = []
    
    # NES parameters
    nes_population_sizes = [50, 500]
    nes_learning_rate = 0.01
    nes_sigma = 0.1
    
    # GD parameters
    gd_learning_rate = 0.01
    
    # Run experiments
    for dim in dimensions:
        print(f"\nRunning comparison for dimension {dim}")
        
        # Gradient Descent
        print("Running Gradient Descent...")
        gd = GradientDescent(dim=dim, learning_rate=gd_learning_rate)
        start_time = time.time()
        gd_results = gd.optimize(max_iterations)
        gd_time = time.time() - start_time
        
        # Store GD results
        results_data.append({
            'method': 'GD',
            'dimension': dim,
            'population': 1,  # GD doesn't use population
            'errors': gd_results['errors'],
            'objective_values': gd_results['objective_values'],
            'iterations': gd_results['iterations'],
            'time': gd_time
        })
        
        # NES for different population sizes
        for pop_size in nes_population_sizes:
            print(f"Running NES with population {pop_size}...")
            nes = NES(dim=dim, population_size=pop_size, 
                     learning_rate=nes_learning_rate, sigma=nes_sigma)
            
            start_time = time.time()
            nes_results = nes.optimize(max_iterations)
            nes_time = time.time() - start_time
            
            results_data.append({
                'method': f'NES (pop={pop_size})',
                'dimension': dim,
                'population': pop_size,
                'errors': nes_results['errors'],
                'objective_values': np.abs(nes_results['convergence']),
                'iterations': len(nes_results['errors']),
                'time': nes_time
            })
    
    return results_data

def plot_comparison(results_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Color mapping for dimensions
    colors = {2: 'black', 5: 'blue', 10: 'green', 25: 'cyan'}
    # Line style mapping for methods
    styles = {'GD': '-', 'NES (pop=50)': '--', 'NES (pop=500)': ':'}
    
    # Plot for each dimension
    for dim in [2, 5, 10, 25]:
        dim_results = [r for r in results_data if r['dimension'] == dim]
        
        for result in dim_results:
            method = result['method']
            generations = np.arange(1, len(result['errors']) + 1)
            
            # Plot relative error (left)
            axes[0, 0].plot(generations, 
                          np.array(result['errors']) * 100,
                          color=colors[dim],
                          linestyle=styles[method],
                          label=f'Dim {dim} - {method}')
            
            # Plot objective value (right)
            axes[0, 1].plot(generations,
                          result['objective_values'],
                          color=colors[dim],
                          linestyle=styles[method])
            
            # Plot computation time (bottom)
            axes[1, 0].bar(f'{dim}D-{method}', 
                         result['time'],
                         label=method)
            
            # Plot iterations to convergence (bottom)
            axes[1, 1].bar(f'{dim}D-{method}',
                         result['iterations'])
    
    # Configure plots
    for ax in axes[0]:
        ax.set_xscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_xlabel('Iterations (log scale)')
    
    # Specific settings for each plot
    axes[0, 0].set_ylabel('Relative error (%)')
    axes[0, 0].set_title('Convergence - Relative Error')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel('Objective value (log scale)')
    axes[0, 1].set_title('Convergence - Objective Value')
    
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Computation Time')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    axes[1, 1].set_ylabel('Iterations')
    axes[1, 1].set_title('Iterations to Convergence')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'optimization_comparison_{datetime.now().strftime("%Y%m%d_%H%M")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Run comparison
    print("Starting comparison experiments...")
    results = run_comparison()
    
    # Save results to CSV
    df_results = pd.DataFrame([{
        'method': r['method'],
        'dimension': r['dimension'],
        'population': r['population'],
        'final_error': r['errors'][-1],
        'final_objective': r['objective_values'][-1],
        'iterations': r['iterations'],
        'time': r['time']
    } for r in results])
    
    df_results.to_csv(f'optimization_comparison_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', 
                      index=False)
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    print("Done! Check the generated files for results.")