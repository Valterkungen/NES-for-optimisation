import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from flameLib import getPsi, getUV, getF, getPerf
from compute_flame_performance import compute_flame_performance

# Wrapper for the flame performance function
def flame_objective(params):
    nozc, nozw = params
    # Add constraints through penalty
    if nozw <= 0:  # Ensure positive width
        return -1e6
    try:
        perf = compute_flame_performance(float(nozc), float(nozw), refine=1, plot=False)
        return perf
    except:
        return -1e6  # Return large negative value if computation fails

def save_generation_data(generation_number, samples, fitness_values, theta, output_dir='nes_generations'):
    """Save data for a single generation to CSV"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame with generation data
    gen_data = pd.DataFrame({
        'nozc': samples[:, 0],
        'nozw': samples[:, 1],
        'fitness': fitness_values,
        'is_mean': False
    })
    
    # Add the mean (theta) point
    mean_point = pd.DataFrame({
        'nozc': [theta[0]],
        'nozw': [theta[1]],
        'fitness': [flame_objective(theta)],
        'is_mean': [True]
    })
    
    # Combine population and mean
    gen_data = pd.concat([gen_data, mean_point], ignore_index=True)
    
    # Save to CSV
    filename = f'{output_dir}/up_sigma_generation_{generation_number:03d}.csv'
    gen_data.to_csv(filename, index=False)
    return filename

class GradientDescent:
    def __init__(self, learning_rate=0.01, rng_key=None):
        self.learning_rate = learning_rate
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        self.rng_key = rng_key
        
        # Initialize starting point
        self.theta = jnp.array([0.0, 0.5])  # [nozc, nozw]
        
        # We'll use finite differences for gradient since we can't use JAX's autodiff
        self.eps = 1e-4
        
    def compute_gradient(self, theta):
        f0 = flame_objective(theta)
        grad = jnp.zeros(2)
        
        for i in range(2):
            theta_plus = theta.at[i].add(self.eps)
            f_plus = flame_objective(theta_plus)
            grad = grad.at[i].set((f_plus - f0) / self.eps)
            
        return grad
    
    def optimize(self, max_iterations, error_tol=1e-4):
        best_theta = self.theta
        best_value = flame_objective(self.theta)
        
        values = []
        thetas = []
        
        for iteration in range(max_iterations):
            # Compute gradient
            gradient = self.compute_gradient(self.theta)
            
            # Update parameters
            self.theta += self.learning_rate * gradient
            
            # Evaluate and track
            current_value = flame_objective(self.theta)
            values.append(current_value)
            thetas.append(self.theta.copy())
            
            # Update best
            if current_value > best_value:
                best_value = current_value
                best_theta = self.theta.copy()
            
            if iteration % 10 == 0:
                print(f"GD Iteration {iteration}: Value = {current_value:.4f}, Parameters = {self.theta}")
        
        return {
            'solution': best_theta,
            'values': values,
            'parameters': thetas,
            'iterations': iteration + 1
        }

class NES:
    def __init__(self, population_size=100, learning_rate=0.01, sigma=1, rng_key=None):
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.sigma_max, self.sigma_min = jnp.array(sigma), jnp.array(1e-4)
        self.sigma = self.sigma_max
        
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        self.rng_key = rng_key
        
        # Initialize starting point
        self.theta = jnp.array([0.0, 0.5])  # [nozc, nozw]
    
    def optimize(self, max_generations, error_tol=1e-4):
        # Track evolution path
        evolution_path = jnp.zeros(2)
        c = 0.1  # Learning rate for path update
        d = 1.0  # Damping for sigma update

        best_solution = self.theta
        best_fitness = flame_objective(self.theta)
        
        values = []
        parameters = []
        
        for generation in range(max_generations):
            start_time = time.time()
            # Sample solutions
            self.rng_key, subkey = random.split(self.rng_key)
            samples = self.theta + self.sigma * random.normal(subkey, (self.population_size, 2))
            
            # Evaluate fitness
            fitness_values = jnp.array([flame_objective(sample) for sample in samples])
            fitness_values_normalized = (fitness_values - jnp.mean(fitness_values)) / (jnp.std(fitness_values) + 1e-8)
            # Save generation data
            save_generation_data(generation, samples, fitness_values, self.theta)
            # Calculate gradient and update
            log_derivatives = (samples - self.theta) / (self.sigma**2)
            grad_estimate = jnp.sum(fitness_values_normalized[:, None] * log_derivatives, axis=0)
            fisher_estimate = jnp.dot(log_derivatives.T, log_derivatives) / self.population_size
            natural_gradient = jnp.linalg.solve(fisher_estimate + 1e-8 * jnp.eye(2), grad_estimate)
            
            # Update evolution path
            evolution_path = (1 - c) * evolution_path + jnp.sqrt(c * (2 - c)) * natural_gradient
            # Update parameters
            self.theta += self.learning_rate * natural_gradient
            # Update sigma based on path length
            expected_path_length = jnp.sqrt(2) * jnp.sqrt(jnp.prod(jnp.array(self.theta.shape)))
            sigma_mult = jnp.exp((jnp.linalg.norm(evolution_path) - expected_path_length) / (d * expected_path_length))
            self.sigma = jnp.clip(self.sigma * sigma_mult, self.sigma_min, self.sigma_max)
   
            
            # Evaluate and track
            current_fitness = flame_objective(self.theta)
            values.append(current_fitness)
            parameters.append(self.theta.copy())
            
            # Update best
            if current_fitness > best_fitness:
                best_solution = self.theta.copy()
                best_fitness = current_fitness
            
            if generation % 10 == 0:
                print(f"NES Generation {generation}: Value = {current_fitness:.4f}, Parameters = {self.theta}")
            print(f"Generation time: {time.time() - start_time:.2f}s")
        return {
            'solution': best_solution,
            'values': values,
            'parameters': parameters,
            'iterations': generation + 1
        }

def run_comparison():
    results_data = []
    
    # Run Gradient Descent
    print("\nRunning Gradient Descent...")
    gd = GradientDescent(learning_rate=0.01)
    start_time = time.time()
    gd_results = gd.optimize(max_iterations=1)
    gd_time = time.time() - start_time
    
    results_data.append({
        'method': 'GD',
        'best_params': gd_results['solution'],
        'best_value': flame_objective(gd_results['solution']),
        'values': gd_results['values'],
        'time': gd_time
    })
    
    # Run NES with different population sizes
    for pop_size in [50]:
        print(f"\nRunning NES with population {pop_size}...")
        nes = NES(population_size=pop_size, learning_rate=0.01, sigma=0.1)
        start_time = time.time()
        nes_results = nes.optimize(max_generations=100)
        nes_time = time.time() - start_time
        
        results_data.append({
            'method': f'NES (pop={pop_size})',
            'best_params': nes_results['solution'],
            'best_value': flame_objective(nes_results['solution']),
            'values': nes_results['values'],
            'time': nes_time
        })
    
    return results_data

def plot_comparison(results_data):
    plt.figure(figsize=(15, 10))
    
    # Plot convergence
    plt.subplot(2, 1, 1)
    for result in results_data:
        plt.plot(result['values'], label=f"{result['method']}")
    plt.xlabel('Iterations')
    plt.ylabel('Flame Performance')
    plt.title('Optimization Convergence')
    plt.grid(True)
    plt.legend()
    
    # Plot computation time
    plt.subplot(2, 1, 2)
    methods = [r['method'] for r in results_data]
    times = [r['time'] for r in results_data]
    plt.bar(methods, times)
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'flame_optimization_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
    plt.close()

if __name__ == "__main__":
    # Run comparison
    print("Starting optimization comparison...")
    results = run_comparison()
    
    # Print results
    print("\nOptimization Results:")
    for result in results:
        print(f"\n{result['method']}:")
        print(f"Best parameters: nozc = {result['best_params'][0]:.4f}, nozw = {result['best_params'][1]:.4f}")
        print(f"Best performance: {result['best_value']:.4f}")
        print(f"Computation time: {result['time']:.2f} seconds")
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    print("Done! Check the generated files for results.")