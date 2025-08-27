import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap, grad
import time
import json
import os

# Define the Rosenbrock function
def fitness_function(samples, rosenbrock=True):
    def rosenbrock(x):
        return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    if rosenbrock:
        return -rosenbrock(samples)

class NES:
    def __init__(self, dim, population_size, learning_rate, sigma, rng_key):
        self.dim = dim
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.sigma_max, self.sigma_min = jnp.array(sigma), jnp.array(1e-4)  # Standard deviation for sampling  
        self.sigma = self.sigma_max
        self.rng_key, subkey = random.split(rng_key)  
        self.theta = random.normal(subkey, (dim,))  # Initial solution

    def optimize(self, max_generations, track_convergence=False, track_parameters=False, error_tol = 1e-4):
        best_solution = self.theta
        best_fitness = fitness_function(best_solution)

        # Initialize lists for convergence and parameter tracking if requested
        convergence = [] if track_convergence else None
        parameter_history = [] if track_parameters else None
        errors = []
        
        for generation in range(max_generations):

            # Update RNG key and draw samples
            self.rng_key, subkey = random.split(self.rng_key)
            samples = self.theta + self.sigma * random.normal(subkey, (self.population_size, self.dim))
            
            # Vectorized fitness evaluation
            fitness_values = vmap(fitness_function)(samples)
            fitness_values_normalized = (fitness_values - jnp.mean(fitness_values)) / (jnp.std(fitness_values) + 1e-8)

            # Calculate gradient estimate and Fisher estimate
            log_derivatives = (samples - self.theta) / (self.sigma**2)
            grad_estimate = jnp.sum(fitness_values_normalized[:, None] * log_derivatives, axis=0) 
            fisher_estimate = jnp.dot(log_derivatives.T, log_derivatives) / self.population_size
            natural_gradient = jnp.linalg.solve(fisher_estimate, grad_estimate)

            # Adapt learning rate if necessary
            self.adaptation_sampling(natural_gradient, fitness_values)
            previous_fitness = fitness_function(self.theta)
            self.theta += self.learning_rate * natural_gradient
            self.sigma = self.sigma_min - (self.sigma_min - self.sigma_max) *jnp.exp(-generation)
            
            # Update best solution
            current_fitness = fitness_function(self.theta)
            if current_fitness > best_fitness:
                best_solution = self.theta
                best_fitness = current_fitness
            
            # Track convergence and parameters if enableda
            if track_convergence:
                convergence.append(current_fitness)
            if track_parameters:
                parameter_history.append(self.theta.copy())
            
            # Save the error    
            error = jnp.linalg.norm(jnp.ones(self.dim) - best_solution) / jnp.linalg.norm(jnp.ones(self.dim))
            errors.append(error)

        # Return results based on tracking options
        results = {'solution': best_solution, 'errors': errors, 'generation': generation}
        if track_convergence:
            results['convergence'] = convergence
        if track_parameters:
            results['parameters'] = parameter_history

        return results

    def adaptation_sampling(self, update, fitness_values, c_prime=0.1, scaling_factor=1.1, max=1.0):
        eta_prime = jnp.minimum(scaling_factor * self.learning_rate, max)
        self.rng_key, subkey = random.split(self.rng_key)
        theta_prime = self.theta + eta_prime * update
        samples_prime = theta_prime + self.sigma * random.normal(subkey, (self.population_size, self.dim))
        fitness_values_prime = vmap(fitness_function)(samples_prime)

        # Revert learning rate if NaNs are found
        if jnp.isnan(fitness_values_prime).any():
            self.learning_rate = self.learning_rate_initial
            return

        # Compare average fitness values
        avg_fitness = jnp.mean(fitness_values)
        avg_fitness_prime = jnp.mean(fitness_values_prime)
        self.learning_rate = eta_prime if avg_fitness_prime > avg_fitness else (1 - c_prime) * self.learning_rate + c_prime * self.learning_rate_initial

if __name__ == "__main__":
    # Initialize RNG key for JAX random number generation
    key = np.random.randint(0, 1000)
    rng_key = random.PRNGKey(key)

    def produce_results():
        # Initialize parameters
        
        benchmark_configuration = {
            'dimensions': [2, 5, 10, 25],
            'population_size': [50, 500],
            'max_generations': 100001,
            'learning_rate': 0.01,
            'sigma': 0.1
        }

        # Initialize a list to store each row of results data
        results_data = []
        learning_rate = benchmark_configuration['learning_rate']
        sigma = benchmark_configuration['sigma']
        max_gen = benchmark_configuration['max_generations']
        # Loop through each dimension and population size
        for dim in benchmark_configuration['dimensions']:
            for pop in benchmark_configuration['population_size']:
                # Create NES instance with the current dimension and population size
                nes = NES(dim=dim, population_size=pop, learning_rate=learning_rate, sigma=sigma, rng_key=rng_key)
                
                # Capture start time for this iteration
                start_time = time.time()
                
                # Train and capture convergence data
                results = nes.optimize(max_generations=max_gen, track_convergence=True, track_parameters=True)
                optimal_theta_nes = np.array(results['solution']).astype(float) 
                convergence = np.array(results['convergence']).astype(float).tolist()  
                errors = np.array(results['errors']).astype(float).tolist()            
                parameter_history = np.array(results['parameters']).astype(float).tolist() 
                generation = int(results['generation']) 
                
                # Calculate the error for optimal theta
                optimal_theta_nes_error = float(np.sqrt(np.sum((np.ones(dim) - optimal_theta_nes)**2)))  
                time_taken = float(time.time() - start_time)
                
                
                results_data.append({
                    'Population': pop,
                    'Dimension': dim,
                    'Generation': generation,
                    'Error': optimal_theta_nes_error,
                    'Time (s)': time_taken,
                    'Errors': errors,
                    'Parameters History': parameter_history,
                    'Convergence': convergence
                })
                
                print(f"Time taken for dimension {dim} and population {pop} is {time_taken} seconds")

        current_directory = os.path.dirname(os.path.abspath(__file__))
        filename = "nes_results_dense_100k.json"

        with open(os.path.join(current_directory, filename), "w") as f:
            json.dump(results_data, f, indent=4)
    
        print(json.dumps(results_data, indent=4))
        
    produce_results()
        