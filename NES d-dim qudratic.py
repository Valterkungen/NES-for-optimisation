import jax.numpy as jnp
import jax.random as random
from jax import vmap, grad
import jax
import numpy as np
from scipy.stats import mannwhitneyu
import optax
import time
import pandas as pd
# Define the Rosenbrock function
def rosenbrock(x):
    return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

class ADAM:
    def __init__(self, objective, dim, generations, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.generations = generations
        self.optimizer = optax.adam(learning_rate)
        self.objective = objective
        self.params = jnp.zeros(dim)
        self.opt_state = self.optimizer.init(self.params)
    
    # Step function with JIT compilation applied directly
    def step(self, params, opt_state):
        grads = grad(self.objective)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Apply JIT as a function decorator
    jitted_step = jax.jit(step, static_argnames=['self'])

    def train(self):
        for _ in range(self.generations):
            self.params, self.opt_state = self.jitted_step(self.params, self.opt_state)
        return self.params

        
    
class NES:
    def __init__(self, dim, population_size, learning_rate, sigma, rng_key):

        self.dim = dim
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.sigma = jnp.array(sigma)  # Standard deviation for sampling    
        self.theta = random.normal(rng_key, (dim,))  # Initial solution
        self.rng_key = rng_key  # JAX RNG key for reproducibility
        self.decay_rate = 0.99  # For example, 1% reduction per generation


    def train(self, num_generations):
        # Program should return the best solution found
        best_solution = self.theta
        best_fitness = self.fitness_function(best_solution)

        for generation in range(num_generations):
            # Update RNG key and draw samples from a normal distribution
            self.rng_key, subkey = random.split(self.rng_key)
            samples = self.theta + self.sigma * random.normal(subkey, (self.population_size, self.dim))
            
            # Vectorized fitness evaluation of samples
            fitness_values = vmap(self.fitness_function)(samples)
            #utilities = self.utility(fitness_values)
            # Log-derivatives
            log_derivatives = (samples - self.theta) / (self.sigma**2)

            # Normalize fitness values
            fitness_values_normalized = (fitness_values - jnp.mean(fitness_values)) / (jnp.std(fitness_values) + 1e-8)
            grad_estimate = jnp.sum(fitness_values_normalized[:, None] * log_derivatives, axis=0) 
        
            fisher_estimate = jnp.dot(log_derivatives.T, log_derivatives) / self.population_size
        
            natural_gradient = jnp.linalg.solve(fisher_estimate, grad_estimate)
            # Check adaptation of learningsrate
            self.adaptation_sampling(natural_gradient, fitness_values)
            self.theta += self.learning_rate * natural_gradient
            self.sigma *= self.decay_rate
            
            '''if generation % 100 == 0:
                formatted_theta = [f"{x:.3f}" for x in self.theta]  # Format each element to 3 decimal places
                print('iter %d, lr %.3f,  w: [%s], reward: %.5f, mean reward: %.5f' % 
                (generation, self.learning_rate, ", ".join(formatted_theta), self.fitness_function(self.theta), jnp.mean(fitness_values)))
            '''

            if self.fitness_function(self.theta) > best_fitness:
                best_solution = self.theta
                best_fitness = self.fitness_function(self.theta)

        return best_solution
    
    
    
    def fitness_function(self, samples, quad = True):
        #Quadratic function
        if quad:
            return -jnp.sum(samples**2)

        else:
            return -rosenbrock(samples)


    # Disregard utility function for now
    '''def utility(self, fitness_values):
    # Rank transformation: best fitness gets rank 1, worst gets rank `population_size`
        ranks = jnp.argsort(-fitness_values)
        ranks += 1
        
        # Constant value for log term
        log_half_lambda_plus_one = jnp.log(self.population_size / 2 + 1)

        # Calculate the utility for each rank
        def utility_value(rank):
            return jnp.maximum(0, log_half_lambda_plus_one - jnp.log(rank))

        # Numerator for each utility based on rank
        utilities_numerator = jnp.array([utility_value(rank) for rank in ranks])

        # Final utility values
        utilities = utilities_numerator / jnp.sum(utilities_numerator) - 1 / self.population_size

        return utilities
    
    def utility_norm(self, fitness_values):
        # Rank samples based on fitness (higher fitness gets lower rank number)
        ranks = jnp.argsort(-fitness_values)
        # Rank-based utility assignment
        utility = jnp.zeros_like(fitness_values)
        # Give positive values to higher ranks and negative values to lower ranks
        utility = jnp.max(ranks) - ranks
        
        # Normalize utilities to sum to zero
        utility = utility - jnp.mean(utility)
        
        return utility'''

    def adaptation_sampling(self, update, fitness_values, c_prime=0.1, scaling_factor=1.1, max = 1.0):
        # Propose a slightly increased learning rate
        eta_prime = jnp.minimum(scaling_factor * self.learning_rate, max)

        # Generate new samples with the hypothetical learning rate eta_prime
        self.rng_key, subkey = random.split(self.rng_key)
        self.theta_prime = self.theta + eta_prime * update
        samples_prime = self.theta_prime + self.sigma * random.normal(subkey, (self.population_size, self.dim))
        fitness_values_prime = vmap(self.fitness_function)(samples_prime)

        # Check if fitness values contain NaNs
        if jnp.isnan(fitness_values_prime).any():
            #print("Warning: NaN detected with eta_prime. Reverting learning rate to initial value.")
            self.learning_rate = self.learning_rate_initial  # Revert to safe initial value
            return

        # Calculate average fitness values for both current and hypothetical learning rates
        avg_fitness = jnp.mean(fitness_values)
        avg_fitness_prime = jnp.mean(fitness_values_prime)

        # Update learning rate based on which set of samples performed better
        if avg_fitness_prime > avg_fitness:
            # Increase the learning rate if eta_prime performs better, but keep it below max_learning_rate
            self.learning_rate = eta_prime
        else:
            # Decrease learning rate towards the initial value if eta_prime does not improve
            self.learning_rate = (1 - c_prime) * self.learning_rate + c_prime * self.learning_rate_initial
            

        

if __name__ == "__main__":
    # Initialize RNG key for JAX random number generation
    key = np.random.randint(0, 1000)
    rng_key = random.PRNGKey(key)
    
    population_size = 50
    learning_rate = 0.001
    sigma = 0.1
    num_generations = 100
    quadratic = lambda x: jnp.sum(x**2)
    
    # Set the dimension to 10 for a 10-dimensional Rosenbrock function
    #Run test for increasing dimensions and generations
    dimensions = [1, 5, 10, 25, 50]
    generations = [10, 100, 1000, 10000]
    errors_nes = []
    errors_adam = []
    times_nes = []
    times_adam = []

    for dim in dimensions:
        for gen in generations:
            nes = NES(dim=dim, population_size=population_size, learning_rate=learning_rate, sigma=sigma, rng_key=rng_key)
            adam = ADAM(objective = quadratic, dim=dim, generations=gen)
            start_time = time.time()
            optimal_theta_nes = nes.train(num_generations=gen)
            times_nes.append(time.time() - start_time)
            optimal_theta_nes_error = nes.fitness_function(optimal_theta_nes)
            errors_nes.append(optimal_theta_nes_error)
            
            start_time = time.time()
            optimal_theta_adam = adam.train() 
            times_adam.append(time.time() - start_time)
            optimal_theta_adam_error = adam.objective(optimal_theta_adam)
            errors_adam.append(optimal_theta_adam_error)

            

    #Plot the results
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    print("NES errors: ", np.array(errors_nes))
    print("ADAM errors: ", np.array(errors_adam))
    #errors_nes = [0.5, 0.3, 0.25, 0.45, 0.35, 0.28, 0.6, 0.4, 0.35]  # NES errors
    #errors_adam = [0.48, 0.33, 0.29, 0.50, 0.36, 0.3, 0.55, 0.42, 0.34]  # ADAM errors

    # Color map for unique combinations
    cmap_nes = plt.get_cmap("Blues", len(dimensions) * len(generations))
    cmap_adam = plt.get_cmap("Oranges", len(dimensions) * len(generations))
    
    # Plot 1: Error vs Dimension for each generation
    plt.figure(figsize=(10, 6))

    # Plot NES error
    for i, gen in enumerate(generations):
        color = cmap_nes(i)
        plt.plot(dimensions, errors_nes[i::len(generations)], label=f'NesES - Dim {gen}', marker='o')#, color=color)
     

    # Plot ADAM error
    for i, gen in enumerate(generations):
        color = cmap_adam(i)
        plt.plot(dimensions, errors_adam[i::len(generations)], label=f'ADAM - Dim {gen}', marker='s')#, color=color)
    
    # Add labels and legend
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.title('Error vs Dimension for NES and ADAM with Generation Coloring')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Error vs Generations for each dimension
    
    plt.figure(figsize=(10, 6))

    # Plot NES times
    for i, dim in enumerate(dimensions):
        color = cmap_nes(i)
        plt.plot(generations, errors_nes[i::len(dimensions)], label=f'NesES - Dim {dim}', marker='o')#, color=color)
     

    # Plot ADAM times
    for i, dim in enumerate(dimensions):
        color = cmap_adam(i)
        plt.plot(generations, errors_adam[i::len(dimensions)], label=f'ADAM - Dim {dim}', marker='s')#, color=color)

    # Add labels, title, and legend
    plt.xlabel('Generations')
    plt.ylabel('Error')
    plt.title('Error vs Generations for NES and ADAM across Dimensions')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Plot 3 the time taken for each algorithm
    plt.figure(figsize=(10, 6))

    # Plot NES times
    for i, dim in enumerate(dimensions):
        color = cmap_nes(i)
        plt.plot(generations, times_nes[i::len(dimensions)], label=f'NesES - Dim {dim}', marker='o')#, color=color)
     

    # Plot ADAM times
    for i, dim in enumerate(dimensions):
        color = cmap_adam(i)
        plt.plot(generations, times_adam[i::len(dimensions)], label=f'ADAM - Dim {dim}', marker='s')#, color=color)

    # Add labels and legend
    plt.xlabel('Generation')
    plt.ylabel('Time')
    plt.title('Time vs Generation for NES and ADAM with Generation Coloring')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    
