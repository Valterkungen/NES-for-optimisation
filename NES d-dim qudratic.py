import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpy as np
from scipy.stats import mannwhitneyu
def rosenbrock(x):
    return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
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
            #utilities = self.utility_norm(fitness_values)
    
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
            
            if generation % 100 == 0:
                formatted_theta = [f"{x:.3f}" for x in self.theta]  # Format each element to 3 decimal places
                print('iter %d, lr %.3f,  w: [%s], reward: %.5f, mean reward: %.5f' % 
                (generation, self.learning_rate, ", ".join(formatted_theta), self.fitness_function(self.theta), jnp.mean(fitness_values)))


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
        ranks = jnp.argsort(-fitness_values) + 1

        # Constant value for log term
        log_half_lambda_plus_one = jnp.log(self.population_size / 2 + 1)

        # Calculate the utility for each rank
        def utility_value(rank):
            return jnp.maximum(0, log_half_lambda_plus_one - jnp.log(rank))

        # Numerator for each utility based on rank
        utilities_numerator = jnp.array([utility_value(rank) for rank in ranks])

        # Denominator to normalize utilities
        denominator = jnp.sum(jnp.array([utility_value(rank) for rank in range(1, self.population_size + 1)]))

        # Final utility values
        utilities = utilities_numerator / denominator - 1 / self.population_size
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
            print("Warning: NaN detected with eta_prime. Reverting learning rate to initial value.")
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
    
    # Set the dimension to 10 for a 10-dimensional Rosenbrock function
    dim = 3
    population_size = 50
    learning_rate = 0.001
    sigma = 0.1

    # Initialize NES with 10 dimensions and train with Rosenbrock function
    nes = NES(dim=dim, population_size=population_size, learning_rate=learning_rate, sigma=sigma, rng_key=rng_key)
    
    # Set `quad=False` to use the Rosenbrock function instead of the quadratic function
    nes.fitness_function = lambda x, quad=False: -rosenbrock(x)
    
    optimal_theta = nes.train(num_generations=2000)
    print('Optimal solution:', optimal_theta)
