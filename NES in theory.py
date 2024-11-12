import jax.numpy as jnp
import jax.random as random
import numpy as np
# Program to optimize rosenbrock function using Natural Evolution Strategies
# Define the Rosenbrock function
def quadratic(z):
    return jnp.sum(z**2)

def fitness_function(w):
    return -quadratic(w)

class NES:
    def __init__(self, dim, population_size, learning_rate, sigma, rng_key):
        self.dim = dim
        self.population_size = population_size  # λ in the pseudo code
        self.learning_rate = learning_rate  # α in the pseudo code
        self.sigma = sigma  # Standard deviation for sampling
        self.theta = jnp.ones(dim)  # Initial solution
        self.rng_key = rng_key  # JAX RNG key for reproducibility

    def train(self, num_generations):
        for generation in range(num_generations):

            self.rng_key, subkey = random.split(self.rng_key)
            samples = self.theta + self.sigma * random.normal(subkey, (self.population_size, self.dim))

            # Evaluate the fitness of the samples
            fitness_values = np.array([fitness_function(sample) for sample in samples])
            fitness_values = (fitness_values - np.mean(fitness_values)) / np.std(fitness_values)

            # Log-derivatives
            log_derivatives = (self.theta - samples) / (self.sigma**2)

            # Compute gradient estimate
            gradient_estimate = jnp.mean(fitness_values[:, None] * log_derivatives, axis=0)

            # Fisher information matrix
            fisher_information = jnp.mean(jnp.einsum('ij,ik->ijk', log_derivatives, log_derivatives), axis=0)

            # Update θ
            self.theta += self.learning_rate * jnp.linalg.solve(fisher_information, gradient_estimate)
            
            if generation % 100 == 0:
                print('iter %d. w: %s, reward: %f' % 
                (generation, str(self.theta), fitness_function(self.theta)))

        return self.theta
    


if __name__ == "__main__":
    ## Initialize RNG key for JAX random number generation
    rng_key = random.PRNGKey(0)
    
    # Initialize NES and train
    dim = 2  # Rosenbrock function is typically evaluated in 2D
    nes = NES(dim=dim, population_size=50, learning_rate=0.001, sigma=0.1, rng_key=rng_key)
    optimal_theta = nes.train(num_generations=1000)
    print('final solution w: %s,reward: %f' % (optimal_theta, fitness_function(optimal_theta)))
