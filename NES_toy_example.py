import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpy as np

# Define the Rosenbrock function
def quadratic(z):
    return jnp.sum(z**2)

def fitness_function(w):
    return -quadratic(w)

class NES:
    def __init__(self, dim, population_size, learning_rate, sigma, rng_key):
        self.dim = dim
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.theta = jnp.ones(dim)  # Initial solution
        self.rng_key = rng_key  # JAX RNG key for reproducibility

    def train(self, num_generations):
        for generation in range(num_generations):
            # Update RNG key and draw samples from a normal distribution
            self.rng_key, subkey = random.split(self.rng_key)
            samples = self.theta + self.sigma * random.normal(subkey, (self.population_size, self.dim))
            
            # Vectorized fitness evaluation of samples
            fitness_values = vmap(fitness_function)(samples)
            fitness_values_normalized = (fitness_values - jnp.mean(fitness_values)) / jnp.std(fitness_values)

            self.theta += self.learning_rate * jnp.dot(fitness_values_normalized, samples) / (self.population_size * self.sigma)
            if generation % 100 == 0:
                print('iter %d. w: %s, reward: %f' % 
                (generation, str(self.theta), fitness_function(self.theta)))
        return self.theta

if __name__ == "__main__":
    # Initialize RNG key for JAX random number generation
    rng_key = random.PRNGKey(0)
    
    # Initialize NES and train
    dim = 2  # Rosenbrock function is typically evaluated in 2D
    nes = NES(dim=dim, population_size=50, learning_rate=0.001, sigma=0.1, rng_key=rng_key)
    optimal_theta = nes.train(num_generations=1000)
    
    print('final solution w: %s,reward: %f' % (optimal_theta, fitness_function(optimal_theta)))

