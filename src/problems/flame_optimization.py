import numpy as np
import time
import matplotlib.pyplot as plt
from ..utils.compute_flame_performance import compute_flame_performance

class NaturalEvolutionStrategy:
    def __init__(self, population_size=50, learning_rate=0.1, sigma_init=0.1, seed=None):
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.sigma = np.float64(sigma_init)  # Explicit float64 type
        self.bounds = [(-1.5, 0), (0, 4.0)]
        self.dim = len(self.bounds)
        
        # Initialize mean with explicit float64 type
        self.mean = np.array([-1.3,1.5], dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        
        # Track history
        self.fitness_history = []
        self.time_history = []
        self.sigma_history = []
        self.mean_history = []
        self.best_solution = None
        self.best_fitness = float('-inf')

    def _clip_to_bounds(self, samples):
        """Clip samples to stay within bounds"""
        return np.clip(samples, 
                      np.array([b[0] for b in self.bounds], dtype=np.float64),
                      np.array([b[1] for b in self.bounds], dtype=np.float64))

    def fitness_shaping(self, rewards):
        """Convert raw fitness values to utilities using rank-based shaping"""
        ranks = np.argsort(np.argsort(-rewards))
        utilities = np.maximum(0, np.log(self.population_size/2 + 1) - np.log(ranks + 1))
        utilities = utilities - np.mean(utilities)
        utilities = utilities / (np.std(utilities) + 1e-8)
        return utilities.astype(np.float64)  # Ensure float64 output

    def optimize(self, generations=100, sigma_min=1e-5, sigma_max=1.0, tolerance=1e-6, patience=20):
        start_time = time.time()
        stagnation_counter = 0
        prev_best = float('-inf')
        
        print("\nStarting Natural Evolution Strategy optimization")
        print(f"Population size: {self.population_size}")
        print(f"Initial parameters: nozc={self.mean[0]:.6f}, nozw={self.mean[1]:.6f}")
        print("-" * 50)

        for gen in range(generations):
            gen_start_time = time.time()
            
            # Generate samples using current mean and sigma
            noise = self.rng.normal(0, 1, (self.population_size, self.dim)).astype(np.float64)
            samples = self.mean + self.sigma * noise
            samples = self._clip_to_bounds(samples)
            
            # Evaluate fitness for all samples
            fitness_values = np.array([compute_flame_performance(sample[0], sample[1], refine=1, plot=False) 
                                     for sample in samples], dtype=np.float64)
            
            # Update best solution
            max_idx = np.argmax(fitness_values)
            if fitness_values[max_idx] > self.best_fitness:
                self.best_fitness = fitness_values[max_idx]
                self.best_solution = samples[max_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Compute utilities using fitness shaping
            utilities = self.fitness_shaping(fitness_values)
            
            # Compute gradients with respect to mean and sigma
            log_derivative_mean = noise / self.sigma
            log_derivative_sigma = (np.linalg.norm(noise, axis=1)**2 - self.dim) / self.sigma
            
            # Compute Fisher information matrix
            fisher = np.dot(log_derivative_mean.T, log_derivative_mean) / self.population_size
            reg_fisher = fisher + np.eye(fisher.shape[0], dtype=np.float64) * 1e-5
            
            # Compute natural gradient
            grad_mean = np.dot(utilities, log_derivative_mean) / self.population_size
            grad_sigma = np.dot(utilities, log_derivative_sigma) / self.population_size
            
            # Update parameters using natural gradient
            self.mean = self.mean + self.learning_rate * np.linalg.solve(reg_fisher, grad_mean)
            self.mean = self._clip_to_bounds(self.mean.reshape(1, -1))[0]
            
            # Update sigma with adaptive scaling
            self.sigma = self.sigma * np.exp(self.learning_rate * 0.1 * grad_sigma)
            self.sigma = np.clip(self.sigma, 
                               np.float64(sigma_min), 
                               np.float64(sigma_max) * np.exp(-gen/generations))
            
            # Track history
            self.fitness_history.append(float(self.best_fitness))
            self.time_history.append(float(time.time() - start_time))
            self.sigma_history.append(float(self.sigma))
            self.mean_history.append(self.mean.copy())
            
            # Print progress
            if (gen + 1) % 5 == 0:
                print(f"\nGeneration {gen + 1}/{generations}")
                print(f"Best performance: {self.best_fitness:.10f}")
                print(f"At nozc={self.best_solution[0]:.6f}, nozw={self.best_solution[1]:.6f}")
                print(f"Current σ: {self.sigma:.6f}")
                print(f"Generation time: {time.time() - gen_start_time:.2f}s")
            
            # Check convergence
            if abs(self.best_fitness - prev_best) < tolerance:
                if stagnation_counter >= patience:
                    print("\nConverged! No improvement in best fitness.")
                    break
            prev_best = self.best_fitness

    def print_report(self):
        """Print optimization results"""
        print("\nNatural Evolution Strategy Optimization Report")
        print("-" * 50)
        print("\nSearch Bounds:")
        print(f"Nozzle Center (nozc): [{self.bounds[0][0]}, {self.bounds[0][1]}]")
        print(f"Nozzle Width (nozw): [{self.bounds[1][0]}, {self.bounds[1][1]}]")
        
        print("\nBest Solution Found:")
        print(f"Nozzle Center (nozc): {self.best_solution[0]:.10f}")
        print(f"Nozzle Width (nozw): {self.best_solution[1]:.10f}")
        print(f"Performance: {self.best_fitness:.10f}")
        
        print("\nFinal Parameters:")
        print(f"Final sigma: {self.sigma:.6f}")
        print(f"Total generations: {len(self.fitness_history)}")
        print(f"Total optimization time: {self.time_history[-1]:.2f} seconds")

    def plot_optimization_results(self):
        """Plot optimization results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert lists to numpy arrays for plotting
        time_array = np.array(self.time_history)
        fitness_array = np.array(self.fitness_history)
        sigma_array = np.array(self.sigma_history)
        mean_history = np.array(self.mean_history)
        
        # Plot fitness history
        ax1.plot(time_array, fitness_array)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Best Performance')
        ax1.set_title('Optimization Progress')
        
        # Plot sigma history
        ax2.plot(time_array, sigma_array)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Sigma')
        ax2.set_title('Exploration Rate (Sigma) History')
        
        # Plot parameter trajectories
        ax3.plot(time_array, mean_history[:, 0], label='nozc')
        ax3.plot(time_array, mean_history[:, 1], label='nozw')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Parameter Value')
        ax3.set_title('Parameter Evolution')
        ax3.legend()
        
        # Plot final solution in parameter space
        ax4.scatter(self.best_solution[0], self.best_solution[1], color='red', 
                   marker='*', s=200, label='Best Solution')
        ax4.set_xlabel('nozc')
        ax4.set_ylabel('nozw')
        ax4.set_title('Final Solution in Parameter Space')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('nes_flame_performance_optimization.png', dpi=300)
        plt.show()

def run_optimization():
    # Configuration
    config = {
        'population_size': 50,
        'learning_rate': 0.01,
        'sigma_init': 0.1,
        'seed': 42
    }
    
    # Initialize and run optimization
    nes = NaturalEvolutionStrategy(**config)
    nes.optimize(generations=50)
    
    # Print and plot results
    nes.print_report()
    nes.plot_optimization_results()
    
    return nes

if __name__ == "__main__":
    optimizer = run_optimization()