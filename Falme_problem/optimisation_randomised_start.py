from scipy.optimize import minimize
import numpy as np
from compute_flame_performance import compute_flame_performance

class FlameOptimizer:
    def __init__(self, n_starts=10):
        self.n_starts = n_starts
        self.bounds = [(-2.5, 2.5), (0, 10.0)]  # Fixed bounds for the search space
        self.results = []

    def objective(self, params):
        nozc, nozw = params
        return -compute_flame_performance(nozc, nozw, refine=1, plot=False)

    def generate_random_start(self):
        return np.array([
            np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
            np.random.uniform(self.bounds[1][0], self.bounds[1][1])
        ])

    def optimize(self):
        for i in range(self.n_starts):
            initial_guess = self.generate_random_start()
            result = minimize(
                self.objective,
                initial_guess,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            optimization_result = {
                'start_point': initial_guess,
                'final_point': result.x,
                'performance': -result.fun,
                'success': result.success,
                'iterations': result.nit
            }
            self.results.append(optimization_result)

    def analyze_results(self):
        # Convert results to numpy array for easier analysis
        performances = np.array([r['performance'] for r in self.results])
        
        # Find the best result
        best_idx = np.argmax(performances)
        best_result = self.results[best_idx]
        
        # Calculate statistics
        mean_performance = np.mean(performances)
        std_performance = np.std(performances)
        
        # Check for potential local minima
        # Group similar solutions (within 1% of parameter range)
        unique_solutions = []
        thresholds = [
            (self.bounds[0][1] - self.bounds[0][0]) * 0.01,  # 1% of nozc range
            (self.bounds[1][1] - self.bounds[1][0]) * 0.01   # 1% of nozw range
        ]
        
        for result in self.results:
            new_solution = True
            for unique in unique_solutions:
                if np.all(np.abs(result['final_point'] - unique['final_point']) < thresholds):
                    new_solution = False
                    break
            if new_solution:
                unique_solutions.append(result)
        
        return {
            'best_result': best_result,
            'statistics': {
                'mean_performance': mean_performance,
                'std_performance': std_performance,
                'n_local_minima': len(unique_solutions)
            },
            'unique_solutions': unique_solutions
        }

    def print_report(self):
        analysis = self.analyze_results()
        
        print(f"\nOptimization Report ({self.n_starts} random starts)")
        print("-" * 50)
        print("\nSearch Bounds:")
        print(f"Nozzle Center (nozc): [{self.bounds[0][0]}, {self.bounds[0][1]}]")
        print(f"Nozzle Width (nozw): [{self.bounds[1][0]}, {self.bounds[1][1]}]")
        
        print("\nBest Solution Found:")
        print(f"Nozzle Center (nozc): {analysis['best_result']['final_point'][0]:.10f}")
        print(f"Nozzle Width (nozw): {analysis['best_result']['final_point'][1]:.10f}")
        print(f"Performance: {analysis['best_result']['performance']:.10f}")
        print(f"Starting Point: nozc={analysis['best_result']['start_point'][0]:.10f}, "
              f"nozw={analysis['best_result']['start_point'][1]:.10f}")
        
        print("\nStatistics:")
        print(f"Mean Performance: {analysis['statistics']['mean_performance']:.10f}")
        print(f"Standard Deviation: {analysis['statistics']['std_performance']:.10f}")
        print(f"Number of Unique Solutions: {analysis['statistics']['n_local_minima']}")
        
        if analysis['statistics']['n_local_minima'] > 1:
            print("\nPotential Local Minima Found:")
            for i, sol in enumerate(analysis['unique_solutions'], 1):
                print(f"\nLocal Minimum {i}:")
                print(f"  nozc: {sol['final_point'][0]:.10f}")
                print(f"  nozw: {sol['final_point'][1]:.10f}")
                print(f"  performance: {sol['performance']:.10f}")
                print(f"  starting point: nozc={sol['start_point'][0]:.10f}, "
                      f"nozw={sol['start_point'][1]:.10f}")

# Usage example
if __name__ == "__main__":
    # Create optimizer instance with 10 random starts
    optimizer = FlameOptimizer(n_starts=10)
    
    # Run optimization
    optimizer.optimize()
    
    # Print detailed report
    optimizer.print_report()