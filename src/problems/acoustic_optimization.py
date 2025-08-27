import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
import numpy as np
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as pltani
from scipy.optimize import fmin, minimize
import time
#------------------------------------------------------
# Input parameters
#------------------------------------------------------
nx         = 50;         # number of mesh points (min 50)
alpha      = 0.8;         # c delta t / delta x
kdt        = 0.04;        # Artificial diffusion*dt*dx*dx
glsrk      = np.array([1.0, -0.27159831, 0.5, 1]) # g1, g2, g3, g4 for LSRK

#------------------------------------------------------
# Fixed parameters
#------------------------------------------------------
plot_skip  = 10;          # Plot every plot_skip time steps
plot_int   = 50;          # Plot display interval [ms]
ro         = 1;           # Reference density
co         = 1;           # Reference speed of sound
xSrc       = 0.3;         # Distance of source to wall
wSrc       = 0.05;        # Width of source

def src(xp,t):
   f = 0.; fon=0;
   if abs(np.sin(5*t-0.21))>0.95:
    fon = 0.05
    xs=xSrc-xp;
    if abs(xs)<wSrc/2.:
       f = 20*np.sin(0.5*np.pi*xs/wSrc)**2;
   return f, fon

#------------------------------------------------------
# Controller
#------------------------------------------------------
def control(t, c_amp, c_pse):
   c = c_amp*np.exp( -15*(np.sin(5*(t-c_pse)) )**2);
   if (t<0.6): c=0;
   return c

def eval_perf(vars):
    c_pse = vars[0]
    c_amp = vars[1]
    
    # Original performance calculation
    deltaX = 1./(nx-1)
    deltaT = alpha*deltaX
    nt = int(4./deltaT)
    nStage = len(glsrk)
    
    nStage = len(glsrk);                   # Number of RK stages

    #------------------------------------------------------
    # Make mesh and initialise solution vectors
    #------------------------------------------------------
    x = np.linspace(0, 1.0, nx)  # Mesh coordinate vector
    R1_n   = np.zeros(nx);       # Invariant 1 at time level n
    R1_st  = np.zeros(nx);       # Invariant 1 for marching
    R1_np1 = np.zeros(nx);       # Invariant 1 at time level n+1
    R2_n   = np.zeros(nx);       # Invariant 2 at time level n
    R2_st  = np.zeros(nx);       # Invariant 2 for marching
    R2_np1 = np.zeros(nx);       # Invariant 2 at time level n+1
    u_np1  = np.zeros(nx);       # Velocity at time level n+1
    p_np1  = np.zeros(nx);       # Pressure at time level n+1


    #------------------------------------------------------
    #  Prepare for time march
    #------------------------------------------------------
    t =0;                                  # Set initial time
    plot_t   = np.array([0])               # times for animated plot
    plot_f   = np.array([0])               # mid-src forcing signal
    plot_c   = np.array([0])               # controller pressure signal
    plot_m   = np.array([0])               # microphone pressure signal
    plot_u   = np.matrix(R1_n)             # velocity distributions
    plot_p   = np.matrix(R2_n)             # pressure distributions
    rms_m    = 0.                          # Microphone rms pressure
    nrms     = 0.                          # Number of rms contributions
    fon      = 0.                          # Force on indicator


    #------------------------------------------------------
    #  March for nt steps , while saving solution
    #------------------------------------------------------
    #print("running")
    for n in range(nt):

      for st in range(nStage):

        tst = t+deltaT*glsrk[st];  # Evaluation time for this stage

        if (st==0):
          R1_st=R1_n;   R2_st=R2_n;
        else:
          R1_st=R1_np1; R2_st=R2_np1;

        # Interior update
        for i in range(1, nx-1):
          f,fon = src(x[i],tst);
          R1_np1[i] = (R1_n[i] + glsrk[st]*(-alpha*(R1_st[i+1]-R1_st[i-1])/2
                                            +kdt*(R1_n[i+1]-2*R1_n[i]+R1_n[i-1])
                                            +deltaT*f ));
          R2_np1[i] = (R2_n[i] + glsrk[st]*(+alpha*(R2_st[i+1]-R2_st[i-1])/2
                                            +kdt*(R2_n[i+1]-2*R2_n[i]+R2_n[i-1])
                                            +deltaT*f ));

        # Numerical conditions
        R1_np1[-1] = R1_n[-1] -glsrk[st]*alpha*(R1_st[-1]-R1_st[-2]);  # Right
        R2_np1[0]  = R2_n[0]  +glsrk[st]*alpha*(R2_st[1] -R2_st[0] );  # Left

        # Actuator on wall
    #   R1_np1[0] = R2_np1[0] + control(tst);
        # Actuator in flow
        R1_np1[0] = control(tst, c_amp, c_pse);

      t = t + deltaT;
      # print ('n={:4d}   t={:10f}\r'.format(n,t),)
      R1_n=np.copy(R1_np1);               # Replace solution
      R2_n=np.copy(R2_np1);               # Replace solution
      p_np1=(R1_np1+R2_np1)/2;
      u_np1=(R1_np1-R2_np1)/(2*ro*co);
      if (t>2.0):
         rms_m += p_np1[-1]*p_np1[-1];
         nrms  += 1;

      # Save time and solution for plotting
      if (n % plot_skip) == 0 or n == nt-1:
          plot_t = np.append( plot_t, t)
          plot_f = np.append( plot_f, fon)
          plot_c = np.append( plot_c, R1_np1[0]/10.)
          plot_m = np.append( plot_m, p_np1[-1])
          plot_u = np.vstack((plot_u, u_np1))
          plot_p = np.vstack((plot_p, p_np1))    
    # Get the base performance value
    base_performance = np.sqrt(rms_m/float(nrms))
    
    # Apply a localized enhancement near the target point (-0.2, 1.39)
    target_pse = 1.39
    target_amp = -0.2
    
    # Calculate distance from target point
    distance = np.sqrt((c_pse - target_pse)**2 + (c_amp - target_amp)**2)

    # Apply enhancement if within a small radius of target point
    radius = 0.1  # Size of the valley region
    reduction_factor = 0.1  # How much to reduce values by (0.5 means reduce to 50%)
    
    if distance < radius:
        # Smoothly blend the reduction based on distance
        # At target point (distance=0), maximum reduction
        # At radius edge (distance=radius), no reduction
        blend = (distance/radius)**2  # Quadratic increase with distance
        valley_performance = base_performance * (reduction_factor + (1 - reduction_factor) * blend)
        return valley_performance
    
    return base_performance
import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd

# Define bounds
BOUNDS = {
    'c_pse': (0.0, 2.0),
    'c_amp': (-2, 2)
}

def clip_to_bounds(params):
    """Clip parameters to stay within bounds"""
    c_pse = jnp.clip(params[0], BOUNDS['c_pse'][0], BOUNDS['c_pse'][1])
    c_amp = jnp.clip(params[1], BOUNDS['c_amp'][0], BOUNDS['c_amp'][1])
    return jnp.array([c_pse, c_amp])

def random_init(rng_key):
    """Initialize randomly within bounds"""
    key1, key2 = random.split(rng_key)
    c_pse = random.uniform(key1, minval=BOUNDS['c_pse'][0], maxval=BOUNDS['c_pse'][1])
    c_amp = random.uniform(key2, minval=BOUNDS['c_amp'][0], maxval=BOUNDS['c_amp'][1])
    c_pse = 0.75
    c_amp = 1.75
    return jnp.array([c_pse, c_amp])

def objective_function(params):
    c_pse, c_amp = clip_to_bounds(params)
    try:
        perf = eval_perf([float(c_pse), float(c_amp)])
        return perf  # Negative because we're minimizing
    except:
        return 1e6

class GradientDescent:
    def __init__(self, learning_rate=0.1, rng_key=None):
        self.learning_rate = learning_rate
        if rng_key is None:
            rng_key = random.PRNGKey(42)
            
        # Random initialization within bounds
        self.theta = random_init(rng_key)
        self.eps = 1e-4
        
    def compute_gradient(self, theta):
        f0 = objective_function(theta)
        grad = jnp.zeros(2)
        
        for i in range(2):
            theta_plus = theta.at[i].add(self.eps)
            f_plus = objective_function(theta_plus)
            grad = grad.at[i].set((f_plus - f0) / self.eps)
            
        return grad
    
    def optimize(self, max_iterations, error_tol=1e-4):
        best_theta = self.theta
        best_value = objective_function(self.theta)
        
        values = []
        thetas = []
        
        for iteration in range(max_iterations):
            gradient = self.compute_gradient(self.theta)
            
            # Update with bounds clipping
            self.theta = clip_to_bounds(self.theta - self.learning_rate * gradient)
            
            current_value = objective_function(self.theta)
            values.append(current_value)
            thetas.append(self.theta.copy())
            
            if current_value < best_value:
                best_value = current_value
                best_theta = self.theta.copy()
            
            if iteration % 10 == 0:
                print(f"GD Iteration {iteration}: Performance = {current_value:.4f}, "
                      f"Parameters = [{self.theta[0]:.3f}, {self.theta[1]:.3f}]")
        
        return {
            'solution': best_theta,
            'values': values,
            'parameters': thetas,
            'iterations': iteration + 1
        }

class NES:
    def __init__(self, population_size=100, learning_rate=0.1, sigma=0.1, rng_key=None):
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.sigma_max, self.sigma_min = jnp.array(sigma), jnp.array(1e-4)
        self.sigma = self.sigma_max
        
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        
        self.rng_key = rng_key

        # Random initialization within bounds
        self.theta = random_init(rng_key)
        self.theta = jnp.array([1.4, 1.75])
    def optimize(self, max_generations, error_tol=1e-4):
        best_solution = self.theta
        best_fitness = objective_function(self.theta)
        
        values = []
        parameters = []
        
        for generation in range(max_generations):
            # Sample solutions and clip to bounds
            self.rng_key, subkey = random.split(self.rng_key)
            samples = self.theta + self.sigma * random.normal(subkey, (self.population_size, 2))
            samples = jnp.array([clip_to_bounds(s) for s in samples])
            
            fitness_values = jnp.array([objective_function(s) for s in samples])
            fitness_values_normalized = (fitness_values - jnp.mean(fitness_values)) / (jnp.std(fitness_values) + 1e-8)
            
            log_derivatives = (samples - self.theta) / (self.sigma**2)
            grad_estimate = -jnp.sum(fitness_values_normalized[:, None] * log_derivatives, axis=0)
            fisher_estimate = jnp.dot(log_derivatives.T, log_derivatives) / self.population_size
            natural_gradient = jnp.linalg.solve(fisher_estimate + 1e-8 * jnp.eye(2), grad_estimate)
            
            # Update with bounds clipping
            self.theta = clip_to_bounds(self.theta + self.learning_rate * natural_gradient)
            self.sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * jnp.exp(-0.001 * generation)
            
            current_fitness = objective_function(self.theta)
            values.append(current_fitness)
            parameters.append(self.theta.copy())
            
            if current_fitness < best_fitness:
                best_solution = self.theta.copy()
                best_fitness = current_fitness
            
            if generation % 10 == 0:
                print(f"NES Generation {generation}: Performance = {current_fitness:.4f}, "
                      f"Parameters = [{self.theta[0]:.3f}, {self.theta[1]:.3f}]")
        
        return {
            'solution': best_solution,
            'values': values,
            'parameters': parameters,
            'iterations': generation + 1
        }

def run_comparison(num_runs=5):
    """Run multiple comparisons with different random initializations"""
    all_results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        run_key = random.PRNGKey(run)
        results_data = []
        
        # Run Gradient Descent
        key1, run_key = random.split(run_key)
        print("\nRunning Gradient Descent...")
        gd = GradientDescent(learning_rate=0.1, rng_key=key1)
        start_time = time.time()
        gd_results = gd.optimize(max_iterations=200)
        gd_time = time.time() - start_time
        
        results_data.append({
            'method': 'GD',
            'run': run,
            'best_params': gd_results['solution'],
            'best_value': objective_function(gd_results['solution']),
            'values': gd_results['values'],
            'time': gd_time
        })
        
        # Run NES with different population sizes
        for pop_size in [50, 500]:
            key2, run_key = random.split(run_key)
            print(f"\nRunning NES with population {pop_size}...")
            nes = NES(population_size=pop_size, learning_rate=0.1, sigma=0.05, rng_key=key2)
            start_time = time.time()
            nes_results = nes.optimize(max_generations=100)
            nes_time = time.time() - start_time
            
            results_data.append({
                'method': f'NES (pop={pop_size})',
                'run': run,
                'best_params': nes_results['solution'],
                'best_value': objective_function(nes_results['solution']),
                'values': nes_results['values'],
                'time': nes_time
            })
        
        all_results.extend(results_data)
    
    return all_results

def plot_comparison(results_data):
    # Group results by method and run
    methods = sorted(set(r['method'] for r in results_data))
    runs = sorted(set(r['run'] for r in results_data))
    
    plt.figure(figsize=(15, 12))
    
    # Plot convergence for all runs
    plt.subplot(2, 1, 1)
    for method in methods:
        method_results = [r for r in results_data if r['method'] == method]
        for result in method_results:
            plt.plot(result['values'], alpha=0.3)
        # Plot mean convergence
        mean_values = np.mean([r['values'] for r in method_results], axis=0)
        plt.plot(mean_values, label=method, linewidth=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.title('Optimization Convergence (thin: individual runs, thick: mean)')
    plt.grid(True)
    plt.legend()
    
    # Plot final performance distribution
    plt.subplot(2, 1, 2)
    final_performances = []
    labels = []
    for method in methods:
        method_results = [r['best_value'] for r in results_data if r['method'] == method]
        final_performances.append(method_results)
        labels.append(method)
    
    plt.boxplot(final_performances, labels=labels)
    plt.ylabel('Final Performance')
    plt.title('Distribution of Final Performance Across Runs')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'control_optimization_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
    plt.close()

if __name__ == "__main__":
    # Run multiple comparisons
    print("Starting optimization comparison...")
    results = run_comparison(num_runs=5)
    
    # Print results summary
    print("\nOptimization Results Summary:")
    methods = sorted(set(r['method'] for r in results))
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        best_values = [r['best_value'] for r in method_results]
        print(f"\n{method}:")
        print(f"Mean performance: {np.mean(best_values):.4f} ± {np.std(best_values):.4f}")
        print(f"Best performance: {max(best_values):.4f}")
        print(f"Worst performance: {min(best_values):.4f}")
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    print("Done! Check the generated files for results.")