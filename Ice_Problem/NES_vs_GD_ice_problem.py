import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import TriFEMLibGF24
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
import time

def find_dev(vars):
  start_time = time.time()
  #=========================================================
  # Input parameters
  #=========================================================
  n=1    # Mesh refinement factor
  a=vars[0]              # ice thickness amplitude
  b=vars[1]               # Displacement of minimum from zero

  #=========================================================
  # Fixed parameters
  #=========================================================
  xmQ,ymQ =-10, 10.5     # Position of satellite Q
  xmS,ymS = 10, 9.39     # Position of satellite S
  urQ=149.93642043913104 # satellite potential at location Q
  urS=130.41395604946775 # satellite potential at location S
  yIce=2.0;              # Upper ice boundary

  #=========================================================
  # Create the mesh
  #=========================================================
  mesh = TriFEMLibGF24.TriMesh();
  mesh.loadMesh(n, yIce, a, b)
  # print ("Mesh: nVert=",mesh.nVert,"nElem=",mesh.nElem);
  #mesh.plotMesh();# quit(); 1

  #=========================================================
  # Create a finite-element space.
  # This object maps the degrees of freedom in an element
  # to the degrees of freedom of the global vector.
  #=========================================================
  fes = TriFEMLibGF24.LinTriFESpace(mesh)

  #=========================================================
  # Prepare the global left-hand matrix, right-hand vector
  # and solution vector
  #=========================================================
  sysDim = fes.sysDim
  LHM    = np.zeros((sysDim,sysDim));
  RHV    = np.zeros(sysDim);
  solVec = np.zeros(sysDim);

  #=========================================================
  # Assemble the global left-hand matrix and
  # right-hand vector by looping over the elements
  #print ("Assembling system of dimension",sysDim)
  #=========================================================

  for elemIndex in range(mesh.nElem):

    #----------------------------------------------------------------
    # Create a FiniteElement object for
    # the element with index elemIndex
    #----------------------------------------------------------------
    elem = TriFEMLibGF24.LinTriElement(mesh,elemIndex)

    #----------------------------------------------------------------
    # Initialise the element vector and matrix to zero.
    # In this case we have only one unknown varible in the PDE (u),
    # So the element vector dimension is the same as
    # the number of shape functions (psi_i)  in the element.
    #----------------------------------------------------------------
    evDim   = elem.nFun
    elemVec = np.zeros((evDim))
    elemMat = np.zeros((evDim,evDim))

    #----------------------------------------------------------------
    # Evaluate the shape function integrals in the vector and matrix
    # by looping over integration points (integration by quadrature)
    # int A = sum_ip (w_ip*A_ip) where A is the function to be
    # integrated and w_ip is the weight of an integration point
    #----------------------------------------------------------------
    for ip in range(elem.nIP):

      # Retrieve the coordinates and weight of the integration point
      xIP  = elem.ipCoords[ip,0]
      yIP  = elem.ipCoords[ip,1]
      ipWeight = elem.ipWeights[ip];

      yWat = TriFEMLibGF24.iceWat(xIP, a, b)

      # Compute the local value of the source term, f
      if yIP<=yWat :
        fIP = 0
      elif yIP<=yIce :
        fIP = -200
      else :
        fIP = 0.


      # Retrieve the gradients evaluated at this integration point
      # - psi[i] is the value of the function psi_i at this ip.
      # - gradPsi[i] is a vector contraining the x and y
      #   gradients of the function psi_i at this ip
      #   e.g.
      #     gradPsi[2][0] is the x gradient of shape 2 at point xIP,yIP
      #     gradPsi[2][1] is the y gradient of shape 2 at point xIP,yIP
      psi     = elem.getShapes(xIP,yIP)
      gradPsi = elem.getShapeGradients(xIP,yIP)


      # Add this ip's contribution to the integrals in the
      # element vector and matrix
      for i in range(evDim):
        elemVec[i] += ipWeight*psi[i]*fIP;   # Right-hand side of weak form
        for j in range(evDim):
          # ***** Change the line below for the desired left-hand side
          # elemMat[i,j] += 1.
          elemMat[i, j] += - ipWeight * (gradPsi[i][0] * gradPsi[j][0] + gradPsi[i][1] * gradPsi[j][1])


    #----------------------------------------------------------------
    # Add the completed element matrix and vector to the system
    #----------------------------------------------------------------
    fes.addElemMat(elemIndex, elemMat, LHM )
    fes.addElemVec(elemIndex, elemVec, RHV )

  #=========================================================
  #print ("Applying boundary conditions")
  #=========================================================
  # Lower boundary conditions
  #=========================================================
  coord = np.asarray(fes.lowerCoords)[:,0]
  for i in range(fes.nLower):
     row = int(fes.lowerDof[i]);
     LHM[row,:]   = 0.
     LHM[row,row] = 1.
     RHV[row]     = 0.


  #=========================================================
  #print ("Solving the system")
  #=========================================================
  #fes.printMatVec(LHM,RHV,"afterConstraints")
  solVec = np.linalg.solve(LHM, RHV)


  #=========================================================
  # Find and output potential and the difference from the
  # reference values at the measurement points
  #=========================================================
  umQ  =fes.getLeftBndU(solVec,   ymQ)
  umS  =fes.getRightBndU(solVec,  ymS)
  dev = math.sqrt( (umQ-urQ)*(umQ-urQ) + (umS-urS)*(umS-urS) )
  #print(f'Print deviation: {dev}')
  #print(f'Print a: {a} - b: {b}')
  iteration_time = time.time() - start_time
  #print(f'Iteration time: {iteration_time}')
  return dev

# Define bounds for the ice problem
BOUNDS = {
    'a': (0.0, 2.0),    # thickness amplitude bounds
    'b': (-2.0, 2.0)    # displacement bounds
}

def clip_to_bounds(params):
    """Clip parameters to stay within bounds"""
    a = jnp.clip(params[0], BOUNDS['a'][0], BOUNDS['a'][1])
    b = jnp.clip(params[1], BOUNDS['b'][0], BOUNDS['b'][1])
    return jnp.array([a, b])

def random_init(rng_key):
    """Initialize randomly within bounds"""
    key1, key2 = random.split(rng_key)
    a = random.uniform(key1, minval=BOUNDS['a'][0], maxval=BOUNDS['a'][1])
    b = random.uniform(key2, minval=BOUNDS['b'][0], maxval=BOUNDS['b'][1])
    return jnp.array([a, b])

def objective_function(params):
    a, b = clip_to_bounds(params)
    try:
        dev = find_dev([float(a), float(b)])
        return dev  # Already minimizing deviation
    except:
        return 1e6

class GradientDescent:
    def __init__(self, learning_rate=0.01, rng_key=None):
        self.learning_rate = learning_rate
        if rng_key is None:
            rng_key = random.PRNGKey(42)
            
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
            
            if iteration % 5 == 0:
                print(f"GD Iteration {iteration}: Deviation = {current_value:.4f}, "
                      f"Parameters = [a={self.theta[0]:.3f}, b={self.theta[1]:.3f}]")
        
        return {
            'solution': best_theta,
            'values': values,
            'parameters': thetas,
            'iterations': iteration + 1
        }

class NES:
    def __init__(self, population_size=100, learning_rate=0.01, sigma=0.1, rng_key=None):
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.sigma_max, self.sigma_min = jnp.array(sigma), jnp.array(1e-4)
        self.sigma = self.sigma_max
        
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        
        self.rng_key = rng_key
        self.theta = random_init(rng_key)
    
    def optimize(self, max_generations, error_tol=1e-4):
        best_solution = self.theta
        best_fitness = objective_function(self.theta)
        
        values = []
        parameters = []
        
        for generation in range(max_generations):
            self.rng_key, subkey = random.split(self.rng_key)
            samples = self.theta + self.sigma * random.normal(subkey, (self.population_size, 2))
            samples = jnp.array([clip_to_bounds(s) for s in samples])
            
            fitness_values = jnp.array([objective_function(s) for s in samples])
            fitness_values_normalized = (fitness_values - jnp.mean(fitness_values)) / (jnp.std(fitness_values) + 1e-8)
            
            log_derivatives = (samples - self.theta) / (self.sigma**2)
            grad_estimate = -jnp.sum(fitness_values_normalized[:, None] * log_derivatives, axis=0)
            fisher_estimate = jnp.dot(log_derivatives.T, log_derivatives) / self.population_size
            natural_gradient = jnp.linalg.solve(fisher_estimate + 1e-8 * jnp.eye(2), grad_estimate)
            
            self.theta = clip_to_bounds(self.theta + self.learning_rate * natural_gradient)
            self.sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * jnp.exp(-0.001 * generation)
            
            current_fitness = objective_function(self.theta)
            values.append(current_fitness)
            parameters.append(self.theta.copy())
            
            if current_fitness < best_fitness:
                best_solution = self.theta.copy()
                best_fitness = current_fitness
            
            if generation % 5 == 0:
                print(f"NES Generation {generation}: Deviation = {current_fitness:.4f}, "
                      f"Parameters = [a={self.theta[0]:.3f}, b={self.theta[1]:.3f}]")
        
        return {
            'solution': best_solution,
            'values': values,
            'parameters': parameters,
            'iterations': generation + 1
        }

def run_comparison(num_runs=5):
    all_results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        run_key = random.PRNGKey(run)
        results_data = []
        
        # Run Gradient Descent
        key1, run_key = random.split(run_key)
        print("\nRunning Gradient Descent...")
        gd = GradientDescent(learning_rate=0.01, rng_key=key1)
        start_time = time.time()
        gd_results = gd.optimize(max_iterations=100)
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
        # for pop_size in [50, 500]:
        #     key2, run_key = random.split(run_key)
        #     print(f"\nRunning NES with population {pop_size}...")
        #     nes = NES(population_size=pop_size, learning_rate=0.01, sigma=0.1, rng_key=key2)
        #     start_time = time.time()
        #     nes_results = nes.optimize(max_generations=100)
        #     nes_time = time.time() - start_time
            
        #     results_data.append({
        #         'method': f'NES (pop={pop_size})',
        #         'run': run,
        #         'best_params': nes_results['solution'],
        #         'best_value': objective_function(nes_results['solution']),
        #         'values': nes_results['values'],
        #         'time': nes_time
        #     })
        
        all_results.extend(results_data)
    
    return all_results

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
        best_params = method_results[np.argmin(best_values)]['best_params']
        print(f"\n{method}:")
        print(f"Mean deviation: {np.mean(best_values):.4f} ± {np.std(best_values):.4f}")
        print(f"Best deviation: {min(best_values):.4f}")
        print(f"Best parameters: a={best_params[0]:.4f}, b={best_params[1]:.4f}")
        print(f"Worst deviation: {max(best_values):.4f}")
    
    # Create plots
    print("\nGenerating comparison plots...")
    print("Done! Check the generated files for results.")