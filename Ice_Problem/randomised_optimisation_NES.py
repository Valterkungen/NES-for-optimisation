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
  print(f'Print deviation: {dev}')
  print(f'Print a: {a} - b: {b}')
  iteration_time = time.time() - start_time
  #print(f'Iteration time: {iteration_time}')
  return dev

class DevOptimizer:
    def __init__(self, n_starts=10):
        self.n_starts = n_starts
        self.bounds = [(0, 2), (-10, 10)]  # Bounds for a and b
        self.results = []
        self.total_time = 0

    def generate_random_start(self):
        return np.array([
            np.random.uniform(self.bounds[0][0], self.bounds[0][1]),  # a
            np.random.uniform(self.bounds[1][0], self.bounds[1][1])   # b
        ])

    def optimize(self):
        start_time = time.time()
        
        for i in range(self.n_starts):
            initial_guess = self.generate_random_start()
            #print(f"\nStarting optimization {i+1}/{self.n_starts}")
            #print(f"Initial guess: a={initial_guess[0]:.6f}, b={initial_guess[1]:.6f}")
            
            result = minimize(
                find_dev,
                initial_guess,
                bounds=self.bounds,
                method='L-BFGS-B',
                options={'ftol': 1e-4, 'gtol': 1e-4}
            )
            
            optimization_result = {
                'start_point': initial_guess,
                'final_point': result.x,
                'deviation': result.fun,
                'success': result.success,
                'iterations': result.nit,
                'message': result.message
            }
            self.results.append(optimization_result)
            
            #print(f"Finished optimization {i+1}")
            print(f"Final point: a={result.x[0]:.6f}, b={result.x[1]:.6f}")
            print(f"Deviation: {result.fun:.6f}")
            
        self.total_time = time.time() - start_time

    def analyze_results(self):
        # Convert results to numpy array for easier analysis
        deviations = np.array([r['deviation'] for r in self.results])
        
        # Find the best result
        best_idx = np.argmin(deviations)  # Using argmin since we're minimizing deviation
        best_result = self.results[best_idx]
        
        # Calculate statistics
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)
        
        # Check for potential local minima
        # Group similar solutions (within 1% of parameter range)
        unique_solutions = []
        thresholds = [
            (self.bounds[0][1] - self.bounds[0][0]) * 0.01,  # 1% of a range
            (self.bounds[1][1] - self.bounds[1][0]) * 0.01   # 1% of b range
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
                'mean_deviation': mean_deviation,
                'std_deviation': std_deviation,
                'n_local_minima': len(unique_solutions)
            },
            'unique_solutions': unique_solutions
        }

    def print_report(self):
        analysis = self.analyze_results()
        
        print(f"\nOptimization Report ({self.n_starts} random starts)")
        print("-" * 50)
        print(f"\nTotal Optimization Time: {self.total_time:.2f} seconds")
        
        print("\nSearch Bounds:")
        print(f"a: [{self.bounds[0][0]}, {self.bounds[0][1]}]")
        print(f"b: [{self.bounds[1][0]}, {self.bounds[1][1]}]")
        
        print("\nBest Solution Found:")
        print(f"a: {analysis['best_result']['final_point'][0]:.10f}")
        print(f"b: {analysis['best_result']['final_point'][1]:.10f}")
        print(f"Deviation: {analysis['best_result']['deviation']:.10f}")
        print(f"Starting Point: a={analysis['best_result']['start_point'][0]:.10f}, "
              f"b={analysis['best_result']['start_point'][1]:.10f}")
        
        print("\nStatistics:")
        print(f"Mean Deviation: {analysis['statistics']['mean_deviation']:.10f}")
        print(f"Standard Deviation: {analysis['statistics']['std_deviation']:.10f}")
        print(f"Number of Unique Solutions: {analysis['statistics']['n_local_minima']}")
        
        if analysis['statistics']['n_local_minima'] > 1:
            print("\nPotential Local Minima Found:")
            for i, sol in enumerate(analysis['unique_solutions'], 1):
                print(f"\nLocal Minimum {i}:")
                print(f"  a: {sol['final_point'][0]:.10f}")
                print(f"  b: {sol['final_point'][1]:.10f}")
                print(f"  deviation: {sol['deviation']:.10f}")
                print(f"  starting point: a={sol['start_point'][0]:.10f}, "
                      f"b={sol['start_point'][1]:.10f}")

# Usage example
if __name__ == "__main__":
    # Create optimizer instance with 10 random starts
    optimizer = DevOptimizer(n_starts=10)
    
    # Run optimization
    optimizer.optimize()
    
    # Print detailed report
    optimizer.print_report()