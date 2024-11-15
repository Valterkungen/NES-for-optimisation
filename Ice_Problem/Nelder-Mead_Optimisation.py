import math
import numpy as np
import matplotlib.pyplot as plt
import TriFEMLibGF24
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
  #print(f'Print a: {a}')
  #print(f'Print b: {b}')
  iteration_time = time.time() - start_time
  #print(f'Iteration time: {iteration_time}')
  return dev

class NelderMeadOptimizer:
    def __init__(self, 
                 bounds=[(0, 2), (-10, 10)],
                 alpha=1.0,    # reflection coefficient
                 beta=0.5,     # contraction coefficient
                 gamma=2.0,    # expansion coefficient
                 delta=0.5,    # shrink coefficient
                 max_iter=100,
                 tolerance=1e-6):
        self.bounds = bounds
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def _clip_to_bounds(self, point):
        """Clip a point to stay within bounds"""
        return np.clip(point, 
                      [self.bounds[0][0], self.bounds[1][0]], 
                      [self.bounds[0][1], self.bounds[1][1]])
    
    def _initialize_simplex(self, initial_guess):
        """Initialize the simplex around the initial guess"""
        n = len(initial_guess)
        simplex = np.zeros((n + 1, n))
        
        # First vertex is the initial guess
        simplex[0] = initial_guess
        
        # Other vertices are created by moving along each dimension
        step_sizes = [0.05 * (self.bounds[i][1] - self.bounds[i][0]) for i in range(n)]
        
        for i in range(n):
            point = initial_guess.copy()
            point[i] += step_sizes[i]
            simplex[i + 1] = self._clip_to_bounds(point)
            
        return simplex
    
    def optimize(self, initial_guess=None):
        start_time = time.time()
        n = 2  # dimension of the problem
        
        # If no initial guess provided, start at center of bounds
        if initial_guess is None:
            initial_guess = np.array([
                (self.bounds[0][0] + self.bounds[0][1]) / 2,
                (self.bounds[1][0] + self.bounds[1][1]) / 2
            ])
        
        # Initialize simplex
        simplex = self._initialize_simplex(initial_guess)
        
        # Evaluate all points in simplex
        values = np.array([find_dev(point) for point in simplex])
        
        print("\nStarting Nelder-Mead optimization")
        print(f"Initial simplex shape: {simplex.shape}")
        print(f"Initial best value: {np.min(values):.10f}")
        print("-" * 50)
        
        for iteration in range(self.max_iter):
            iter_start_time = time.time()
            
            # Order points in simplex from best to worst
            order = np.argsort(values)
            simplex = simplex[order]
            values = values[order]
            
            # Store best solution
            if values[0] < self.best_fitness:
                self.best_fitness = values[0]
                self.best_solution = simplex[0].copy()
            
            # Calculate centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            reflection = self._clip_to_bounds(
                centroid + self.alpha * (centroid - simplex[-1])
            )
            reflection_value = find_dev(reflection)
            
            if values[0] <= reflection_value < values[-2]:
                # Accept reflection
                simplex[-1] = reflection
                values[-1] = reflection_value
            
            elif reflection_value < values[0]:
                # Try expansion
                expansion = self._clip_to_bounds(
                    centroid + self.gamma * (reflection - centroid)
                )
                expansion_value = find_dev(expansion)
                
                if expansion_value < reflection_value:
                    simplex[-1] = expansion
                    values[-1] = expansion_value
                else:
                    simplex[-1] = reflection
                    values[-1] = reflection_value
            
            else:
                # Try contraction
                contraction = self._clip_to_bounds(
                    centroid + self.beta * (simplex[-1] - centroid)
                )
                contraction_value = find_dev(contraction)
                
                if contraction_value < values[-1]:
                    simplex[-1] = contraction
                    values[-1] = contraction_value
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = self._clip_to_bounds(
                            simplex[0] + self.delta * (simplex[i] - simplex[0])
                        )
                        values[i] = find_dev(simplex[i])
            
            # Store iteration statistics
            iter_time = time.time() - iter_start_time
            iter_stats = {
                'iteration': iteration,
                'best_point': simplex[0].copy(),
                'best_value': values[0],
                'worst_value': values[-1],
                'simplex_size': np.std(simplex, axis=0),
                'time': iter_time
            }
            self.history.append(iter_stats)
            
            # Print progress
            if (iteration + 1) % 5 == 0:
                print(f"\nIteration {iteration + 1}/{self.max_iter}")
                print(f"Best deviation: {values[0]:.10f}")
                print(f"At a={simplex[0][0]:.6f}, b={simplex[0][1]:.6f}")
                print(f"Simplex size: {np.std(simplex, axis=0)}")
                print(f"Iteration time: {iter_time:.2f}s")
            
            # Check convergence
            if np.max(np.std(simplex, axis=0)) < self.tolerance:
                print("\nConverged! Simplex size below tolerance.")
                break
        
        self.total_time = time.time() - start_time
    
    def print_report(self):
        print("\nNelder-Mead Optimization Report")
        print("-" * 50)
        print(f"\nTotal Optimization Time: {self.total_time:.2f} seconds")
        
        print("\nSearch Bounds:")
        print(f"a: [{self.bounds[0][0]}, {self.bounds[0][1]}]")
        print(f"b: [{self.bounds[1][0]}, {self.bounds[1][1]}]")
        
        print("\nBest Solution Found:")
        print(f"a: {self.best_solution[0]:.10f}")
        print(f"b: {self.best_solution[1]:.10f}")
        print(f"Deviation: {self.best_fitness:.10f}")
        
        print("\nOptimization Progress:")
        for i in range(0, len(self.history), max(1, len(self.history)//5)):
            stats = self.history[i]
            print(f"\nIteration {stats['iteration']}:")
            print(f"  Best deviation: {stats['best_value']:.10f}")
            print(f"  Worst deviation: {stats['worst_value']:.10f}")
            print(f"  Simplex size: {stats['simplex_size']}")

# Usage example with multiple starts
if __name__ == "__main__":
    n_starts = 5  # Number of random starts
    
    best_fitness = float('inf')
    best_solution = None
    
    print(f"\nRunning Nelder-Mead with {n_starts} random starts")
    print("-" * 50)
    
    for start in range(n_starts):
        print(f"\nRandom Start {start + 1}/{n_starts}")
        
        # Generate random initial guess within bounds
        initial_guess = np.array([
            np.random.uniform(0, 2),      # a
            np.random.uniform(-10, 10)     # b
        ])
        
        # Create and run optimizer
        optimizer = NelderMeadOptimizer(max_iter=100, tolerance=1e-6)
        optimizer.optimize(initial_guess)
        
        # Update best solution
        if optimizer.best_fitness < best_fitness:
            best_fitness = optimizer.best_fitness
            best_solution = optimizer.best_solution
            
        print(f"\nStart {start + 1} complete:")
        print(f"Best deviation: {optimizer.best_fitness:.10f}")
        print(f"At a={optimizer.best_solution[0]:.6f}, b={optimizer.best_solution[1]:.6f}")
    
    print("\nOverall Best Solution:")
    print(f"a: {best_solution[0]:.10f}")
    print(f"b: {best_solution[1]:.10f}")
    print(f"Deviation: {best_fitness:.10f}")