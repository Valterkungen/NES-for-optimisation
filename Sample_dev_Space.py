import numpy as np
import pandas as pd
import time
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

def sample_dev_space():
    # Create grid points
    a_values = np.linspace(0, 3, 10)  # First variable (ice thickness amplitude)
    b_values = np.linspace(-10, 10, 40)  # Second variable (displacement)
    
    # Initialize results list
    results = []
    
    # Sample the function at each grid point
    total_points = len(a_values) * len(b_values)
    current_point = 0
    
    print(f"Sampling {total_points} points...")
    
    for a in a_values:
        for b in b_values:
            current_point += 1
            print(f"Processing point {current_point}/{total_points}: a={a}, b={b}")
            
            try:
                dev = find_dev([a, b])
                results.append({
                    'a': a,
                    'b': b,
                    'dev': dev
                })
            except Exception as e:
                print(f"Error at point a={a}, b={b}: {str(e)}")
                # Store NaN for failed evaluations
                results.append({
                    'a': a,
                    'b': b,
                    'dev': np.nan
                })
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_filename = 'Ass 2 fr/dev_samples.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")
    
    return df

# Run the sampling
if __name__ == "__main__":
    start_time = time.time()
    results_df = sample_dev_space()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")