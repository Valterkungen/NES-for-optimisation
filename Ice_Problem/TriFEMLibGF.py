#================================================================= 
# TriFEMLib: A number of classes to assist in implementing
# finite-element methods on meshes of triangles
#================================================================= 
import math
import numpy as np
import matplotlib
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import scipy.interpolate as interp

#=================================================================
# TriMesh class definition.
# TriMesh objects create and hold data for meshes of triangles.
# Either structured algebraic or Delaunay triangulation can be used.
# To use, create an empty object, set the parameters below 
# and call "loadMesh"
#=================================================================
class TriMesh(object):

  #=========================================================
  # Public data
  #=========================================================
  
  x1         =  -10.;                 # Left boundary position
  x2         =  10.;                  # Right boundary position
  y1         =  1;                    # inner radius
  y2         =  11;                   # outer radius
  bnx        =  21;                   # Background elements in x
  bny_ice    =  1;                    # Background elements in y (ice)
  bny_wat    =  3;                    # Background elements in y (water)
  bny        =  15;                   # Background elements in y (space)
  dx         = (x2-x1)/float(bnx-1);  # Mesh spacing in x
  dy         = (y2-y1)/float(bny-1);  # Mesh spacing in y
  minTriArea = (dx+dy)/10000.;        # Minimum triangle size
  refine     = 1;                     # Mesh refinement factor
  meshType  = "str"                   # str=structured, otherwise Delaunay

  vertices  = None;   # Mesh Vertices as list
  vertArray = None;   # Mesh Vertices as array
  elements  = None;   # Mesh Elements
  elemArray = None;   # Mesh Elements as array
  nVert     = None;   # Number of vertArray
  nElem     = None;   # Number of elements
  dtri      = None;   # Delaunay triangulation
  mask      = None;   # Triangle mask
  leftVI    = None;   # Left boundary vertex list
  rightVI   = None;   # Right boundary vertex list
  lowerVI   = None;   # Lower boundary vertex list
  upperVI   = None;   # Upper boundary vertex list
  
  #=========================================================
  # Object constructor
  #=========================================================
  def __init__(self):  
   created = 1;
			
  #=========================================================
  # Upper boundary definition
  #=========================================================
  def topBoundShape(self, x):
    return self.y2 + np.cos(math.pi*0.5*x/(self.x2-self.x1))
  
			
  #=========================================================
  # Loads the vertices and elements (call after setting
  # the desired parameters)
  #=========================================================
  def loadMesh(self,n,yIce,alpha,beta): 

    self.bnx     *= n;
    self.bny     *= n;
    self.bny_ice *= n;
    self.bny_wat *= n;
				
    # Load background mesh  
    xb = np.linspace(self.x1, self.x2, self.bnx+1)				
    ylow = np.zeros(self.bnx+1)
    ywat = []
    for i in range(xb.size):
        ywat.append(iceWat(xb[i], alpha, beta))
    ywat = np.array(ywat)    
    yice = np.ones(self.bnx+1)*yIce
    ytop = self.topBoundShape(xb)
    #Generate y grid points
    yvert = []				
    for i in range(self.bnx+1):
      buffer1 = np.linspace(ylow[i], ywat[i], self.bny_wat+1)
      buffer2 = np.linspace(ywat[i],yice[i], self.bny_ice+1)[1:]
      buffer3 = np.linspace(yice[i], ytop[i], self.bny+1)[1:]					
      yvert.append(np.concatenate((buffer1,buffer2, buffer3)))
    yvert = np.array(yvert)		

    #Load mesh vertex vrray
    self.vertices=[];
    self.leftVI=[];
    self.rightVI=[];
    self.lowerVI=[];
    self.upperVI=[];
    vi =0
    for j in range(self.bny_wat+1+self.bny_ice+self.bny):
      for i in range(self.bnx+1):
        if (i==0):	     self.leftVI.append(vi);
        if (i==self.bnx): self.rightVI.append(vi);
        if (j==0):	     self.lowerVI.append(vi);
        if (j==self.bny): self.upperVI.append(vi);
        self.vertices.append( (xb[i], yvert[i,j]) );
        vi +=1;
   
    self.nVert=len(self.vertices);
    self.vertArray = np.asarray(self.vertices);


    # Use Delaunay triangulation and mask bnd-only elements
    self.dtri = tri.Triangulation(self.vertArray[:,0], self.vertArray[:,1]);		
    self.elements = self.dtri.triangles			
    self.nElem=len(self.elements);
    self.elemArray=np.asarray(self.elements);
				
				

  #=========================================================
  # Returns the coordinates of a vertex
  #=========================================================
  def getVertCoords(self, vertInd):
    return self.vertices[vertInd];


  #=========================================================
  # Prints the vertices and elements to a file
  #=========================================================
  def printMesh(self,basename="mesh"):
  
    mFile = open(basename+".txt", 'w')
    mFile.write("innerVI"); mFile.write(str(self.innerVI));
    mFile.write("\n\n")
    mFile.write("outerVI"); mFile.write(str(self.outerVI));
    mFile.write("\n\nVertices:\n")

    for i in range(self.nVert):
      mFile.write("vi=");mFile.write(str(i));
      mFile.write(" xy=");mFile.write(str(self.vertArray[i,:]));
      mFile.write("\n")

    mFile.write("\n\nElements:\n")
    for e in range(self.nElem):
      mFile.write("ei=");mFile.write(str(e));
      mFile.write(" vert=");mFile.write(str(self.elements[e]));
      mFile.write("\n")


  #=========================================================
  # Plots the mesh
  #=========================================================
  def plotMesh(self):
  
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(111)
    ax.set_title("Mesh")
    ax.set_xlabel('x',size=14,weight='bold')
    ax.set_ylabel('y',size=14,weight='bold')
    plt.axes().set_aspect('equal', 'datalim')
    xy = np.asarray(self.vertices);
    plt.triplot(xy[:,0],xy[:,1],self.elements,'bo-');
#   plt.savefig('mesh.png',dpi=250)
    plt.show()



#**********************************************************************



#================================================================= 
# LinTriFESpace class definition.
# LinTriFESpace provides the mapping from local element data
# to the global system for a mesh of linear triangles. 
# It also provides boundary condition row and coordinate 
# information. One variable per vertex is assumed for now.
#================================================================= 
class LinTriFESpace(object):


  #=========================================================
  # Public data
  #=========================================================
  nVar          = 1;      # Number of variables (=1 for now)
  mesh          = None    # Link to mesh object
  nLeft         = None    # Number of rows for inner BCs
  nRight        = None    # Number of rows for outer BCs
  nLower        = None    # Number of rows for inner BCs
  nUpper        = None    # Number of rows for outer BCs
  leftDof       = None    # Global dof for lower BC
  rightDof      = None    # Global dof for upper BC
  lowerDof      = None    # Global dof for lower BC
  upperDof      = None    # Global dof for upper BC
  leftCoords    = None    # Coordinates for left vertices
  rightCoords   = None    # Coordinates for right vertices
  lowerCoords   = None    # Coordinates for left vertices
  upperCoords   = None    # Coordinates for right vertices
  sysDim        = None    # Global system dimension


  #=========================================================
  # Object constructor
  #=========================================================
  def __init__(self,mesh): 
  
    self.mesh          = mesh
    self.nVar          = 1;
    self.sysDim        = mesh.nVert;  # x nVar
    self.leftDof       = mesh.leftVI;
    self.rightDof      = mesh.rightVI;
    self.lowerDof      = mesh.lowerVI;
    self.upperDof      = mesh.upperVI;
    self.leftCoords    = []
    self.rightCoords   = []
    self.lowerCoords   = []
    self.upperCoords   = []
    self.nLeft         = len(mesh.leftVI);
    self.nRight        = len(mesh.rightVI);
    self.nLower        = len(mesh.lowerVI);
    self.nUpper        = len(mesh.upperVI);
    for i in range(self.nLeft):
       vi = int(mesh.leftVI[i]);
       self.leftCoords.append(mesh.vertices[vi]);
    for i in range(self.nRight):
       vi = int(mesh.rightVI[i]);
       self.rightCoords.append(mesh.vertices[vi]);
    for i in range(self.nLower):
       vi = int(mesh.lowerVI[i]);
       self.lowerCoords.append(mesh.vertices[vi]);
    for i in range(self.nUpper):
       vi = int(mesh.upperVI[i]);
       self.upperCoords.append(mesh.vertices[vi]);



  #=========================================================
  # Adds the element matrix to the global matrix
  #=========================================================
  def addElemMat(self, ei, elemMat, gloMat):

    # Retrieve the element vertex indices
    vertInd  = self.mesh.elements[ei];
    nVert    = vertInd.shape[0]

    # Add element matrix to the global matrix
    # In this case nVar=1 is assumed
    for m in range(nVert):
      for n in range(nVert):
        gloMat[vertInd[m],vertInd[n]] += elemMat[m,n];


  #=========================================================
  # Adds the element vector to the global vector
  #=========================================================
  def addElemVec(self, el, elemVec, gloVec):
  
    # Retrieve the node index of element vertices
    vertInd  = self.mesh.elements[el];
    nVert    = vertInd.shape[0]

    # Add element vector to the global vector. 
    # In this case nVar=1 is assumed
    for m in range(nVert):
      gloVec[vertInd[m]] +=  elemVec[m]


  #=========================================================
  # Prints a matrix and vector to two files
  #=========================================================
  def printMatVec(self, mat, vec, basename):
    np.savetxt(basename+"_mat.txt", mat)
    np.savetxt(basename+"_vec.txt", vec)
       
   

  #=========================================================
  # Plots the mesh
  #=========================================================
  def plotMesh(self, plt, title=""):

    plt.set_title("Mesh")
#   plt.set_xlabel('x',size=14,weight='bold')
#   plt.set_ylabel('y',size=14,weight='bold')
    plt.set_aspect('equal');
    plt.set_xlim(-0.1,5.1); plt.set_ylim(-0.1,1.2);
    xy = np.asarray(self.mesh.vertices);
    vals=plt.triplot(xy[:,0],xy[:,1],self.mesh.elements,'b-',linewidth=0.5);
    return vals

  #=========================================================
  # Makes a contour plot of the solution 
  #=========================================================
  def plotSoln(self, solVec, xmS, ymS, xmQ, ymQ, title=""):
    
    cmap = matplotlib.cm.get_cmap('jet')
    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('x',size=14,weight='bold')
    ax.set_ylabel('y',size=14,weight='bold')
    plt.axes().set_aspect('equal', 'datalim')
    xy = np.asarray(self.mesh.vertices);
#    plt.imshow(plt.imread(r'topo_europa.png'), interpolation='nearest', extent=[-0.5,1.495,0,1], alpha=1.0)				
    plt.triplot(xy[:,0],xy[:,1],self.mesh.elements,'b-',linewidth=0.1);
    plt.tricontourf(self.mesh.dtri,solVec, alpha=0.8, cmap=cmap)
    plt.plot(xmS, ymS, 'bs', xmQ, ymQ, 'bs')				
    plt.colorbar()
#    plt.savefig('sol.pdf',dpi=250)
#   plt.show()
    plt.tight_layout()
				
	


  #=========================================================
  # Determines the solution at left boundary y position
  #=========================================================
  def getLeftBndU(self, solVec, xmeas):

        #find closest points
        xy  = np.asarray(self.leftCoords)[:,1]
        d = np.fabs(xy-xmeas)
        vec = np.vstack((xy,d,self.leftDof)).T
        vec = vec[vec[:,1].argsort()]
        y1  = vec[0,0]; y2 = vec[1,0]
        d1  = vec[0,1]; d2 = vec[1,1]
        i1 = int(vec[0,-1]); i2 = int(vec[1,-1])

        # Interpolate
        print ("interpolating between y=",y1," and ",y2)
        umeas = (solVec[i1]*d2+solVec[i2]*d1)/(d1+d2);
        return umeas

  #=========================================================
  # Determines the solution at a right boundary y position
  #=========================================================
  def getRightBndU(self, solVec, xmeas):

        #find closest points
        xy  = np.asarray(self.rightCoords)[:,1]
        d = np.fabs(xy-xmeas)
        vec = np.vstack((xy,d,self.rightDof)).T
        vec = vec[vec[:,1].argsort()]
        y1  = vec[0,0]; y2 = vec[1,0]
        d1  = vec[0,1]; d2 = vec[1,1]
        i1 = int(vec[0,-1]); i2 = int(vec[1,-1])

        # Interpolate
        print ("interpolating between y=",y1," and ",y2)
        umeas = (solVec[i1]*d2+solVec[i2]*d1)/(d1+d2);
        return umeas



  #=========================================================
  # Determines the solution at left boundary y position
  #=========================================================
  def getLeftBndUy(self, solVec, xmeas):

        #find closest points
        xy  = np.asarray(self.leftCoords)[:,1]
        d = np.fabs(xy-xmeas)
        vec = np.vstack((xy,d,self.leftDof)).T
        vec = vec[vec[:,1].argsort()]
        y1  = vec[0,0]; y2 = vec[1,0]
        d1  = vec[0,1]; d2 = vec[1,1]
        i1 = int(vec[0,-1]); i2 = int(vec[1,-1])

        # Interpolate
        print ("differentiating between y=",y1," and ",y2)
        uymeas = (solVec[i2]-solVec[i1])/(y2-y1); 
        return uymeas

			
  #=========================================================
  # Determines the solution at a right boundary y position
  #=========================================================
  def getRightBndUy(self, solVec, xmeas):
			
      #find closest points
      xy  = np.asarray(self.rightCoords)[:,1]
      d = np.fabs(xy-xmeas)		
      vec = np.vstack((xy,d,self.rightDof)).T
      vec = vec[vec[:,1].argsort()]
      y1  = vec[0,0]; y2 = vec[1,0]
      d1  = vec[0,1]; d2 = vec[1,1]
      i1 = int(vec[0,-1]); i2 = int(vec[1,-1])
		
	  # Interpolate
      print ("differentiating between y=",y1," and ",y2)
      uymeas = (solVec[i2]-solVec[i1])/(y2-y1); 
      return uymeas


      				
  #=========================================================
  # Determines the solution at a right boundary y position
  #=========================================================
  def findMinimumPotential(self, solvec):
      xy = np.asarray(self.mesh.vertices);
      vec = np.vstack((xy[:,0], xy[:,1], solvec)).T
      idx = [i for i,v in enumerate(xy[:,1]) if v ==2.0]
      vec = vec[idx,:]
      loc = np.where(vec[:,2] == vec[:,2].min())
      return vec[loc,:2][0][0]	
#**********************************************************************



#================================================================= 
# LinTriElement class definition.
# LinTriElement provides the shape and shape gradient values
# for a linear 2D triangle element defined in physical coordinates.
#================================================================= 
class LinTriElement(object):

  #=========================================================
  # Public data
  #=========================================================
  nFun        = 3      # Number of shape functions in this element.
  vertIndices = None   # Vertex indices
  vertCoords  = None   # Vertex coordinates
  area        = None   # Element area
  nIP         = None   # Number of integration points
  ipCoords    = None   # Coordinates for each integration point
  ipWeights   = None   # Quadrature weight for each ip
  ipScheme    = 4      # Integration scheme

  #=========================================================
  # Object constructor
  #=========================================================
  def __init__(self, mesh, elemIndex): 

    self.vertIndices = mesh.elements[elemIndex]
    v1 = mesh.vertices[self.vertIndices[0]]
    v2 = mesh.vertices[self.vertIndices[1]]
    v3 = mesh.vertices[self.vertIndices[2]]
    self.vertCoords = np.vstack((v1,v2,v3))

    x1 = v1[0]; y1 = v1[1];
    x2 = v2[0]; y2 = v2[1];
    x3 = v3[0]; y3 = v3[1];

    Ae = 0.5*(x2*y3 + x1*y2 + x3*y1 - x3*y2 - x1*y3 - x2*y1);

    self.psi1A = (x2*y3-x3*y2)/(2.*Ae);
    self.psi1B = (y2-y3)/(2.*Ae);
    self.psi1C = (x3-x2)/(2.*Ae);

    self.psi2A = (x3*y1-x1*y3)/(2.*Ae);
    self.psi2B = (y3-y1)/(2.*Ae);
    self.psi2C = (x1-x3)/(2.*Ae);

    self.psi3A = (x1*y2-x2*y1)/(2.*Ae);
    self.psi3B = (y1-y2)/(2.*Ae);
    self.psi3C = (x2-x1)/(2.*Ae);
   
    self.area = Ae; 
    self.setQuadrature()
    
  #=========================================================
  # Set the quadrature points and the weigths
  #=========================================================
  def setQuadrature(self):

    if self.ipScheme == 1:
      # One point scheme
      self.nIP = 1
      w = np.array([1.])
      eta = np.zeros((self.nIP,3))
      eta[0,0] = 1./3.
      eta[0,1] = 1./3.
      eta[0,2] = 1./3.

    elif self.ipScheme == 2:
      # 3 point scheme 1
      self.nIP = 3
      w = np.array([1./3., 1./3., 1./3.])
      eta = np.zeros((self.nIP,3))
      eta[0,0] = 1./2.; eta[0,1] = 1./2.; eta[0,2] = 0.
      eta[1,0] = 1./2.; eta[1,1] = 0.   ; eta[1,2] = 1./2.
      eta[2,0] = 0.   ; eta[2,1] = 1./2.; eta[2,2] = 1./2.

    elif self.ipScheme == 3:
      # 3 point scheme 2
      self.nIP = 3
      w = np.array([1./3., 1./3., 1./3.])
      eta = np.zeros((self.nIP,3))
      eta[0,0] = 2./3.; eta[0,1] = 1./6.; eta[0,2] = 1./6.
      eta[1,0] = 1./6.; eta[1,1] = 2./3.; eta[1,2] = 1./6.
      eta[2,0] = 1./6.; eta[2,1] = 1./6.; eta[2,2] = 2./3.

    else:
      # 4 point scheme 1
      self.nIP = 4
      w = np.array([-27/48., 25./48., 25./48., 25./48.])
      eta = np.zeros((self.nIP,3))
      eta[0,0] = 1./3.;   eta[0,1] = 1./3.;   eta[0,2] = 1./3.
      eta[1,0] = 11./15.; eta[1,1] = 2./15.;  eta[1,2] = 2./15.
      eta[2,0] = 2./15.;  eta[2,1] = 11./15.; eta[2,2] = 2./15.
      eta[3,0] = 2./15.;  eta[3,1] = 2./15.;  eta[3,2] = 11./15.


    # Set the weights and transform to physical coordinates
    self.ipWeights = w * self.area
    self.ipCoords=np.zeros((self.nIP,2))
    for ip in range(self.nIP):
      for d in range(2):
         self.ipCoords[ip,d] = ( eta[ip,0]*self.vertCoords[0,d] 
                                +eta[ip,1]*self.vertCoords[1,d] 
                                +eta[ip,2]*self.vertCoords[2,d])
 

  #=========================================================
  # Returns the three shape functions at the point (x,y)
  #=========================================================
  def getShapes(self, x=None, y=None):

    psi    = np.zeros(3);
    psi[0] = self.psi1A + self.psi1B*x + self.psi1C*y;
    psi[1] = self.psi2A + self.psi2B*x + self.psi2C*y;
    psi[2] = self.psi3A + self.psi3B*x + self.psi3C*y;

    return psi


  #=========================================================
  # Returns the gradients of the three shape functions 
  # a the point (x,y). gradPsi[i] points to a vector 
  # with the x and y gradients of shape function i
  #=========================================================
  def getShapeGradients(self, x=None, y=None):

    gradPsi = np.zeros((3,2));
    gradPsi[0,0] = self.psi1B; gradPsi[0,1] = self.psi1C;
    gradPsi[1,0] = self.psi2B; gradPsi[1,1] = self.psi2C;
    gradPsi[2,0] = self.psi3B; gradPsi[2,1] = self.psi3C;

    return gradPsi

#=========================================================
# Returns the height of the ice-water interface
#=========================================================
def iceWat(x, a, b):
    yIceWat = 1.8 -a*(1.-math.cos(math.pi*(b-x)/20.))
    return yIceWat 