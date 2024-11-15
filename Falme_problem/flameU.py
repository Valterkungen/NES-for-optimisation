#================================================================= 
#
# AE2220-II: Computational Modelling 
# Code for work session 1 - In-class exercise
#
#=================================================================
# This code provides a base for computing the advection 
# reaction equation for Z on a rectangular domain
#
# lines 25-32:   Input parameters 
# lines 92-116:  Implementation of finite-difference scheme.
#                This is based on an explicit space march in y.
#                ****** indicates where information is to be added.
# 
#=================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from flameLib import getPsi,getUV,getF,getPerf

#=========================================================
# Input parameters
#=========================================================
nozc       = 2.0;                # Nozzle centre
nozw       = 1.0;  
refine = 3              # Nozzle width
imax       = 40*refine;                 # Number of mesh points in i
nmax       = 80*refine**2;                 # Number of mesh points in j

maxl       = 50;                 # maximum grid lines on plots
stride     = 4;                  # Point skip rate for suface plot


#=========================================================
# Load the mesh coordinates
#=========================================================
x1     = -2.5;                    # Forward boundary position
x2     =  2.5;                    # Rear boundary position
y1     =  0. ;                    # lower boundary position
y2     = 10.0;                    # Upper boundary position
dx     = (x2-x1)/(float(imax-1)); # Mesh spacing in x
dy     = (y2-y1)/(float(nmax-1)); # Mesh spacing in y
x      = np.zeros((imax,nmax));   # Mesh x coordinates
y      = np.zeros((imax,nmax));   # Mesh y coordinates

# Mesh coordinates
for n in range(0, nmax):
  x[:,n] = np.linspace(x1, x2, imax)

for i in range(0, imax):
  y[i,:] = np.linspace(y1, y2, nmax)


#=========================================================
# Load the stream function and velocities 
#=========================================================
psi=np.zeros((imax,nmax));
u  =np.zeros((imax,nmax));
v  =np.zeros((imax,nmax));
crn=np.zeros((imax,nmax));

for n in range(0, nmax):
  for i in range(0, imax):
    psi[i,n]      = getPsi(x[i,n],y[i,n])
    u[i,n],v[i,n] = getUV (x[i,n],y[i,n])
    crn[i,n]      = u[i,n]*dy/(v[i,n]*dx)
    
#=========================================================
# Initialise the mass fraction distribution to zero
# then set the inflow condition
#=========================================================
Z = np.zeros((imax,nmax));  
F = np.zeros((imax,nmax));

xnozLeft =nozc-0.5*nozw;
xnozRight=nozc+0.5*nozw;
mfar     = 0.0075/nozw;  # max fuel/air ratio

for i in range(0, imax):
  if (x[i,0] >xnozLeft) & (x[i,0] <xnozRight) :
    dist = x[i,0]-nozc;
    Z[i,0] = mfar*0.5*(1.0+math.cos( 6.2832*(dist/nozw-1.0) ))

#=========================================================
# Since v is always positive, we can march explicitly in y, 
# solving for the local value of the mass fraction, Z.
#=========================================================

for n in range(0, nmax-1):   # March upwards from y=0 to y=10
 
   # Compute F
   for i in range(0, imax):
     F[i,n] = getF(u[i,n],v[i,n],Z[i,n])
 
   # Update left boundary (node i = 0)  ******
   # Since u<0 on the left boundary a (first-order) numerical condition
   # should be implemented. Finish the line and uncomment it
   Z[0,n+1] = Z[0,n] - u[0,n]*dy*(Z[1,n]-Z[0,n])/(v[0,n]*dx) + F[i,n]*dy/v[i,n]

  
   # Update interior (nodes i=1 to imax-2) 
   # Here we use a first-order upwind approximation for dZ/dx 
   # dependent on the local advection velocity (note v is always positive) 
   for i in range(1, imax-1):
     if u[i,n] < 0.:
         Z[i,n+1] = Z[i,n] - (Z[i+1,n]-Z[i,n])*dy*u[i,n]/(v[i,n]*dx) + F[i,n]*dy/v[i,n]
     else: 
         Z[i,n+1] = Z[i,n] - (Z[i,n]-Z[i-1,n])*dy*u[i,n]/(v[i,n]*dx) + F[i,n]*dy/v[i,n]


   # Update right boundary (node i = imax-1)  ******
   # Since u>0 on the right boundary a (first-order) numerical condition
   # should be implemented. Finish the line and uncomment it.
   Z[imax-1,n+1] =  Z[imax-1,n] - u[imax-1,n]*dy*(Z[imax-1,n]-Z[imax-2,n])/(v[imax-1,n]*dx) + F[i,n]*dy/v[i,n]
                



#=========================================================
# Evaluate performance and print summary
#=========================================================
perf = getPerf(imax,nmax,x,y,dx,dy,F)
print('------------------------------------------')
print('Summary:')
print('------------------------------------------')
print('imax x nmax       = ',imax,' x',nmax)
print('Flame performance = ',perf)
print('------------------------------------------')


#=========================================================
#  Plot results
#=========================================================
fig = plt.figure(figsize=(18,8))

ax1 = plt.subplot2grid((2,5), (0,0), colspan=3, rowspan=2, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Z')
ax1.plot_surface(x, y, Z, shade=True, rstride=stride,cstride=stride,
cmap=plt.cm.gnuplot, linewidth=0, antialiased=True);

ax2 = plt.subplot2grid((2,5), (0,3),rowspan=1, aspect=0.75)
ax2.set_title(r"Streamlines")
a = ax2.contour(x, y, psi, 20)

ax3 = plt.subplot2grid((2,5), (1,3),rowspan=1)
ax3.set_title(r"$u\Delta y / v\Delta x$")
a = ax3.contourf(x,y,crn,10, cmap=plt.cm.jet)
if (imax<maxl) & (nmax<maxl):
 ax3.plot(x,y, '-k', x.transpose(), y.transpose(), '-k')
fig.colorbar(a, ax=ax3)

ax4 = plt.subplot2grid((2,5), (0,4),rowspan=1)
ax4.set_title(r"$Z$ - Mass fraction")
a = ax4.contourf(x,y, Z, cmap=plt.cm.gnuplot)
if (imax<maxl) & (nmax<maxl):
 ax4.plot(x,y, '-k', x.transpose(), y.transpose(), '-k')
fig.colorbar(a, ax=ax4)

ax5 = plt.subplot2grid((2,5), (1,4),rowspan=1)
ax5.set_title(r"f - Flame Intensity")
a = ax5.contourf(x, y, -1000*F, cmap=plt.cm.hot )
if (imax<maxl) & (nmax<maxl):
  ax5.plot(x, y, '-k', x.transpose(), y.transpose(), '-k')
fig.colorbar(a, ax=ax5)

ax1.view_init(30, -120)
#plt.savefig('flame.png',dpi=250)
plt.show()

print ('done')
