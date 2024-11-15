#================================================================= 
#
# AE2220-II: Computational Modelling 
# Additional Code for work session 1
#
#=================================================================
# This file includes some additional functions to assist 
# with the problem definition. 
# 
# There is no need to inspect or edit this file to complete
# the work session
#
# For the interested, the flow speeds in the chamber are assumed 
# to be low enough so that the velocity is mostly divergence free. 
# In this case the energy equation (describing e.g. temperature)
# becomes decoupled from the momentum equations. 
# It is still mixed hyperbolic-elliptic, however, so solving
# for temperature to determine the reaction rate would in principle 
# require at least one more PDE in our system, as well as giving up 
# the marching procedure. This goes beyond the level of complexity 
# appropriate for a work session, so we approximate the net effect on the 
# source term in the mass fraction equation using an empirical model.
# 
#=================================================================
import math


#=================================================================
# Global stream function data
#=================================================================
k0=3;
k1= 2.0; a1=-1.; b1=16.;
k2=-4.0; a2=-6.; b2=-3.;
k3=-3.0; a3=-6.; b3= 4.;
k4=-4.0; a4=-6.; b4=10.;
k5=-6.0; a5=+6.; b5=-1.;
k6=-6.0; a6=+8.; b6=+11.;

#=================================================================
# Returns the stream function at x,y
#=================================================================
def getPsi(x,y):
  dx1=x-a1; dy1=y-b1; dx2=x-a2; dy2=y-b2; 
  dx3=x-a3; dy3=y-b3; dx4=x-a4; dy4=y-b4; 
  dx5=x-a5; dy5=y-b5; dx6=x-a6; dy6=y-b6; 
  psi = -2.0*x + k1*math.atan2(dy1,dx1) + k2*math.atan2(dy2,dx2) \
               + k3*math.atan2(dy3,dx3) + k4*math.atan2(dy4,dx4) \
               + k5*math.atan2(dy5,dx5) + k6*math.atan2(dy6,dx6)
  return psi


#=================================================================
# Returns velocities at x,y
# Depending on the intensity of the velocity field, it 
# may be necessary to adjust k to avoid -ve Z.
#=================================================================
def getUV(x,y):

  dx1=x-a1; dy1=y-b1; dx2=x-a2; dy2=y-b2; 
  dx3=x-a3; dy3=y-b3; dx4=x-a4; dy4=y-b4; 
  dx5=x-a5; dy5=y-b5; dx6=x-a6; dy6=y-b6; 

  u = + (k1*dx1)/(dx1*dx1+dy1*dy1) \
      + (k2*dx2)/(dx2*dx2+dy2*dy2) \
      + (k3*dx3)/(dx3*dx3+dy3*dy3) \
      + (k4*dx4)/(dx4*dx4+dy4*dy4) \
      + (k5*dx5)/(dx5*dx5+dy5*dy5) \
      + (k6*dx6)/(dx6*dx6+dy6*dy6) 

  v = + (k1*dy1)/(dx1*dx1+dy1*dy1) \
      + (k2*dy2)/(dx2*dx2+dy2*dy2) \
      + (k3*dy3)/(dx3*dx3+dy3*dy3) \
      + (k4*dy4)/(dx4*dx4+dy4*dy4) \
      + (k5*dy5)/(dx5*dx5+dy5*dy5) \
      + (k6*dy6)/(dx6*dx6+dy6*dy6)  + k0;

  return (u,v)


#=================================================================
# Returns the reaction term based on 
# the local flow state and mass fraction
#=================================================================
def getF(u,v,Z):
  magv=math.sqrt(u*u+v*v)
  vlim=3.2;
  FVal = 0.
  if ((magv<vlim) and (Z>0.)):
     vfuns = math.sin(3.14159*magv/vlim);
     vfun = 0.67*math.atan(vfuns*vfuns/0.05)
     FVal = - (vlim-magv)*vfun*0.5*Z;
  return min(0.,FVal)


#=================================================================
# Evaluates the performance of the flame
#=================================================================
def getPerf(imax,nmax,x,y,dx,dy,F):
  perf = 0;
  for n in range(0, nmax-1):
    for i in range(0, imax-1):
       floc=-1000*0.25*(F[i,n]+F[i+1,n]+F[i,n+1]+F[i+1,n+1]);
       xloc=0.25*(x[i,n]+x[i+1,n]+x[i,n+1]+x[i+1,n+1])+1.;
       perf += dx*dy*math.exp(-10.0*(floc-1.0)*(floc-1.0))*math.exp(-10.0*xloc*xloc);

  return perf
