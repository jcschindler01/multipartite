
"""
Generate two-qubit density matrices from Bloch vectors and calculate various entropies.

Use the Hilbert-Schmidt inner product (A,B) = Tr(A^{+} B) on the space of complex matrices.

The Pauli matrices

sigma_x = ( 0, 1)
          ( 1, 0)

sigma_y = ( 0, -i)
          ( i,  0)

sigma_z = ( 1,  0)
          ( 0, -1)

form an orthogonal basis for the space of traceless Hermitian matrices (normalize with factor of sqrt(2)).

The method is based on my "density_matrix_geometry.pdf", but differs in one significant way:
yAB is OUTSIDE the parentheses here, where it is INSIDE in eqn 3 of the doc.
"""

import numpy as np
from multipartite.mptools import *

## constants
d, dA, dB = 4., 2., 2.
alpha, beta = (1./np.sqrt(6.)), (1./np.sqrt(6.))

## sigma matrix basis
one  = np.eye(2)
ones = np.ones((2,2))
s = (1./np.sqrt(2.))*np.array([
                               [[1., 0.],[0.,-1.]],
                               [[0., 1.],[1., 0.]],
                               [[0., -1j],[1j,0.]],
                              ], dtype=complex)
ONE = np.eye(4)



## check for valid Bloch vectors
def isvalid_c(cA=np.zeros(3), cB=np.zeros(3), cAB=np.zeros((3,3))):
  #print("\nisvalid_c")
  valid = True
  zA, zB, zAB = np.sum(cA**2), np.sum(cB**2), np.sum(cAB**2)
  z0 = alpha**2*dB*np.sum(cA**2)+beta**2*dA*np.sum(cB**2)+np.sum(cAB**2)
  if not (zA<=1. and zB<=1. and zAB<=1. and z0<=1.):
    valid = False
  if valid==False:
    pass
    #print(1*"BAD COEFFS cA cB cAB\n")
  #print("valid = %s"%(valid))
  #print("   zA,    zB,   zAB,    z0   (<=? 1.)")
  #print("%.3f, %.3f, %.3f, %.3f"%(zA, zB, zAB, z0))
  #print()
  return bool(1*valid)

## check for valid density matrix
def isvalid_rho(rho, tol=1e-12):
  #print("\nisvalid_rho")
  valid = True
  herm = np.abs(rho-dag(rho))
  ztr  = np.abs(np.trace(rho)-1.)
  vals = np.linalg.eigvalsh(rho)
  if not np.all(herm)<tol:
    valid = False
  if not ztr<tol:
    valid = False
  if not np.all(vals>=-tol):
    valid = False
  if valid==False:
    ##print(1*"BAD RHO\n")
    rho = 0. * rho
  #print("valid = %s"%(valid))
  #print("max(|rho - dag(rho)|) = %.3f    <? tol"%(np.max(herm)))
  #print("    |tr(rho) - 1|     = %.3f    <? tol"%(ztr))
  #print("eigenvals >? -tol")
  #print(np.round(vals,3))
  #print()
  return 1.*rho, bool(1*valid)

## check for valid projectors
def isvalid_proj(proj, tol=1e-12):
  valid = True
  n = len(proj[0])
  if not np.all(np.abs(np.sum(proj, axis=0)-np.eye(n))<tol):
    valid = False
    print("\nproj sum not identity")
    print(np.round(p,6))
  for p in proj:
    if not np.all(np.abs(p-dag(p))<tol):
      valid = False
      print("\nproj not hermitian")
      print(np.round(p,6))
    if not np.all(np.abs(p@p - p)<tol):
      valid = False
      print("\nproj not projector")
      print(np.round(p,6))
  for i in range(len(proj)):
    for j in range(i+1,len(proj)):
      if not np.all(np.abs(proj[i]@proj[j])<tol):
        valid = False
        print("\nproj not orthogonal")
        print(np.round(proj[i]@proj[j],6))
  return valid

## create a "sigma matrix" representation of n x n hermitian traceless matrices
## number of matrices is n^2-1 aka 3 for 2, 8 for 3, 15 for 4
def SIGMA(n=2):
  ss = []
  for i in range(n):
    for j in range(n):
      s = np.zeros((n,n), dtype=complex)
      if i==j and i<n-1:
        s[:i+1,:j+1] = np.eye(i+1)
        s[ i+1, j+1] = -(i+1.)
        s = s/np.sqrt((i+2.)*(i+1.))
        ss += [1.*s]
      if i>j:
        s[i,j] = 1.
        s[j,i] = 1.
        s = s/np.sqrt(2.)
        ss += [1.*s]
      if i<j:
        s[i,j] = -1.j
        s[j,i] = 1.j
        s = s/np.sqrt(2.)
        ss += [1.*s]
  return 1.*np.array(ss, dtype=complex)


## traceless hermitian matrix (x = sum_i c_i sigma_i) from real Bloch vector c with (|c|^2 = |x|^2).
def X(c=np.zeros(3)):
  return 1.*np.array(np.sum([c[i]*s[i] for i in range(len(s))], axis=0))

## joint traceless hermitian matrix xAB from joint Bloch vector cAB
def XAB(cAB=np.zeros((3,3))):
  return 1.*np.sum([cAB[i,j]*kprod(s[i],s[j]) for i in range(3) for j in range(3)], axis=0)

## density matrix from Bloch coeffs
def RHO(cA=np.zeros(3), cB=np.zeros(3), cAB=np.zeros((3,3)), product=False):
  #print("\nRHO")
  #print("cA")
  #print(repr(np.round(cA ,3)))
  #print("cB")
  #print(repr(np.round(cB ,3)))
  #print("cAB")
  if product==True:
    cAB = 0.5*np.outer(cA,cB)
  #print(repr(np.round(cAB,3)))
  valid_c = isvalid_c(cA,cB,cAB)
  a = ONE/d
  b = np.sqrt((d-1.)/d)*ksum(alpha*X(cA),beta*X(cB))
  c = XAB(cAB)
  rho = a + b + c
  #print("rho")
  #print(repr(np.round(rho,3)))
  rho, valid_rho = isvalid_rho(rho)
  return 1.*rho



## Bloch coeffs from rho
def CAB_RHO(rho):
  rhoA, rhoB = RHOA(rho), RHOB(rho)
  xA = np.sqrt(dA/(dA-1.)) * (rhoA - one/dA)
  xB = np.sqrt(dB/(dB-1.)) * (rhoB - one/dB)
  cA = np.array([np.real(dot(s[i],xA)) for i in range(len(s))])
  cB = np.array([np.real(dot(s[i],xB)) for i in range(len(s))])
  xAB = (rho - ONE/d)
  yAB = xAB - ksum(alpha*xA, beta*xB)
  cAB = np.nan * np.ones((3,3))
  for i in range(len(s)):
    for j in range(len(s)):
      cAB[i,j] = np.real(dot( kprod(s[i],s[j]), yAB))
  return 1.*cA, 1.*cB, 1.*cAB


## arbitrary 2d state vector
def PSI(theta=0., phi=0.):
  return np.array([np.cos(theta), np.exp(-1j*phi)*np.sin(theta)], dtype=complex)

