"""-----------------
Analysis of local observational entropies in the state 
rho = (1/7) * (3 |00><00| + 4 |1+><1+|).
-----------------
"""

## imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import multipartite as mp

## set up output log
np.set_printoptions(linewidth=200)
if True:
  sys.stdout = open('LOex.out.txt','w')

## print output tag
print("--------------------------------------")
print("LOex.out.txt")
print("Output generated by LOex.py.")
print()
print("multipartite.__version__ = %s"%(mp.__version__))
print("--------------------------------------")
print()

## print docstring
print(__doc__)

## dimension
n = [2,2]

## rho
psi1 = np.array([1.,0.,0.,0.])
rho1 = mp.RHO_PSI(psi1)

print("rho1 =")
print(rho1)
print()

## rho
psi2 = np.array([0.,0.,1.,1.])
rho2 = mp.RHO_PSI(psi2)

print("rho2 =")
print(rho2)
print()

## rho
rho = (1./7.)*(3.*rho1 + 4.*rho2)

print("rho = (1/7)*(3*rho1+4*rho2)")
print()
print("7 * rho =")
print(7. * rho)
print()

## eig
print("The eigenvalues and eigenvectors are exactly known by definition. Check them.")
print()

vals, vecs = mp.eig(rho)

print("7 * vals =")
print(7. * vals)
print()

print("vecs = ")
[print(np.round(vecs[i],3)) for i in range(len(vecs))]
print()


## reduced densities
print(
"""
The reduced densities should be exactly

rhoA = (3/7)*|0><0| + (4/7)*|1><1| = (1/7)*diag(3,4)

rhoB = (3/7)*|0><0| + (4/7)*|+><+|
     = (1/7)* (5 2, 2 2)

Check them.
"""
)

rhoA, rhoB = mp.REDUCE(rho,n)

print("7 * rhoA =")
print(7. * rhoA)
print()

print("7 * rhoB =")
print(7. * rhoB)
print()

print("RhoA is already diagonal in computational basis.")
print()

print("Eigvals of rhoB are exactly 6/7, 1/7")
print("with vecs (2,1), (-1,2). Check them.")
print()

Bvals, Bvecs = mp.eig(rhoB)

print("7 * Bvals =")
print(7. * Bvals)
print()

print("-sqrt(5) * Bvecs =")
[print(-np.sqrt(5.) * Bvecs[i]) for i in range(len(Bvecs))]
print()

print(mp.isvalid_rho(rho)[1])












