
import numpy as np

## matrix operations
def dag(A):
  return 1.*np.conjugate(np.transpose(A))

def dot(A,B):
  return np.trace(dag(A)@B)

def norm(A):
  return np.sqrt(np.abs(dot(A,A)))

def kprod(A,B):
  return np.kron(A,B)

def ksum(A,B):
  return np.kron(A,one) + np.kron(one,B)

def eig(A):
  vals, vecs = np.linalg.eigh(A)
  vecs = np.transpose(vecs) ## so vecs[0] is an eigenvector
  return 1.*vals, 1.*vecs

def couter(psi):
  return 1.*np.outer(psi, np.conjugate(psi))


## entropies
def H(x):
  """ H(x) = - x log2 x """
  HH = np.zeros_like(x)
  HH[x >0.] = -x[x>0.]*np.log2(x[x>0.])
  HH[x==0.] = 0.
  HH[x <0.] = np.nan
  return 1.*HH

def SVN(rho, decimals=12):
  vals, vecs = eig(rho)
  vals = np.round(vals, decimals) ## avoid tiny negative results
  return np.sum(H(vals))


