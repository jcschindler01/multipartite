
"""Defines tools in general dimension d=d."""

import numpy as np
from multipartite.tools import constants as const

## useful operations

def dag(A):
  return 1.*np.conjugate(np.transpose(A))

def eigh(A):
  vals, vecs = np.linalg.eigh(A)
  vecs = np.transpose(vecs) ## so vecs[0] is an eigenvector
  return 1.*vals, 1.*vecs

def H(x):
  """ H(x) = - x log2 x """
  HH = np.zeros_like(x)
  HH[x >0.] = -x[x>0.]*np.log2(x[x>0.])
  HH[x==0.] = 0.
  HH[x <0.] = np.nan
  return 1.*HH

def SVN(rho, decimals=15):
  vals, vecs = eigh(rho)
  vals = np.round(vals, decimals) ## avoid tiny negative results
  return np.sum(H(vals))



## matrix class
class matrix:
  
  """Matrix in dxd dimensions."""
  
  def __init__(self, M=np.zeros((2,2)), tol=1e-15):
    ## all matrix attributes
    self.M = 1.*M.astype(complex) ## matrix values
    self.d = None ## dimension
    self.t = None ## trace
    ## Hermitian matrix attributes
    self.vals, self.vecs = None, None ## eigendecomp
    ## d=2 hermitian matrix attributes
    self.bloch2 = None ## 2x2 bloch vector (t,x,y,z)
    ## density matrix attributes
    self.SVN = None
    ## tolerance for equalities
    self.tol = 1.*tol
    ## update from M
    self.update()

  def isHermitian(self):
    eps = np.abs(self.M - dag(self.M))
    if np.all(eps < self.tol):
      return True
    else:
      return False

  def isPositive(self):
    if self.isHermitian():
      if np.all(self.vals >= 0.):
        return True
    return False

  def isDensity(self):
    eps = np.abs(self.t - 1.)
    if self.isHermitian() and self.isPositive() and (eps < self.tol):
      return True
    return False

  def isTNI(self):
    ## trace-non-increasing if M positive and 1-M also positive
    if self.isHermitian() and self.isPositive():
      if np.all(self.vals <= 1.):
        return True
    return False

  def isPOVMe(self):
    if self.isHermitian() and self.isPositive() and self.isTNI():
      return True
    return False

  def getBloch2(self):
    ## Bloch vector defined by A = (1/d) sum_k c_k sigma_k for k=0..3
    ## tr(sigma_i sigma_j) = d delta_ij
    ## c_k = tr(M sigma_k)
    txyz = np.array([np.trace(self.M @ const.sigma[k]) for k in range(4)], dtype=complex)
    return 1.*np.abs(txyz)

  def setBloch2(self, txyz=np.zeros(4)):
    self.M = 0.5 * np.sum([txyz[k]*const.sigma[k] for k in range(4)], axis=0).astype(complex)
    self.update()

  ## update based on current M
  def update(self):
    ## clear all
    self.d, self.t, self.vals, self.vecs, self.bloch2, self.SVN = None, None, None, None, None, None
    ## all matrix attributes
    self.d = len(self.M) ## dimension
    self.t = np.trace(self.M) ## trace
    ## Hermitian matrix attributes
    if self.isHermitian():
      ## all hermitian matrix attributes
      self.t = np.abs(self.t) ## real trace
      self.vals, self.vecs = eigh(self.M) ## eigendecomp
      ## d=2 hermitian matrix attributes
      if self.d==2:
        self.bloch2 = self.getBloch2() ## 2x2 bloch vector (t,x,y,z)
    ## density matrix properties
    if self.isDensity():
      self.SVN = SVN(self.M)

  def report(self):
    print("matrix.report")
    print()
    for key in vars(self):
      print("%s = "%key)
      print(vars(self)[key])
      print()
    print("isHermitian = %s"%self.isHermitian())
    print("isPositive  = %s"%self.isPositive())
    print("isTNI       = %s"%self.isTNI())
    print()
    print("isDensity   = %s"%self.isDensity())
    print("isPOVMe     = %s"%self.isPOVMe())
    print()



M = np.random.random((2,2)) + 1j*np.random.random((2,2))

M = M + dag(M)

M = 0.5*np.eye(2) + .25*const.X

m = matrix(M)

c = np.array([2,0,0,0])

m.setBloch2(c)

m.report()

