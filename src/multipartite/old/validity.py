
import numpy as np

dag = lambda x: np.conjugate(np.transpose(x))

def isPOSITIVE(M, tol=1e-15):
  vals, vecs = np.linalg.eigh(M)
  if not np.all(vals >= 0.):
    print("M is not POSITIVE")
    print(np.round(M,2))
    print(vals)
    print("REPEAT: M is not POSITIVE")
    return False
  else:
    return True

def isHERMITIAN(M, tol=1e-15):
  z = M - dag(M)
  if np.trace(dag(z)@z) < tol:
    return True
  else:
    return False

def isDENSITY(M, tol=1e-15):
  if not isHERMITIAN(M, tol=tol):
    return False
  elif not isPOSITIVE(M):
    return False
  elif not np.abs(np.trace(M)-1.) < tol:
    return False
  else:
    return True

def isPOVM(MM, tol=1e-15):
  for M in MM:
    if not isHERMITIAN(M, tol):
      return False
    if not isPOSITIVE(M):
      return False
    z = np.sum(MM, axis=0)-np.eye(len(MM[0]))
    if not np.abs(np.trace(dag(z)@z)) < tol:
      return False
    else:
      return True
