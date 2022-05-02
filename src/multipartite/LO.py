
import numpy as np
from multipartite.constants import *
from multipartite.matrix import *
from multipartite.validity import *

def isVALID(rho,A,B):
  if not isDENSITY(rho):
    print("RHO is not DENSITY")
    return False
  elif not isPOVM(A):
    print("A is not POVM")
    return False
  elif not isPOVM(B):
    print("B is not POVM")
    return False
  else:
    return True

def S_OBS_RDE(rho):
  ## initalize inputs
  rho, dA, dB = rho.astype(complex), 2., 2.
  ## use bloch vector method to extract reduced densities
  rhoA = (1./dA)*np.sum([np.trace(rho @ kprod(sigma[i],sigma[0]))*sigma[i] for i in range(len(sigma))], axis=0)
  rhoB = (1./dB)*np.sum([np.trace(rho @ kprod(sigma[0],sigma[i]))*sigma[i] for i in range(len(sigma))], axis=0)
  ## get eigenvectors
  valsA, vecsA = eig(rhoA)
  valsB, vecsB = eig(rhoB)
  ## create POVMs
  A = [couter(vecsA[i]) for i in range(len(valsA))]
  B = [couter(vecsB[i]) for i in range(len(valsB))]
  ## run S_OBS_LO
  return S_OBS_LO(rho, A, B, RDE_out=False).copy()


def S_OBS_LO(rho=0.5*np.eye(2), A=[np.eye(2)], B=[np.eye(2)], RDE_out=True, decimals=15):
  """
  Observational entropy of local POVM measurement.
  Recieves a density and two local POVMs.
  Returns a dictionary with all the info you could want.
  """
  ## process inputs
  rho = rho.astype(complex)
  A, B = np.stack([A[i].astype(complex) for i in range(len(A))]), np.stack([B[i].astype(complex) for i in range(len(B))])
  dA, dB = len(A[0]), len(B[0])
  d = dA*dB
  ## initialize output
  out = dict()
  ## check validity
  if not isVALID(rho,A,B):
    return None
  ## initialize probabilities and volumes
  pxy = np.nan * np.ones((len(A),len(B))).astype(complex)
  px = np.nan * np.ones(len(A))
  py = np.nan * np.ones(len(B))
  Vxy = np.nan * np.ones((len(A),len(B))).astype(complex)
  Vx = np.nan * np.ones((len(A)))
  Vy = np.nan * np.ones((len(B)))
  ## calculate joint probabilities and volumes
  for x in range(len(A)):
    for y in range(len(B)):
      pxy[x,y]=dot(rho, kprod(A[x],B[y]))
      Vxy[x,y]=np.trace(kprod(A[x],B[y]))
  ## round to avoid tiny negative probabilities
  pxy = np.round(pxy, decimals)
  Vxy = np.round(Vxy, decimals)
  ## ensure real and discard imaginary parts
  if not (np.all(np.abs(pxy.imag)<1e-15) and np.all(np.abs(Vxy.imag)<1e-15)):
    return None
  else:
    pxy = 1.*pxy.real
    Vxy = 1.*Vxy.real
  ## ensure positive
  if not (np.all(pxy>=0.) and np.all(Vxy>=0.)):
    return None
  ## calculate marginal probabilities and volumes
  for x in range(len(px)):
    px[x] = np.sum(pxy[x,:])
    Vx[x] = np.trace(A[x]).real
  for y in range(len(py)):
    py[y] = np.sum(pxy[:,y])
    Vy[y] = np.trace(B[y]).real
  ## calculate product marginal distribution
  pxpy = np.nan * pxy
  for x in range(len(A)):
    for y in range(len(B)):
      pxpy[x,y]= px[x]*py[y]
  ## bloch vectors
  ## defined by rho = (1/d) sum r_{mn} sigma_m o sigma_n for m,n in [0,1,2,3]
  ## tr(rho (sigma_m o sigma_n)) = r_{mn}
  r = np.nan * np.ones((4,4))
  for alpha in range(len(sigma)):
    for beta in range(len(sigma)):
      r[alpha,beta] = np.trace(rho @ kprod(sigma[alpha],sigma[beta])).real
  r = 1.*r
  rA = r[1:,0]
  rB = r[0,1:]
  rAB = r[1:,1:]
  ## POVM bloch vectors
  ax = np.nan * np.ones((len(A),3))
  for x in range(len(A)):
    for i in [1,2,3]:
      ax[x,i-1] = np.trace(A[x] @ sigma[i]).real
  by = np.nan * np.ones((len(B),3))
  for y in range(len(B)):
    for i in [1,2,3]:
      by[y,i-1] = np.trace(B[y] @ sigma[i]).real
  ## reduced densities
  rhoA = (1./dA)*np.sum([r[i,0]*sigma[i] for i in range(len(sigma))], axis=0)
  rhoB = (1./dB)*np.sum([r[0,i]*sigma[i] for i in range(len(sigma))], axis=0)
  ## inputs
  out['rho'] = 1.*rho
  out['A'] = 1.*A
  out['B'] = 1.*B
  ## probabilities and volumes
  out['pxy'] = 1.*pxy
  out['px'] = 1.*px
  out['py'] = 1.*py
  out['Vxy'] = 1.*Vxy
  out['Vx'] = 1.*Vx
  out['Vy'] = 1.*Vy
  ## observational entropies and mutual info
  out['S_AB_RHO'] = np.sum(Vxy*H(pxy/Vxy))
  out['S_A_RHOA'] = np.sum(Vx *H(px /Vx ))
  out['S_B_RHOB'] = np.sum(Vy *H(py /Vy ))
  out['I_cl']     = np.sum(-pxpy[pxy!=0]*H(pxy[pxy!=0]/pxpy[pxy!=0]))
  ## bloch vectors
  out['r'] = 1.*r
  out['rA'] = 1.*rA
  out['rB'] = 1.*rB
  out['rAB'] = 1.*rAB
  ## reduced states
  out['rhoA'] = 1.*rhoA
  out['rhoB'] = 1.*rhoB
  ## quantum entropies and mutual info
  out['S_RHO']  = SVN(rho)
  out['S_RHOA'] = SVN(rhoA)
  out['S_RHOB'] = SVN(rhoB)
  out['I_qm'] = out['S_RHOA'] + out['S_RHOB'] - out['S_RHO']
  ## POVM bloch vectors
  out['POVM_BLOCH_A'] = 1.*ax
  out['POVM_BLOCH_B'] = 1.*by
  ## reduced density eigenbasis
  if RDE_out==True:
    out['RDE_out'] = S_OBS_RDE(rho).copy()
  ## return
  return out.copy()
  

