"""
Let
rho = (1/d) (1 + x)
be a dxd density matrix.

Then x is in su(d), the (d^2-1) dimensional space of traceless Hermitian matrices.

Introduce a basis sigma_i for su(d), orthonormalized by tr(sigma_i sigma_j) = d delta_{ij}.
One such basis is the Pauli matrices.

Thus x = sum_i c_i sigma_i and we have coefficients c_i = tr(rho sigma_i).

Further, denote |c|^2 = sum_i |c_i|^2.
Then tr(x^2) = d |c|^2. And tr(rho^2) <= 1 implies |c|^2 <= 1.

Now consider HA o HB.

Let sigmaA_m and sigmaB_n be ON bases for su(dA) and su(dB) as above.
Then tensor products of (1,sigmaA_m) with (1,sigmaB_n) form an ON basis for Hermitian operators on HA o HB.

Thus we can write
rhoAB = (1/d) ( 1 + xA o 1B + 1A o xB + sum_{mn} c_{mn} sigmaA_m o sigmaB_n  )

with
rhoA = (1/dA) (1 + xA)
rhoB = (1/dB) (1 + xB).

In this way we obtain two marginal Bloch vectors cA_m, cB_n, and a mutual Bloch vector cAB_{mn}.
"""

import numpy as np

## create a "sigma matrix" representation of n x n hermitian traceless matrices
## number of matrices is n^2-1 aka 3 for 2, 8 for 3, 15 for 4
def SIGMA(n=2):
  ## Currently normalized wrong except for n=2 case!!!
  ## Ordered as X,Y,Z for n=2
  ss = []
  for i in range(n):
    for j in range(n):
      s = np.zeros((n,n), dtype=complex)
      if i==j and i<n-1:
        s[:i+1,:j+1] = np.eye(i+1)
        s[ i+1, j+1] = -(i+1.)
        s = s
        ss += [1.*s]
      if i>j:
        s[i,j] = 1.
        s[j,i] = 1.
        s = s
        ss += [1.*s]
      if i<j:
        s[i,j] = -1.j
        s[j,i] = 1.j
        s = s
        ss += [1.*s]
  return 1.*np.array(ss[::-1], dtype=complex)

sig = SIGMA(n=2)

## Get rho from bloch vector.
## Careful -  no norm check
def rho_from_bloch(c=[0.,0.,0.], d=2):
  s = SIGMA(d)
  c = np.array(c).astype(float)
  x = np.sum([c[i]*s[i] for i in range(len(s))], axis=0)
  rho = (1./float(d)) * (np.eye(d) + x)
  return rho

## Get bloch from rho.
## Careful -  no norm check
def bloch_from_rho(rho=0.5*np.eye(2)):
  d = len(rho)
  s = SIGMA(d)
  c = np.real(np.array([np.trace(rho@s[i]) for i in range(len(s))]))
  return 1.*c

