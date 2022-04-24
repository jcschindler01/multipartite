
import numpy as np
from multipartite.constants import *
from multipartite.matrix import *


##

print(sigma[3])

def S_OBS_LO(rho=np.eye(2), A=[np.eye(2)], B=[np.eye(2)]):
  ## process inputs
  rho = rho.astype(complex)
  A, B = np.stack(A.astype(complex)), np.stack(B.astype(complex))

