
import numpy as np
from multipartite.LO import S_OBS_LO as slo
from multipartite.constants import *
from multipartite.matrix import *

# import matplotlib.pyplot as plt
# plt.style.use("classic")

rho = np.eye(4)/4.

A = [projd['0'],projd['1']]
B = [projd['0'],projd['1']]

m1 = slo(rho,A,B)

rde = m1['RDE_out'].copy()




