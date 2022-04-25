
import numpy as np
from multipartite.LO import S_OBS_LO as slo
from multipartite.constants import *
from multipartite.matrix import *

import matplotlib.pyplot as plt
plt.style.use("classic")


###################### inputs ##############################
rho = (3.*kprod(projd['0'],projd['0']) + 4.*kprod(projd['1'],projd['+']))/7.

A = [projd['0'],projd['1']]
B = [projd['-'],projd['+']]
#############################################################

def main():
  ## calculate
  meas = slo(rho,A,B).copy()
  rde = meas['RDE_out'].copy()

  ## print things
  print()
  print(meas.keys())
  print()

  ## fancyplot
  fancyplot(meas)


def fancyplot(meas):
  ## new figure
  fig = plt.figure(1, figsize=(6,6))
  ax1 = fig.add_subplot(221)
  ax2 = fig.add_subplot(222, sharey=ax1)
  ax3 = fig.add_subplot(223, sharex=ax1)
  AX = [ax1,ax2,ax3]
  for ax in [ax1,ax2]:
    ## format axes
    plt.sca(ax)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.gca().set_aspect(1)
    plt.xticks([])
    plt.yticks([])
    ## dots at center
    plt.plot([0],[0], 'k.',  markersize=7, zorder=999)
    ## circle on bloch sphere
    s = 2.*np.pi*np.linspace(-1.,1.001,1001)
    plt.plot(np.cos(s),np.sin(s),'.8')

  ## plot bloch vectors x and z direction
  rA, rB = meas['rA'], meas['rB']
  sty1 = dict(ls='-', lw=2, c='k', zorder=99)
  sty2 = dict(marker='.', c='k', markersize=15, zorder=9999)
  ## lines
  AX[0].plot([0,rA[0]],[0,rA[2]], **sty1)
  AX[1].plot([0,rB[0]],[0,rB[2]], **sty1)
  ## dots
  AX[0].plot([rA[0]],[rA[2]], **sty2)
  AX[1].plot([rB[0]],[rB[2]], **sty2)


  ## plot POVM bloch vectors
  for i in range(len(meas['A'])):
    x, y, z = 1.*meas['POVM_BLOCH_A'][i]
    V = .5*meas['Vx'][i]
    ax1.plot([0,x],[0,z], 'r-', alpha=1.*V, lw=2.*V, zorder=100)
  for i in range(len(meas['B'])):
    x, y, z = 1.*meas['POVM_BLOCH_B'][i]
    V = .5*meas['Vy'][i]
    ax2.plot([0,x],[0,z], 'r-', alpha=1.*V, lw=2.*V, zorder=100)


  ## plot quantum entropies









  ## show
  plt.savefig('LOex.png', dpi=300)



main()
