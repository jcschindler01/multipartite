
import numpy as np
from multipartite.LO import S_OBS_LO
from multipartite.constants import *
from multipartite.matrix import *

import matplotlib.pyplot as plt
plt.style.use("classic")




###################### inputs ##############################

aa, bb = 3,4

aa, bb = float(aa), float(bb)

rho = (1./(aa+bb)) * (aa*kprod(projd['0'],projd['0']) + bb*kprod(projd['1'],projd['+']))

n = 100

#P_RDE = [couter(np.array([2,1]))/5.,couter(np.array([-1,2]))/5.]

B2 = [projd['+'],projd['-']]

#############################################################

## independent variable
s = np.linspace(0,1,n)
theta = np.pi*s

## output variables
Sobs  = np.nan*theta
SobsA = np.nan*theta
SobsB = np.nan*theta
Icl   = np.nan*theta
Svn   = np.nan*theta
SvnA  = np.nan*theta
SvnB  = np.nan*theta
Iqm   = np.nan*theta
outs  = (np.nan*theta).astype(object)


## A POVM
A = [projd['0'],projd['1']]

## iterate over theta changing B POVMs
for i in range(n):

  ## psi
  psi = np.array([np.cos(theta[i]/2.),np.sin(theta[i]/2.)])

  ## B projectors
  B = [couter(psi), np.eye(2)-couter(psi)]


  ## out
  meas = S_OBS_LO(rho,A,B).copy()

  #print(meas.keys())

  ## fill data
  Sobs[i]  = 1.*meas['S_AB_RHO']
  SobsA[i] = 1.*meas['S_A_RHOA']
  SobsB[i] = 1.*meas['S_B_RHOB']
  Icl[i]   = 1.*meas['I_cl']
  Svn[i]   = 1.*meas['S_RHO']
  SvnA[i]  = 1.*meas['S_RHOA']
  SvnB[i]  = 1.*meas['S_RHOB']
  Iqm[i]   = 1.*meas['I_qm']
  outs[i]  = meas.copy()


#plot
if True:
  ## new figure
  fig = plt.figure(1, figsize=(6,6))
  ax1 = fig.add_subplot(221)
  ax2 = fig.add_subplot(222)
  ax3 = fig.add_subplot(223)
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
    ssx = 2.*np.pi*np.linspace(-1.,1.001,1001)
    plt.plot(np.cos(ssx),np.sin(ssx),'.8')
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
  ## plot POVM bloch vectors for A
  for i in range(len(meas['A'])):
    x, y, z = 1.*meas['POVM_BLOCH_A'][i]
    V = .5*meas['Vx'][i]
    ax1.plot([0,x],[0,z], 'r-', alpha=1.*V, lw=2.*V, zorder=100)

## set axes if using above figure thing
if True:
  plt.sca(ax3)

## format axis
plt.xlabel(r"$\theta/\pi$")
plt.ylabel("bits")
plt.xlim(0,1)
plt.ylim(0,2)
plt.xticks([0,1])
plt.yticks([0,1,2])


## plot entropies
plt.plot(s, SvnB, 'm:')
plt.plot(s, SobsB, 'm-')
plt.plot(s, Iqm, 'g--')
plt.plot(s, Icl, 'g-')
plt.plot(s, Svn , 'b--')
plt.plot(s, Sobs, 'b-')


## show approximate optima
alpha = .15

## Sobs min
col = 'b'
## on bloch diagram
ii = 50
meas = outs[ii]
for i in range(len(meas['B'])):
  x, y, z = 1.*meas['POVM_BLOCH_B'][i]
  V = .5*meas['Vy'][i]
  ax2.plot([0,x],[0,z], c=col, alpha=alpha, lw=1., zorder=100)
## on entropy graph
ax3.plot([s[ii],s[ii]],[0,2], c=col, alpha=alpha, lw=1.)

## SobsB min
col = 'm'
## on bloch diagram
ii = 29
meas = outs[ii]
for i in range(len(meas['B'])):
  x, y, z = 1.*meas['POVM_BLOCH_B'][i]
  V = .5*meas['Vy'][i]
  ax2.plot([0,x],[0,z], c=col, alpha=alpha, lw=1., zorder=100)
## on entropy graph
ax3.plot([s[ii],s[ii]],[0,2], c=col, alpha=alpha, lw=1.)


## Icl max
col = 'g'
## on bloch diagram
ii = 70
meas = outs[ii]
for i in range(len(meas['B'])):
  x, y, z = 1.*meas['POVM_BLOCH_B'][i]
  V = .5*meas['Vy'][i]
  ax2.plot([0,x],[0,z], c=col, alpha=alpha, lw=1., zorder=100)
## on entropy graph
ax3.plot([s[ii],s[ii]],[0,2], c=col, alpha=alpha, lw=1.)


## plot the correlated points of rho
ax1.plot([0],[1], marker='*', markersize=15, c='c', zorder=9999)
ax2.plot([0],[1], marker='*', markersize=15, c='c', zorder=9999)
ax1.plot([0],[-1], marker='s', markersize=8, c='c', zorder=9999)
ax2.plot([1],[0], marker='s', markersize=8, c='c', zorder=9999)


## annotations
rhostring=r"$\rho = \dfrac{%d}{%d} \; |00\rangle \langle 00| + \dfrac{%d}{%d} \; |1+\rangle \langle 1+|$"%(aa,aa+bb,bb,aa+bb)

## rhoAB
sty = dict(size=16)
ax1.annotate(r"$\rho_A$", xy=(.96,.98), ha="right", va="top", xycoords="axes fraction", **sty)
ax2.annotate(r"$\rho_B$", xy=(.96,.98), ha="right", va="top", xycoords="axes fraction", **sty)

## entropy labels
sty = dict(size=10, color='b')
ax3.annotate(r"$S_{\rm obs}(\rho)$", xy=(.05,1.62), ha="left", va="bottom", **sty)
ax3.annotate(r"$S_{\rm vn}(\rho)$", xy=(.05,.98), ha="left", va="bottom", **sty)
sty = dict(size=10, color='m')
ax3.annotate(r"$S_{B}(\rho_B)$", xy=(.08,.73), ha="left", va="bottom", **sty)
ax3.annotate(r"$S_{\rm vn}(\rho_B)$", xy=(.95,.6), ha="right", va="bottom", **sty)
sty = dict(size=10, color='g')
ax3.annotate(r"$I_{\rm cl}$", xy=(.08,.15), ha="left", va="bottom", **sty)
ax3.annotate(r"$I_{\rm qm}$", xy=(.02,.6), ha="left", va="top", **sty)


## fourth axis just for annotations
ax4 = fig.add_subplot(224)
ax4.axis('off')
sty = dict(size=12, color='k')

ax4.annotate(rhostring, xy=(-.05,1), ha="left", va="top", xycoords="axes fraction", **sty)
# eqstr = r"$S_{A \otimes B}(\rho) = S_{A}(\rho_A) + S_{B}(\rho_B) - I_{cl}$"
# ax4.annotate(eqstr, xy=(-.05,.75), ha="left", va="top", xycoords="axes fraction", **sty)
# notestr = r"Best $LO^{*}$ measurement here"+"\n"+"optimizes neither marginal entropy"+"\n"+"nor measured mutual info."
# ax4.annotate(notestr, xy=(-.05,.5), ha="left", va="top", xycoords="axes fraction", size=10)




######### NOW DO AGAIN WITH A PARTICULAR POVM B2! ############

if True:

  ## B projectors
  B = B2

  ## out
  meas = S_OBS_LO(rho,A,B).copy()

  #print(meas.keys())

  ## fill data
  Sobs_B2  = 1.*meas['S_AB_RHO']
  SobsA_B2 = 1.*meas['S_A_RHOA']
  SobsB_B2 = 1.*meas['S_B_RHOB']
  Icl_B2   = 1.*meas['I_cl']
  Svn_B2   = 1.*meas['S_RHO']
  SvnA_B2  = 1.*meas['S_RHOA']
  SvnB_B2  = 1.*meas['S_RHOB']
  Iqm_B2   = 1.*meas['I_qm']
  outs_B2  = meas.copy()

  ## plot entropies
  ax3.plot([.5], [SobsB_B2], 'mo')
  ax3.plot([.5], [Icl_B2], 'go')
  ax3.plot([.5], [Sobs_B2], 'bo')

  col, alpha = 'r', .5
  for i in range(len(meas['B'])):
    x, y, z = 1.*meas['POVM_BLOCH_B'][i]
    V = .5*meas['Vy'][i]
    ax2.plot([0,x],[0,z], c=col, alpha=alpha, lw=1., zorder=100)


## save
plt.savefig('002.png', dpi=300)

