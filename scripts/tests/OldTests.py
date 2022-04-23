
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.stats import unitary_group as UG

from multipartite import *



###############
### Tests
###############

## show that annealing finds minimum SOBS in 2x2 case
## should give Smin=0 every time in pure state
## it works!
def test1():
  ## psi
  #psi = np.random.random(2) + 1j*np.random.random(2)
  psi = np.array([1,1+1j])
  ## rho
  rho = RHO_PSI(psi)
  Svn = SVN(rho)
  ## minimize SOBS
  func = lambda x: SOBS(rho, PROJ(SU2(x)))
  eps = .01
  bounds = [(-eps,1.+eps) for i in range(3)]
  optout = scipy.optimize.dual_annealing(func, bounds, maxiter=200, initial_temp=1e1)
  umin = SU2(optout.x)
  projmin = PROJ(umin)
  Smin = optout.fun
  ## print
  print("\nTEST 1")
  print("\npsi")
  print(repr(psi))
  print("\nrho")
  print(repr(np.round(rho,3)))
  print("\nSvn")
  print(np.round(Svn,5))
  print("\noptout")
  print(optout)
  print("\numin")
  print(repr(np.round(umin,3)))
  print("\nprojmin")
  print(repr(np.round(projmin,3)))
  print("\nSmin")
  print(np.round(Smin,5))
  print()


## show that annealing finds minimum SOBS in 3x3 case
## should give Smin=0 every time in pure state
## it works!
def test2():
  ## psi
  psi = np.random.random(3) + 1j*np.random.random(3)
  #psi = np.array([1.,-1j,1.])
  ## rho
  rho = RHO_PSI(psi)
  Svn = SVN(rho)
  ## minimize SOBS
  func = lambda x: SOBS(rho, PROJ(SU3(x)))
  eps = .01
  bounds = [(-eps,1.+eps) for i in range(8)]
  optout = scipy.optimize.dual_annealing(func, bounds, maxiter=100, initial_temp=1e1)
  umin = SU3(optout.x)
  projmin = PROJ(umin)
  Smin = optout.fun
  ## print
  print("\nTEST 2")
  print("\npsi")
  print(repr(psi))
  print("\nrho")
  print(repr(np.round(rho,3)))
  print("\nSvn")
  print(np.round(Svn,5))
  print("\noptout")
  print(optout)
  print("\numin")
  print(repr(np.round(umin,3)))
  print("\nprojmin")
  print(repr(np.round(projmin,3)))
  print("\nSmin")
  print(np.round(Smin,5))
  print()



## compute reduced density matrix in an arbitrary dimensional system
def test3():
  ## input
  n = [2,2,3]
  psi = np.array([1.,0.,0.,0.,0.,1.])
  ## set psi randomly if not given properly
  N = np.prod(n)
  rand = False
  if len(psi) != N:
    psi = np.random.random(N) + 1j*np.random.random(N)
    rand = True
  ## go
  rho = RHO_PSI(psi)
  Svn = SVN(rho)
  red = REDUCE(rho,n)
  Svn_red = [SVN(rr) for rr in red]
  ## print
  print("\nTEST 3")
  if rand==True:
    print("\nrandom psi")
  print("\npsi")
  print(repr(psi))
  print("\nrho")
  print(repr(np.round(rho,3)))
  print("Svn = %.3f"%Svn)
  for m in range(len(n)):
    print("\nREDUCED SYSTEM m=%d"%m)
    print("rho_red")
    print(repr(np.round(red[m],3)))
    print("Svn_red = %.3f"%Svn_red[m])
  print()


## compute entanglement entropy in arbitary pure system
def test4():
  ## input
  n = [2,3]
  psi = np.array([1.,0.,0.,0.,1.,1.,0.])
  maxiter = 100
  initial_temp = 1e2
  eps = .01
  ## psi if copy pasting from output
  ## set psi randomly if not given properly
  N = np.prod(n)
  rand = False
  if len(psi) != N:
    psi = np.random.random(N) + 1j*np.random.random(N)
    rand = True
  ## go
  rho = RHO_PSI(psi)
  Svn = SVN(rho)
  red = REDUCE(rho,n)
  SvnRed = [SVN(rr) for rr in red]
  Sent, projmin = SENT(rho,n, maxiter=maxiter, initial_temp=initial_temp, eps=eps, projout=True)
  ## subsys labels
  sub = 'ABCD'
  ## print
  print("\nTEST 4")
  print("\npsi")
  print(repr(psi))
  for m in range(len(n)):
    print("\nREDUCED SYSTEM m=%d"%m)
    print("rho_red")
    print(repr(np.round(red[m],3)))
  print("\nprojmin")
  print(repr(np.round(projmin,2)))
  print("\nrho")
  print(repr(np.round(rho,3)))
  print()
  if rand==True:
    print("psi = random")
  print( "N = %s"%(len(rho)))
  print( "n = %s"%(n))
  print( "Svn  = %.3f"%(Svn))
  [print("Svn%s = %.3f"%(sub[i],SvnRed[i])) for i in range(len(n))]
  print( "Sent = %.3f"%(Sent))
  print()


## compute entanglement entropy using SENT_BP in a pure BIPARTITE system
def test5():
  ## input
  n = [2,3]
  psi = np.array([1.,0.,0.,1.])
  maxiter = 100
  initial_temp = 1e3
  eps = .05
  Nmax = 10
  tol = 4e-3
  # ## psi if copy pasting from output
  ## set psi randomly if not given properly
  N = np.prod(n)
  rand = False
  if len(psi) != N:
    psi = np.random.random(N) + 1j*np.random.random(N)
    rand = True
  ## go
  rho = RHO_PSI(psi)
  Svn = SVN(rho)
  red = REDUCE(rho,n)
  SvnRed = [SVN(rr) for rr in red]
  Sent, projmin = SENT_BP(rho,n, Nmax=Nmax, tol=tol, maxiter=maxiter, initial_temp=initial_temp, eps=eps, projout=True)
  ## subsys labels
  sub = 'ABCD'
  ## print
  print("\nTEST 5")
  print("\npsi")
  print(repr(psi))
  for m in range(len(n)):
    print("\nREDUCED SYSTEM m=%d"%m)
    print("rho_red")
    print(repr(np.round(red[m],3)))
  print("\nprojmin")
  print(repr(np.round(projmin,2)))
  print("\nrho")
  print(repr(np.round(rho,3)))
  print()
  if rand==True:
    print("psi = random")
  print( "N = %s"%(len(rho)))
  print( "n = %s"%(n))
  print( "Svn  = %.3f"%(Svn))
  [print("Svn%s = %.3f"%(sub[i],SvnRed[i])) for i in range(len(n))]
  print( "Sent = %.3f"%(Sent))
  print()


## compute entanglement entropy in tripartite system
def test6():
  ## input
  n = [2,2,2]
  psi = np.array([1.,0.,0.,1.,0.,0.,0.,1.,np.nan])
  maxiter = 100
  initial_temp = 1e3
  eps = .05
  # ## psi if copy pasting from output
  ## set psi randomly if not given properly
  N = np.prod(n)
  rand = False
  if len(psi) != N:
    psi = np.random.random(N) + 1j*np.random.random(N)
    rand = True
  ## go
  rho = RHO_PSI(psi)
  Svn = SVN(rho)
  red = REDUCE(rho,n)
  SvnRed = [SVN(rr) for rr in red]
  Sent, projmin = SENT(rho,n, maxiter=maxiter, initial_temp=initial_temp, eps=eps, projout=True)
  ## subsys labels
  sub = 'ABCD'
  ## print
  print("\nTEST 6")
  print("\npsi")
  print(repr(psi))
  for m in range(len(n)):
    print("\nREDUCED SYSTEM m=%d"%m)
    print("rho_red")
    print(repr(np.round(red[m],3)))
  print("\nprojmin")
  print(repr(np.round(projmin,2)))
  print("\nrho")
  print(repr(np.round(rho,3)))
  print()
  if rand==True:
    print("psi = random")
  print( "N = %s"%(len(rho)))
  print( "n = %s"%(n))
  print( "Svn  = %.3f"%(Svn))
  [print("Svn%s = %.3f"%(sub[i],SvnRed[i])) for i in range(len(n))]
  print( "Sent = %.3f"%(Sent))
  print()


## check S_{rhoA x rhoB} = S(rhoA) = S(rhoB) in a bipartite system
def test7():
  ## input
  n = [2,2]
  psi = np.array([])
  maxiter = 100
  initial_temp = 1e3
  eps = .05
  # ## psi if copy pasting from output
  ## set psi randomly if not given properly
  N = np.prod(n)
  rand = False
  if len(psi) != N:
    psi = PSI_RAND(n)
    rand = True
  ## go
  rho = RHO_PSI(psi)
  Svn = SVN(rho)
  red = REDUCE(rho,n)
  SvnRed = [SVN(rr) for rr in red]
  Sent, projmin = SENT(rho,n, maxiter=maxiter, initial_temp=initial_temp, eps=eps, projout=True)
  Srhox = SOBS_RHOX(rho,n)
  ## subsys labels
  sub = 'ABCD'
  ## print
  print("\nTEST 7")
  print("\npsi")
  print(repr(psi))
  for m in range(len(n)):
    print("\nREDUCED SYSTEM m=%d"%m)
    print("rho_red")
    print(repr(np.round(red[m],3)))
  print("\nprojmin")
  print(repr(np.round(projmin,2)))
  print("\nrho")
  print(repr(np.round(rho,3)))
  print()
  if rand==True:
    print("psi = random")
  print( "N = %s"%(len(rho)))
  print( "n = %s"%(n))
  print( "Svn   = %.3f"%(Svn))
  [print("Svn%s  = %.3f"%(sub[i],SvnRed[i])) for i in range(len(n))]
  print( "Sent  = %.3f"%(Sent))
  print( "Srhox = %.3f"%(Srhox))
  print()


## compute S_{rhoA x rhoB x rhoC} in a 2x2x2 system
def test8():
  ## print options
  if False:
    import sys
    sys.stdout = open('log.txt','w')
  np.set_printoptions(linewidth=200)
  ## input
  n = [2,2,2]
  psi = np.array([1.,1.,0.,0.,0.,0.,0.,1.])
  maxiter = 200
  initial_temp = 1e3
  eps = .05
  # ## psi if copy pasting from output
  ## set psi randomly if not given properly
  N = np.prod(n)
  rand = False
  if len(psi) != N:
    psi = PSI_RAND(n)
    rand = True
  ## go
  psinormed = psi.astype(complex) / np.sqrt(np.sum(np.abs(psi)**2))
  rho = RHO_PSI(psinormed)
  Svn = SVN(rho)
  red = REDUCE(rho,n)
  SvnRed = [SVN(rr) for rr in red]
  ValsRed = [eig(rr)[0] for rr in red]
  Sent, projmin, projfactors = SENT(rho,n, maxiter=maxiter, initial_temp=initial_temp, eps=eps, projout=True, pfactors=True)
  Srhox = SOBS_RHOX(rho,n)
  ## subsys labels
  sub = 'ABCD'
  ## print
  print("\nTEST 8")
  print("\npsi (before normalization)")
  print(repr(psi))
  for m in range(len(n)):
    print("\nREDUCED SYSTEM m=%d"%m)
    print("\nrho_red")
    print(repr(np.round(red[m],3)))
    print("\nrho_red eigenvals")
    print(repr(np.round(ValsRed[m],3)))
    print("\nlocal factors of projmin")
    print(repr(np.round(projfactors[m],3)))
  print("\nprojmin")
  print(repr(np.round(projmin,2)))
  print("\npsi")
  print(repr(np.round(psinormed,3)))
  print("\nrho")
  print(repr(np.round(rho,3)))
  print()
  if rand==True:
    print("psi = random")
  print( "N = %s"%(len(rho)))
  print( "n = %s"%(n))
  print( "Svn   = %.3f"%(Svn))
  [print("Svn%s  = %.3f"%(sub[i],SvnRed[i])) for i in range(len(n))]
  print( "Sent  = %.3f"%(Sent))
  print( "Srhox = %.3f"%(Srhox))
  print()


## check that tensor product of projectors is working properly
def test9():
  ## import
  import qubits.qubits as qb
  ## set subsys projectors
  UA = np.array([[1.,0.],[0.,1.]], dtype=complex)
  UB = np.array([[1.,0.],[0.,1.]], dtype=complex)
  UC = (1./np.sqrt(2))*np.array([[1.,1.],[1.,-1.]], dtype=complex)
  ## or random
  # UA = UG.rvs(2)
  # UB = UG.rvs(2)
  # UC = UG.rvs(2)
  ##
  U = [UA,UB,UC]
  n = [len(UU) for UU in U]
  N = np.prod(n)
  p = [PROJ(UU) for UU in U]
  subs = 'ABCD'
  ## print
  print("TEST 9")
  for i in range(len(p)):
    print("\nSUBSYS %s"%(subs[i]))
    print("proj")
    print(p[i])
    print("isvalid = %s"%(qb.isvalid_proj(p[i])))
  pprod = PROJMP(p)
  print("\ntensor product projectors")
  print(pprod)
  print("isvalid = %s"%(qb.isvalid_proj(pprod)))
  ## if want to compare to alternate method
  if False:
    ## construct product projectors an alternate way
    pprod2 = []
    for i in range(len(p[0])):
      for j in range(len(p[1])):
        for k in range(len(p[2])):
          pp = np.zeros((N,N), dtype=complex)
          mm, rr = mind(n), rind(n)
          for ii in range(len(pp)):
            for jj in range(len(pp)):
              mmii = [int(s) for s in mm[ii]]
              mmjj = [int(s) for s in mm[jj]]
              pp[rr[ii],rr[jj]] = p[0][i][mmii[0],mmjj[0]]*p[1][j][mmii[1],mmjj[1]]*p[2][k][mmii[2],mmjj[2]]
          pprod2 += [1.*pp]
    pprod2 = 1.*np.array(pprod2, dtype=complex)
    print("\ntensor product projectors 2")
    print(pprod2)
    print("isvalid = %s"%(qb.isvalid_proj(pprod2)))
    ## check if both methods are equal
    print("\nboth methods agree if true")
    tol = 1e-9
    for m in range(len(pprod)):
      print(np.all(np.abs(pprod[m]-pprod2[m]) < tol))
    print()
            
            

## run tests
if True:
  if __name__=="__main__":
    print("\nTESTS\n")
    test4()

