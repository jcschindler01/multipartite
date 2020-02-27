
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.linalg
from scipy.stats import unitary_group as UG
#import qubits.qubits as qb
plt.style.use("classic")



###############
### Matrix Operations
###############
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


###############
### Multipartite Matrix Operations
###############

## basis dictionaries
d22 = {'00':0, '01':1, '10':2, '11':3}
d23 = {'00':0, '01':1, '02':2, '10':3, '11':4, '12':5}
d222 = {'000':0, '001':1, '010':2, '011':3, '100':4, '101':5, '110':6, '111':7}
d223 = {'000':0, '001':1, '002':2, '010':3, '011':4, '012':5, '100':6, '101':7, '102':8, '110':9, '111':10, '112':11}
dxx = {'22':d22, '23':d23, '222':d222, '223':d223}

## dictionary lookup from n=(nA,nB,...)
def ndict(n):
	return dxx[''.join([str(nn) for nn in n])]

## generate list of basis index labels for system of type n=(nA,nB,...), possibly holding some index values fixed
## sums over all index values which are 'x' in the hold argument
## "mind" = "multipartite indices"
def mind(n=[2,2], hold='xx'):
	ss = [''.join([str(i) for i in range(nn)]) for nn in n]
	for i in range(len(n)):
		if not hold[i] == 'x':
			ss[i] = hold[i]
	ijk= [x for x in ss[0]]
	for s in ss[1:]:
		ijk = [x+y for x in ijk for y in s]
	return tuple(ijk)

## "rind" = "rho indices"
def rind(n=[2,2], hold='xx'):
	dd = ndict(n)
	return np.array([dd[idx] for idx in mind(n,hold)], dtype=int)

## compute reduced density matrices given n=(nA,nB,...) and rho
def REDUCE(rho, n):
	## check dims match
	if len(rho)==np.prod(n):
		if len(n)==1:
			red = [1.*rho]
		if len(n)>1:
			red = [np.zeros((nn,nn), dtype=complex) for nn in n]
			## iterate over subspaces
			for m in range(len(n)):
				## iterate over reduced density matrix elements
				for i in range(n[m]):
					for j in range(n[m]):
						## indices to sum over
						hold = len(n)*'x'
						hold = hold[:m]+str(i)+hold[m+1:]
						mi, ri = mind(n,hold), rind(n,hold)
						mj, rj = mind(n,hold.replace(str(i),str(j))), rind(n,hold.replace(str(i),str(j)))
						## fill rho
						red[m][i,j] = np.sum([rho[ri[k],rj[k]] for k in range(len(ri))], axis=0, dtype=complex)
	## return
	return tuple([1.*rr for rr in red])


###############
### Generate Arbitrary 2 or 3 dimensional rank-1 coarse-graining
###############

## function used in parameterizing SU3
def FF(v,w,x,y,z):
	v,w,x,y,z = 1.*v,1.*w,1.*x,1.*y,1.*z
	return -np.cos(v)*np.cos(w)*np.cos(x)*np.exp(1j*y) - np.sin(w)*np.sin(x)*np.exp(-1j*z)

## arbitrary SU matrix parameterized by real vector x in (0,1)
def SU(x):
	if len(x)==3:
		return SU2(x)
	if len(x)==8:
		return SU3(x)

## SU2 parameterized by real vector x
def SU2(x=np.zeros(3)):
	## identity is at x = np.array([0.,0.,0.])
	## periodic as each x=x+1, so use x in (0,1)
	th =  2.*np.pi*x
	su = np.array(
	              [
	              [ np.cos(th[0])*np.exp( 1j*th[1]), np.sin(th[0])*np.exp( 1j*th[2])],
	              [-np.sin(th[0])*np.exp(-1j*th[2]), np.cos(th[0])*np.exp(-1j*th[1])],
	              ], dtype=complex)
	return 1.*su

## SU3 parameterized by real vector x
def SU3(x=np.array([.5,.5,.5,0.,0.,0.,0.,0.])):
	## https://arxiv.org/abs/1303.5904
	## identity is at x = np.array([.5,.5,.5,0.,0.,0.,0.,0.])
	## periodic as each x=x+1, so use x in (0,1)
	x =  2.*np.pi*x
	pi = np.pi
	ph31, th31, th32, ph32, ch32, th21, ch21, ph21 = 1.*x
	su = np.zeros((3,3), dtype=complex)
	## top row
	su[0,0] = FF(th31,0,0,ph31+pi,0)
	su[0,1] = FF(th31-pi/2.,th32,pi,ph32,0)
	su[0,2] = FF(th31-pi/2.,th32-pi/2.,pi,ch32,0)
	## middle row
	su[1,0] = FF(th31-pi/2.,pi,th21,ph21,0)
	su[1,1] = FF(th31,th32,th21,-ph31+ph32+ph21,ch32+ch21)
	su[1,2] = FF(th31,th32-pi/2.,th21,-ph31+ch32+ph21,ph32+ch21)
	## bottom row
	su[2,0] = FF(th31-pi/2.,pi,th21-pi/2.,ch21,0)
	su[2,1] = FF(th31,th32,th21-pi/2,-ph31+ph32+ch21,ch32+ph21)
	su[2,2] = FF(th31,th32-pi/2,th21-pi/2,-ph31+ch32+ch21,ph32+ph21)
	## return
	return 1.*su

## make a set of projectors from the columns of a unitary matrix
def PROJ(U):
	proj = np.array([couter(U[i]) for i in range(len(U))], dtype=complex)
	return 1.*proj

## combine two sets of projectors by tensor product
def PROJPROD(PA,PB):
	return 1.*np.array([kprod(pa,pb) for pa in PA for pb in PB], dtype=complex)

## combine a list of sets of projectors by tensor product
def PROJMP(PX):
	proj = PX[0]
	for j in range(1,len(PX)):
		proj = PROJPROD(proj,PX[j])
	return 1.*proj

## product projectors parameterized by real vector x
## n=(nA,nB,...) dictates how dimensions split into product
def PROJN(x=np.zeros(6),n=[2,2]):
	n = np.array(n, dtype=int)
	if len(x)==np.sum(n**2-1):
		xi = np.split(x, np.cumsum(n**2 - 1)[:-1])
		Pi = [PROJ(SU(xi[j])) for j in range(len(xi))]
	proj = Pi[0]
	for j in range(1,len(Pi)):
		proj = PROJPROD(proj,Pi[j])
	return 1.*proj


###############
### Generate n dimensional Density Matrix
###############

## pure density matrix from psi
def RHO_PSI(psi):
	psi = psi.astype(complex) / np.sqrt(np.sum(np.abs(psi)**2))
	return 1.*couter(psi)

## Haar random pure state
def PSI_RAND(n):
	return 1.*UG.rvs(np.prod(n))[:,0]


###############
### Calculate Entropies
###############

## von Neumann entropy
def SVN(rho, decimals=9):
	vals, vecs = eig(rho)
	vals = np.round(vals, decimals) ## avoid tiny negative results
	mask = (vals != 0.)
	S = np.nan
	if len(vals[mask])>0:
		S = np.sum(-vals[mask]*np.log2(vals[mask]))
	return 1.*S

## observational entropy of rho with coarse-graining projectors P
def SOBS(rho, P, decimals=9):
	p, V = np.zeros(len(P)), np.zeros(len(P))
	for i in range(len(P)):
		V[i] = np.trace(np.real(P[i]))
		p[i] = np.trace(np.real(P[i]@rho@P[i]))
	p = np.round(p, decimals) ## avoid tiny negative results
	mask = p != 0.
	S = np.nan
	if len(p[mask])>0:
		S = np.sum(-p[mask]*np.log2(p[mask]/V[mask]))
	return 1.*S

## entanglement entropy
def SENT(rho=np.eye(4), n=[2,2], maxiter=100, initial_temp=1e3, eps=.05, projout=False):
	n = np.array(n, dtype=int)
	if len(rho)==np.prod(n):
		## set bounds for minimization input parameter x
		lenx = np.sum(n**2-1)
		bounds = [(-eps,1.+eps) for ll in range(lenx)]
		## function to minimize
		func = lambda x: SOBS(rho, PROJN(x,n))
		## minimize
		optout = scipy.optimize.dual_annealing(func, bounds, maxiter=maxiter, initial_temp=initial_temp)
		xmin, Smin = optout.x, optout.fun
		projmin = PROJN(xmin,n)
		## von Neumann entropy
		Svn = SVN(rho)
		## entanglement entropy
		Sent = Smin - Svn
	## return
	if projout==True:
		return 1.*Sent, 1.*projmin
	if projout==False:
		return 1.*Sent

## continue minimizing toward Sent = S(rhoA) in bipartite system
def SENT_BP(rho, n, Nmax=10, tol=1e-9, maxiter=100, initial_temp=1e3, eps=.05, projout=False):
	## init
	N = 0
	SvnA = SVN(REDUCE(rho,n)[0])
	SentMin = np.inf
	ProjMin = []
	DiffMin = np.inf
	## loop
	while (N<Nmax and np.abs(DiffMin)>tol):
		## print
		print("\nN=%d"%N)
		## calculate Sent
		Sent, projmin = SENT(rho,n,maxiter,initial_temp,eps,projout=True)
		diff = Sent-SvnA
		print("current")
		print("maxiter, initial_temp = %d, %.1e"%(maxiter,initial_temp))
		print("Sent = %.6f"%Sent)
		print("SvnA = %.6f"%SvnA)
		print("diff = %.6f"%diff)
		print("tol  = %.6f"%tol)
		## check if best and update
		if np.abs(diff)<np.abs(DiffMin):
			print("is new BEST")
			SentMin = 1.*Sent
			ProjMin = projmin
			DiffMin = diff
		## perturb minimization parameters
		maxiter = maxiter + 100
		initial_temp = (0.5+np.random.rand())*initial_temp
		## increment counter
		N += 1
	## reason for break
	if np.abs(DiffMin) < tol:
		print("\nTolerance Reached: Success!")
	if N==Nmax:
		print("\nMax Iterations Reached")
	## return
	if projout==True:
		return 1.*SentMin, ProjMin
	if projout==False:
		return 1.*SentMin


## calculate SOBS in projectors defined by reduced density matrix eigenbasis
## uniquely defined only when reduced density matrices have a full spectrum of distinct eigenvalues
## when uniquely defined, should be equal to Sent=S(rhoA)=S(rhoB) in bipartite pure state
def SOBS_RHOX(rho, n, decimals=9):
	red = REDUCE(rho, n)
	vecs = [eig(rr)[1] for rr in red]
	projx = [PROJ(v) for v in vecs]
	proj = PROJMP(projx)
	return 1.*SOBS(rho, proj, decimals=decimals)



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


## check S_{rhoA x rhoB x ...} = S(rhoA) = S(rhoB) in a bipartite system
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




## run tests
if False:
	if __name__=="__main__":
		print("\nTESTS\n")
		test7()



