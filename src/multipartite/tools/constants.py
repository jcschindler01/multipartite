
import numpy as np

## sigma matrices
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
sigma = np.stack([I,X,Y,Z])

## vectors
yy = dict()
yy['+z'] = np.array([1, 0 ], dtype=complex)
yy['-z'] = np.array([0, 1 ], dtype=complex)
yy['+x'] = np.array([1, 1 ], dtype=complex)
yy['-x'] = np.array([1,-1 ], dtype=complex)
yy['+y'] = np.array([1, 1j], dtype=complex)
yy['-y'] = np.array([1,-1j], dtype=complex)
yy['0'] = yy['+z']
yy['1'] = yy['-z']
yy['+'] = yy['+x']
yy['-'] = yy['-x']

## normalize vectors
for key in yy.keys():
  yy[key] = yy[key]/np.sqrt( np.conjugate(np.transpose(yy[key]))@yy[key] )

## projection matrices
pp = dict()
for key in yy.keys():
  pp[key] = np.outer(yy[key], np.conjugate(yy[key]))

## 2x2 matrix with Bloch vector x,y,z,t.
def qq(x=0.,y=0.,z=0.,t=1.):
  return .5*((1.+0j)*t*np.eye(2)+(1.+0j)*x*X+(1.+0j)*y*Y+(1.+0j)*z*Z)
