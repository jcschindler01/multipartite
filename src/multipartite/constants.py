
import numpy as np

## sigma matrices
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
sigma = np.stack([I,X,Y,Z])

## vectors
psid = dict()
psid['+z'] = np.array([1, 0 ], dtype=complex)
psid['-z'] = np.array([0, 1 ], dtype=complex)
psid['+x'] = np.array([1, 1 ], dtype=complex)
psid['-x'] = np.array([1,-1 ], dtype=complex)
psid['+y'] = np.array([1, 1j], dtype=complex)
psid['-y'] = np.array([1,-1j], dtype=complex)
psid['0'] = psid['+z']
psid['1'] = psid['-z']
psid['+'] = psid['+x']
psid['-'] = psid['-x']

## normalize vectors
for key in psid.keys():
  psid[key] = psid[key]/np.sqrt( np.conjugate(np.transpose(psid[key]))@psid[key] )

## projection matrices
projd = dict()
for key in psid.keys():
  projd[key] = np.outer(psid[key], np.conjugate(psid[key]))
