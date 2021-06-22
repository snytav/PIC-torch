import numpy as np

w = np.array([0.1,0.2,0.3])

A0 = np.eye(3)

W[0] = - np.linalg.det(A0-A*w)

A1 = np.diag(np.array([0,1,1]))

W[1] = np.linalg.det(A1-A*w)

A2 = np.diag(np.array([0,0,1]))

W[2 = np.linalg.det(A2-A*w)

W[7] = -np.prod(np.subtract(np.array([1,1,0]),w))



