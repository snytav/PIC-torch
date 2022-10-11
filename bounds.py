import numpy as np
from poisson_matrix2D import poisson_matrix2D
from laplace2D import laplace2D

nx = 3
ny = 4

A = poisson_matrix2D(nx,ny)

u2 = np.random.rand(nx+2,ny+2)
print(u2)
f_real = np.dot(A.A,u2[1:-1,1:-1].reshape(nx*ny,1)).reshape(nx,ny)
print('real f            \n',f_real)
f_lap = laplace2D(u2)
print('Laplacian         \n',f_lap)
f = f_lap
print('RHS = Laplacian   \n ',f)
print('diff wit real RHS \n',f-f_real)
f[:,-1] = f[:,-1]+u2[1:-1,-1]
print('added Y+ boundary \n',f-f_real)
f[-1,:] = f[-1,:]+u2[-1,1:-1]
print('added Y- boundary \n',f-f_real)
f[0,:] = f[0,:]+u2[0,1:-1]
print('added X- boundary \n',f-f_real)
f[:,0] = f[:,0]+u2[1:-1,0]
print('added X+ boundary \n',f-f_real)

