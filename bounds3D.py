import numpy as np
from poisson_matrix3D import poisson_matrix3D
from laplace3D import laplace3D
import sys

def bounds3D(u2):
    nx,ny,nz = u2.shape    
    nx = nx - 2
    ny = ny - 2
    nz = nz - 2       # N.B.: nx,ny,nz give the INNER PART size, withOUT boundary layers

    A = poisson_matrix3D(nx+2,ny+2,nz+2)  # matrix for the whole domain

    f_real = np.dot(A.A,u2.reshape((nx+2)*(ny+2)*(nz+2),1)).reshape(nx+2,ny+2,nz+2)
    print('real f            \n',f_real)
    f_lap = laplace3D(u2)
    print('Laplacian         \n',f_lap)
    f = f_lap
    print('RHS = Laplacian   \n ',f)
    print('INNER diff with real RHS \n',f-f_real[1:-1,1:-1,1:-1])
    f = np.zeros((u2.shape))
    f[1:-1,1:-1,1:-1] = f_lap
    print('INNER diff with real RHS \n', f[1:-1, 1:-1, 1:-1] - f_real[1:-1, 1:-1, 1:-1])

    f[:,-1,:] = np.add(f[:,-1,:],u2[:,-1,:])
#sys.exit()

    print('added Y+ boundary \n',f-f_real)
#sys.exit()
    f[-1,:,:] = np.add(f[-1,:,:],u2[-1,:,:])
    print('added Y- boundary \n',f-f_real)
#sys.exit()
    f[0,:,:] = np.add(f[0,:,:],u2[0,:,:])
    print('added X- boundary \n',f-f_real)
    f[:,0,:] = np.add(f[:,0,:],u2[:,0,:])
    print('added X+ boundary \n',f-f_real)

    f[:,:,0] = np.add(f[:,:,0],u2[:,:,0])
    print('added Z- boundary \n',f-f_real)
    f[:,:,-1] = np.add(f[:,:,-1],u2[:,:,-1])
    print('added Z+ boundary \n',f-f_real)

    ff = f-f_real
    print('MAX ',np.max(np.abs(ff)))




if __name__ == "__main__":
    nx = 3    
    ny = 4
    nz = 2

    u2 = np.random.rand(nx+2,ny+2,nz+2)    
    bounds3D(u2)
