###### idea of the solver is the following:
#
#                                                           A*(u_in+u_b) = f
#
###### where A is the full Poisson matrix, with the the size (nx+2)*(ny+2)*(nz+2)**2,
#            [nx,ny,nz] = u_in.shape

######       u_in is the solution in the inner part of domain
######       u_b are the boundaries (expected to be set in advance - not obtained in solution)
######       f is the right-hand side for the whole system

#import torch
import numpy as np
from poisson_matrix3D import poisson_matrix3D
from add_boundary_conditions_to_RHS3D import add_boundary_conditions_to_RHS3D
from laplace3D import laplace3D
from bounds3D import bounds3D
import sys

def solve_poisson3D(u2,f_real,Xminus,Xplus,Yminus,Yplus,Zminus,Zplus):
    nx,ny,nz = f_real.shape
    K3D = poisson_matrix3D(nx,ny,nz)
    #K1D = torch.from_numpy(K1D.A)
    # f_real = np.dot(K3D.A,u2[1:-1,1:-1,1:-1].reshape(nx*ny*nz,1)).reshape(nx,ny,nz)
    f = add_boundary_conditions_to_RHS3D(f,f_real,Xminus,Xplus,Yminus,Yplus,Zminus,Zplus)
    u = np.linalg.solve(K3D.A,f.reshape(f.shape[0]*f.shape[1]*f.shape[2]))

    return u
  

if __name__ == "__main__":
    nx = 3
    ny = 4
    nz = 5

    u2 = np.random.rand(nx+2,ny+2,nz+2)
    bounds3D(u2)
    #sys.exit()
    
    f = laplace3D(u2)

    Xminus = u2[0   ,1:-1,1:-1]
    Xplus  = u2[-1  ,1:-1,1:-1]
    Yminus = u2[1:-1,0   ,1:-1]
    Yplus  = u2[1:-1,-1  ,1:-1]
    Zminus = u2[1:-1,1:-1,0]
    Zplus  = u2[1:-1,1:-1,-1] 
    
    ui = solve_poisson3D(u2,f,Xminus,Xplus,Yminus,Yplus,Zminus,Zplus)
    diff = np.max(np.abs(u2.reshape(u2.shape[0]*u2.shape[1]*u2.shape[2],1)-ui))
    print(diff)


