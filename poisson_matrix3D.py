import scipy.sparse as sp
from poisson_matrix1D import poisson_matrix1D
from poisson_matrix2D import poisson_matrix2D

def poisson_matrix3D(nx,ny,nz):
    K1Dz = poisson_matrix1D(nz)
    K2D  = poisson_matrix2D(nx,ny)
    A1   = sp.kron(K2D,sp.eye(nz))
    A2   = sp.kron(sp.eye(nx*ny),K1Dz)
    return (A1+A2)

