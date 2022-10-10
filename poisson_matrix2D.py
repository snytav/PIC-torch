import scipy.sparse as sp
import numpy as np
from poisson_matrix1D import poisson_matrix1D

def poisson_matrix2D(nx,ny):
    I1Dx = sp.eye(nx)
    I1Dy = sp.eye(ny)
    K1Dx = poisson_matrix1D(nx)
    K1Dy = poisson_matrix1D(ny)
    K2D = sp.kron(K1Dx,I1Dy)+sp.kron(I1Dx,K1Dy)

    return K2D
