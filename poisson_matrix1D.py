import scipy.sparse as sp
import numpy as np

def poisson_matrix1D(N):
    A = -np.ones((N,3))
    A[:,1] = A[:,1]*(-2)
    K1D = sp.spdiags(A.T, [-1, 0, 1], N,N)
    return K1D

