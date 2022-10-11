import numpy as np

def laplace1D(q):
    f =-q[2:]-q[:-2]+2*q[1:-1]
    return f
