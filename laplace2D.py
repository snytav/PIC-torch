import numpy as np

def laplace2D(q):
    f =-q[2:,1:-1]-q[:-2,1:-1]+2*q[1:-1,1:-1]-q[1:-1,2:]-q[1:-1,:-2]+2*q[1:-1,1:-1]
    return f
