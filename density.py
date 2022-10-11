import torch

def XtoL(x,x0,dh):
    lc = torch.divide(torch.subtract(x,x0),dh)
    return lc


def scatter(lc,value,data):
    it = lc.int()
    dit = lc-it

    di = dit[0]
    dj = dit[1]
    dk = dit[2]

    i = it[0]
    j = it[1]
    k = it[2]

    data[i][j][k] +=       value*(1-di)*(1-dj)*(1-dk)
    data[i+1][j][k] +=     value*(di)*(1-dj)*(1-dk)
    data[i+1][j+1][k] +=   value*(di)*(dj)*(1-dk)
    data[i][j+1][k] +=     value*(1-di)*(dj)*(1-dk)
    data[i][j][k+1] +=     value*(1-di)*(1-dj)*(dk)
    data[i+1][j][k+1] +=   value*(di)*(1-dj)*(dk)
    data[i+1][j+1][k+1] += value*(di)*(dj)*(dk)
    data[i][j+1][k+1] +=   value*(1-di)*(dj)*(dk)


def density(xt,x0,dh,Nx,Ny,Nz,value):
    data = torch.zeros((Nx,Ny,Nz))
    for x in xt:
        lc = XtoL(x,x0,dh)
        scatter(lc, value, data)

    return data

import numpy as np
def Succi_density(x,dx,x0,J):
    rho = torch.zeros((J[0] + 2, J[1] + 2, J[2] + 2))

    for r in x:
        js = np.floor(np.divide(r,dx))
        ys = np.divide(r , dx) - (js - 1)
        js_plus_1 = np.mod(js, J) + 1
        i = int(js[0])
        j = int(js[1])
        k = int(js[2])
        i1 = int(js_plus_1[0])
        j1 = int(js_plus_1[1])
        k1 = int(js_plus_1[2])
        d1 = np.divide(1-ys,dx)
        d  = np.divide(ys,dx)
        rho[i][j][k]  += d1[0]*d1[1]*d1[2]
        rho[i1][j][k] += d[0]*d1[1]*d1[2]
        rho[i][j1][k]  += d1[0] * d[1] * d1[2]
        rho[i1][j1][k] += d[0] * d[1] * d1[2]
        rho[i][j][k1]  += d1[0]*d1[1]*d[2]
        rho[i1][j][k1] += d[0]*d1[1]*d[2]
        rho[i][j1][k1]  += d1[0] * d[1] * d[2]
        rho[i1][j1][k1] += d[0] * d[1] * d[2]

    return rho


def make1Dfrom3D(rho):
    rho2D = torch.sum(rho,1)
    rho1D = torch.sum(rho2D,1)
    return rho1D