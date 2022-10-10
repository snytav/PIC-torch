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

