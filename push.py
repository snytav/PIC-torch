import numpy as np
import torch
import time

def get_all_nodes_for_all(lc):
    # cna = [get_all_nodes(l) for l in lc]
    #
    # cn_all = []
    # for l in lc:
    #     t = get_all_nodes(l)
    #     cn_all.append(t)

    shift = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    if lc.is_cuda:
        device = torch.device('cuda')
        shift = shift.to(device)

    n = lc.shape[0]
    shift_all = shift.repeat(n,1,1)
    cn = lc
    cn = cn.unsqueeze(1)
    cn = cn.repeat(1,8,1)
    ind = torch.add(shift_all,cn)
    ind = ind.to(torch.int)


    return ind.tolist()

def get_all_nodes(cell_number): #return [(t[0]+1,t[1],t[2]),(t[0],t[1]+1,t[2]),(t[0],t[1],t[2]+1)]
    shift = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    indices = np.add(cell_number,shift)
    indices = indices.astype(int)
    shift = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
   # n = 5  # number of particles for debug
    #shift_all = shift.repeat(n, 1, 1)
    cn = torch.from_numpy(cell_number)
    ind = torch.add(shift, cn)
    indices1 = ind.numpy().astype(int)

    indices_tuples = [tuple(l) for l in indices1]
    return indices_tuples

def get_field_for_single_particle(grp):
    # fp_element = []
    # for l in grp:
    #     t = ef[l]
    #     fp_element.append(t)

    # return fp_element
    grp = torch.tensor(grp)
    idx = grp[:,0]
    idy = grp[:,1]
    idz = grp[:,2]

    fpel = ef[idx,idy,idz]
    return fpel

def get_fields_for_all_particles( cell_numbers_all):
    # fp1 = [[ef[l] for l in group] for group in cell_numbers_all]
    # return fp

    cn = torch.tensor(cell_numbers_all)
    cn_int = cn.to(torch.long)
    ix = cn_int[:, :, 0]
    iy = cn_int[:, :, 1]
    iz = cn_int[:, :, 2]
    efp = ef[ix,iy,iz]
    return efp

    # fp = []
    # for group in cell_numbers_all:
    #     # t = [ef[l] for l in group]
    #     # fp.append(t)
    #     fp_element = get_field_for_single_particle(group)
    #     fp.append(fp_element)
    #
    # return fp

# [i][j][k]
# [i+1][j][k]
# [i+1][j+1][k]
# [i][j+1][k]
# [i][j][k+1]
# [i+1][j][k+1]
# [i+1][j+1][k+1]
# [i][j+1][k+1]

def get_all_weights(d):
    di = d[0]
    dj = d[1]
    dk = d[2]
    t = [(1-di)*(1-dj)*(1-dk),(di)*(1-dj)*(1-dk),(di)*(dj)*(1-dk),(1-di)*(dj)*(1-dk),(1-di)*(1-dj)*(dk),
 (di)*(1-dj)*(dk),(di)*(dj)*(dk),(1-di)*(dj)*(dk)]
    # return t

    w = torch.from_numpy(d)
    e = torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    tt = torch.sub(e, w)
    t1 = torch.prod(tt, 1)
    m = torch.tensor([1, -1, 1, -1, -1, 1, -1, 1])
    t2 = t1*m
    return t2.numpy()

def get_all_weights_for_all(weights):
    # wall = [get_all_weights(l) for l in weights]

    # w1 = []
    # for l in weights:
    #     t = get_all_weights(l)
    #     w1.append(t)
    #
    w = weights
    n = len(weights)

    e = torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    if w.is_cuda:
        device = torch.device('cuda')
        e = e.to(device)

    e = e.repeat(n,1,1)

    w_sqz = w.unsqueeze(1)
    w2 = w_sqz.repeat(1, 8, 1)
    x = torch.sub(e, w2)
    pp = torch.prod(x, 2)
    w_pp = torch.abs(pp)
    # tt = torch.sub(e, w)
    # t1 = torch.prod(tt, 1)
    # m = torch.tensor([1, -1, 1, -1, -1, 1, -1, 1])
    # t2 = t1 * m

    return w_pp



def delta(xn,num_attribute,feat):
    # xn = xt.numpy()
    x_new_feat = feat[:, num_attribute, :]
    diff = np.subtract(xn, x_new_feat)
    diff_1D = diff.reshape(diff.shape[0] * diff.shape[1])
    abs_diff = np.abs(diff)
    dv = np.max(np.abs(diff_1D))
    res = np.where(abs_diff == np.amax(abs_diff))

    max_diff = np.amax(abs_diff)
    if max_diff > 1e-100:
       i_x_max = res[0].tolist()
       i_y_max = res[1].tolist()
    else:
        i_x_max = 0
        i_y_max = 0

    #max_diff = abs_diff[i_x_max][i_y_max]
    return max_diff,i_x_max,i_y_max


def push(xt,vt,x0t,dht,ef,charge,mass,dt):
    # function performs PARTICLE PUSH, or evaluation of the new coordinates and velocities of model particles
    # Note please: here the velocities are changed under the influence of the electric field, not because of collisions
    # PARAMETERS:
    # xt  - TORCH tensor of coordinates
    # vt  - TORCH tensor of velocities
    # x0t - the point with minimal coordinates for the whole computation domain (tensor of 3 doubles)
    # dht - mesh steps (tensor of 3 doubles)
    # ef  - electric field (tensor of the size ni*nj*nk*3). Each value ef[i][j][k] gives a 3D vector of the electric field at the
    #                 mesh node with number (i,j,k)
    # charge - charge of electrons (using the Imternational System of Units), in Coulomb units
    # mass   - mass of the electron, in kilograms
    # dt     - timestep (in seconds). Please use the same timestep for collision computation.

    lc_t = torch.div(torch.sub(xt, x0t), dht)
    lc_int_t = lc_t.to(torch.int)

    weights_t = torch.sub(lc_t,lc_int_t)
    weights_all = get_all_weights_for_all(weights_t)
    cell_numbers_all = get_all_nodes_for_all(lc_t)
    field_in_points =  get_fields_for_all_particles(cell_numbers_all)
    wt = weights_all
    ft = field_in_points
    # print('qq1')
    if wt.is_cuda:
        device = torch.device('cuda')
        ft = ft.to(device)

    eft = torch.einsum('ij,ijk->ik', wt,ft)

    ef_part = torch.mul(eft, (dt * charge / mass))
    vt = torch.add(vt,ef_part)
    v_dt = torch.mul(vt,dt)
    xt = torch.add(xt,v_dt)
    # d_lc, i_lc_x, i_lc_y = delta(lc_t.numpy(), 1, feat)
    # d_ef, i_ef_x, i_ef_y = delta(np.array(eft.numpy()), 1, feat)
    # d_v, i_v_x, i_v_y = delta(vt.numpy(), 3, feat)
    # d_x, i_x_x, i_x_y = delta(xt.numpy(), 5, feat)
    # print(d_v)
    if xt.is_cuda:
        xn = xt.cpu()
    else:
        xn = xt

    if vt.is_cuda:
        vn = vt.cpu()
    else:
        vn = vt

    return [xn.numpy(),vn.numpy()]


def PIC(xt,vt, x0t, dht, ef, feat):
    device = torch.device('cuda')
    print('using device ',torch.cuda.get_device_name(0))
    gpu_flag = False
    if gpu_flag:
       xt = xt.to(device)
       vt = vt.to(device)
       x0t = x0t.to(device)
       dht = dht.to(device)
       ef = ef.to(device)

    t_start = time.time()
    xn, vn = push(xt, vt, x0t, dht, ef, -1.602176565e-19, 9.10938215e-31, 2e-10)
    t_finish = time.time()

    print('Worktime ',t_finish - t_start)

    dv, iv_x, iv_y = delta(vn, 3, feat)
    dx, ix_x, ix_y = delta(xn, 5, feat)
    print('velocity delta, position delta ', dv, dx)


if __name__ == '__main__':
   # grid step
   # dh = [0.01, 0.01, 0.01]
   # coordinate origin
   # x0 = [-0.1, -0.1, 0]

   # importing array of features
   feat_big = np.loadtxt('electrons_a_4.txt')
   #number of particles
   n = int(feat_big.shape[0]/8/3)
   feat_big = feat_big.reshape(n,8,3)
   feat = feat_big[:,:,:]
   x = feat[:,4,:]
   v = feat[:,2,:]

   #number of nodes
   nx = 3
   ny = 4
   nz = 5

   u2 = np.zeros((nx + 2, ny + 2, nz + 2))
   # bounds3D(u2)
   # sys.exit()

   # f = laplace3D(u2)

   Xminus = u2[0, 1:-1, 1:-1]
   Xplus = u2[-1, 1:-1, 1:-1]
   Yminus = u2[1:-1, 0, 1:-1]
   Yplus = u2[1:-1, -1, 1:-1]
   Zminus = u2[1:-1, 1:-1, 0]
   Zplus = u2[1:-1, 1:-1, -1]

   #importing electric field

   #ef = np.loadtxt('ef_4_.txt')
   #ef = ef.reshape(ni,nj,nk,3)
  # ef = torch.from_numpy(ef)
   N = 1000
   Ny = 10
   L = 100.0
   Np = 20000
   Ly = 1.0
   # Lz = 4
   # xx = torch.random(N)
   x0  = torch.zeros(3)
   dh = torch.Tensor([L/N,Ly/Ny,Ly/Ny])

   xt = torch.Tensor(Np,3)
   xt[:,0] = torch.from_numpy(L*np.random.random(Np))
   xt[:,1] = torch.from_numpy(Ly*np.random.random(Np))
   xt[:,2] = torch.from_numpy(Ly*np.random.random(Np))
   vt = torch.Tensor(N,3)
   x0t = torch.tensor(x0.clone().detach())
   dht = torch.tensor(dh.clone().detach())

   from density import density
   rho = density(xt,x0,dh,N+2,Ny+2,Ny+2,1.0)
   phi = solve_poisson3D(u2, rho, Xminus, Xplus, Yminus, Yplus, Zminus, Zplus)
   # PIC(xt, vt, x0t, dht, ef, feat)

   xn,vn = push(xt,vt,x0t,dht,ef,-1.0,1.0,0.1)
   #
   dv,iv_x,iv_y = delta(vn,3,feat)
   dx,ix_x,ix_y = delta(xn,5,feat)
   print('velocity delta, position delta ',dv,dx)
