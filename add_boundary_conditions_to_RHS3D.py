

def add_boundary_conditions_to_RHS3D(f,f_real,Xminus,Xplus,Yminus,Yplus,Zminus,Zplus):

    f[:,-1,:] = f[:,-1,:] + Yplus
    print('Yplus ',f-f_real)
    f[-1,:,:] = f[-1,:,:] + Xplus
    print('Xplus ',f-f_real)
    f[0,:,:]  = f[0,:,:]  + Xminus
    print('Xminus ',f-f_real)
    f[:,0,:]  = f[:,0,:]  + Yminus
    print('Yminus ',f-f_real)
    f[:,:,0]  = f[:,:,0]  + Zminus
    print('Zminus ',f-f_real)
    f[:,:,-1] = f[:,:,-1] + Zplus
    print('Zplus ',f-f_real)

    return f 

