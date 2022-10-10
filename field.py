import torch

def get_field(phi,dh):
    efx = torch.zeros_like(phi)
    efy = torch.zeros_like(phi)
    efz = torch.zeros_like(phi)

    efx[1:-1,:,:] = (phi[2:,:,:] - phi[0:-2,:,:])/2/dh[0]
    efy[:,1:-1,:] = (phi[:,2:, :] - phi[:,0:-2, :]) / 2 /dh[1]
    efz[:,:,1:-1] = (phi[:,:,2:] - phi[:,:,0:-2]) / 2 /dh[2]

    ef = torch.Tensor([efx,efy,efz])
