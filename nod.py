import torch

shift = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
n = 5 # number of particles for debug
shift_all = shift.repeat(n,1,1)

cn = torch.tensor([0,0,0.5]) # cell umber example for one particle
torch.add(shift_all,cn)
