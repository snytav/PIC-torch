import numpy as np
import torch

f = np.loadtxt('f.txt')
w = np.loadtxt('w.txt')

f  = torch.from_numpy(f)
wt = torch.from_numpy(w)

ft = f.resize(2,8,3)

efs = torch.einsum("ab,cbd->ad", (wt,ft))

print('efs shape ',efs.shape)
print(efs)
