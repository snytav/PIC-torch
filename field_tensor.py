import torch

#(2,8) and 2,8,3 -> 2,3

p=torch.randn(2,3)
q=torch.randn(3,4,5)
print('p.shape ',p.shape)
print('q.shape ',q.shape)

# Solution 2: Using explicit torch.einsum()
res2 = torch.einsum("ab,bcd->ad", (p, q))
print('res2.shape ',res2.shape)
# torch.Size([2, 4, 5])

