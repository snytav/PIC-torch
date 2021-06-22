import torch

p=torch.randn(2,3)
q=torch.randn(3,4,5)

# Solution 1: Reshaping to use 2-dimensional torch.mm()
res1 = torch.mm(p, q.resize(3, 4 * 5)).resize_(2, 4, 5)
print(res1.shape)
# torch.Size([2, 4, 5])

# Solution 2: Using explicit torch.einsum()
res2 = torch.einsum("ab,bcd->acd", (p, q))
print(res2.shape)
# torch.Size([2, 4, 5])

# Checking if results are equal:
print((res1 == res2).all())
# tensor(1, dtype=torch.uint8)
