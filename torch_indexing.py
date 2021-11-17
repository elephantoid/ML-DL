import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)  # x[0,:]

print(x[:, 0].shape)

print(x[2, 0:10])

x[0, 0] = 100

# Fancy indexing
x = torch.arange(10)
indicies = [2, 5, 8]
print(x[indicies])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)
print(x[rows, cols])

# Advanced indexing
x = torch.arange(18)
print(x[(x < 3) | (x > 8)])
print(x[(x > 4) & (x < 10)])
print(x[x.remainder(2) == 0]) # 짝수

# Useful operation
print(torch.where(x>5, x, x*2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension())
print(x.numel()) # count values
