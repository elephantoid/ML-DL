import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 6, 9])

# add
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z = x + y

print(z, z1, z2)

# sub
z = x - y
print(z)

# div
z = torch.true_divide(x, y)
print(z)

# inplace operation
t = torch.zeros(3)
t.add_(x)
print(t)
t += x  # t = t+x

# Exponential
z = x.pow(2)
z = x ** 2
print(z)

# simple comparison
z = x > 2
print(z)

# Matrix Multiplcation
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x1, '\n', x2, '\n', x3)

# matrix exponential
matrix_exp = torch.rand(5,5).normal_(mean=0, std=1)
print(matrix_exp.matrix_power(3))

# element wise multiplication
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 6, 9])
z= x*y
print(z)

# dot product
z = torch.dot(x,y)
print(z)

# Batch Matrix Multiplication
batch= 32
n = 10
m= 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand(((batch, m, p)))
out_bmm = torch.bmm(tensor1, tensor2) # batch, n ,p

# Example of Broadcasting
x1 =torch.rand((5,5))
print(x1)
x2 = torch.rand((1,5))
print(x2)
z = x1 - x2
print(z)

# Other useful tensor operation
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
print(values, indices)
values, indices = torch.min(x, dim=0)
abs_x= torch.abs(x)
z= torch.argmax(x, dim=0)
z= torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y) # if x==y True else False
print(z)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10) # like clamp min ~ max
# if max values is lower than parameter max value is converted like max parameter
print(z)

x= torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x) # True
z = torch.all(x) # False

print(z)