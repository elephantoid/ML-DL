import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

torch.manual_seed(1)

lin = nn.Linear(5, 3)
data = torch.randn(3, 5)
print(lin(data))

data = torch.randn(2, 2)
print(data)
print(F.relu(data))

# Softmax is also in torch.nn.functional
data = torch.randn(5)
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum()) # Sums to 1 because it is a distribution!
print('max probability %.2f' %(F.softmax(data, dim=0).max()))
print(F.log_softmax(data, dim=0))