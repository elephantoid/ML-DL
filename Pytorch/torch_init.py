import torch

# ===================================== #
#         Initializing Tenser
# ===================================== #
# device= "cuda" if torch.cuda.is_available() else "cpu"
# Found GPU0 GeForce GT 740 which is of cuda capability 3.0. PyTorch no longer supports this GPU
# because it is too old. The minimum cuda capability that we support is 3.5.
# warnings.warn(old_gpu_warn % (d, name, major, capability[1]))
# print(torch.cuda.is_available())
# -->True
device='cpu'
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)

# other common initialization methods
x = torch.empty(size=(3,3))
x = torch.zeros((3,3))
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5) # I, eye 대각선으로 1 나머지 0
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10) # 10 step start 0.1 0.2 0.3 ...
x = torch.empty(size=(1,5)).normal_(mean=0, std=1) # 정규분포
x = torch.empty(size=(1,5)).uniform_(0,1) # 균등분포 uniform_(lower, upper)
x = torch.diag(torch.ones(3))


# how to initialize and convert tensors to other types
# 텐서의 dtype을 변환하는 방법
tensor = torch.arange(4)
# print(tensor.bool())
# print(tensor.short()) # int 16
# print(tensor.long()) # int 64
# print(tensor.half()) # float 16
# print(tensor.float()) # float32
# print(tensor.double()) # float64

# Array to Tensor
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
print(tensor.dtype)
np_array_back = tensor.numpy()
print(np_array_back.dtype)