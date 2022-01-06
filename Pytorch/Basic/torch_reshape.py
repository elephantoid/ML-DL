import torch

'''
view, reshape 구조를 변경하나 순서를 변경하지 않는 공통점이 있다.
view = 원본 tensor 기반
reshape = 복사본
contiguous = 메모리상 인접해 있는가 
data_ptr()로 memory space 확인, is_contiguous()를 통해 True or False
transpose(), permute()
transpose는 오직 2개의 차원을 맞교환
permute는 원하는 대로 맞교환
'''
x = torch.arange(9)
# 연속적으로 메모리를 계속 잡아먹음
x_3x3 = x.view(3, 3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)
print(x_3x3.shape)

y = x_3x3.t()  # transpose
print(y)
print(y.contiguous().view(9))  # tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)  # torch.Size([4, 5])
print(torch.cat((x1, x2), dim=1).shape)  # torch.Size([2, 10])
print(torch.cat((x1, x2)).shape)  # torch.Size([4, 5])
z = x1.view(-1)  # flatten entire things
print(z.shape)  # torch.Size([10]) 2x5

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)  # torch.Size([64, 10])

z = x.permute(0, 2, 1)  # origin_x =[64,2,5] permuted_x=[64,5,2]
print(z.shape)  # torch.Size([64, 5, 2])

x = torch.arange(10)  # [10]
print(x.unsqueeze(0).shape) # [1, 10]
print(x.unsqueeze(1).shape) # [10, 1]

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10
z = x.squeeze(1)
print(z.shape) # torch.Size([1, 10])