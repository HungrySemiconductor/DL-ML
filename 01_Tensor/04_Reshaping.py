import torch


# ============================= #
#       Tensor Reshaping        #
# ============================= #

x = torch.arange(9)     # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
 
x_3xe = x.view(3, 3)        # 内存必须连续
x_3x3 = x.reshape(3, 3)     # 不要求内存连续，
print(x_3x3)              # tensor([[0, 1, 2],
#                                  [3, 4, 5],
#                                  [6, 7, 8]])


# ========== 拼接 ============
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape) # 按行拼接，输出torch.size([4, 5]) 
print(torch.cat((x1, x2), dim=1).shape) # 按列拼接，输出torch.size([2, 10]) 

# ========== 展平元素为1维张量 ============
z = x1.view(-1)
print(z.shape)  # torch.size([10])

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)  # torch.size([64, 10])

# ========== 转置 ============
y = x_3x3.t()    # 转置，交换行和列
print(y)

batch = 64
x = torch.rand((batch, 2, 5))
y = x.permute(0, 2, 1)    # 转置，交换2和1维度，即交换行和列维度
print(y.shape)  # torch.size([64, 5, 2])

# ========== 增加维度 ============
x = torch.arange(10)
print(x.shape)
print(x.unsqueeze(0).shape)    # 增加第0个维度，输出torch.size([1, 10])
print(x.unsqueeze(1).shape)    # 增加第1个维度，输出torch.size([10, 1])
print(x.unsqueeze(0).unsqueeze(1).shape)    # 增加第2个维度，输出torch.size([1, 1, 10])


z = x.squeeze(1)    # 压缩第1个维度，输出torch.size([1, 10])
print(z.shape)    
