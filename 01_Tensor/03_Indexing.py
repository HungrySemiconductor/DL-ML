import torch


# ============================= #
#       Tensor Indexing         #
# ============================= #

batch_size = 10
feature = 25

x = torch.rand((batch_size, feature))
print(x.ndimension())   # 输出张量纬度，即2维张量
print(x.numel())        # 输出张量元素个数，即250

# ========== 取值 ============
print(x[0, :])   # 取第一个样本的所有特征值
# print(x[0, :].shape)     # torch.size([25])
print(x[:, 0])   # 取所有样本的第一个特征值
# print(x[:, 0].shape)     # torch.size([10])
print(x[2, 0:10])   # 取第3个样本的前10个特征值(0-9)
# print(x[2, 0:10].shape)     # torch.size([10])

x = torch.arange(10)
indices = [2, 5, 8]
print(x)              # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x[indices])     # tensor([2, 5, 8])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x)
print(x[rows, cols])           # 依次取2个元素，第2行第5列和第1行第1列
print(x[rows, cols].shape)     # torch.size([2])

x = torch.arange(10)
print(x[(x < 2) | (x > 8)])    # 取所有小于2或大于8的元素
print(x[x.remainder(2) == 0])  # 取所有偶数

print(torch.where(x > 5, x, x*2))    # 取所有大于5的元素，否则取元素的2倍
print(torch.tensor([0,0,1,1,2,2,3,4]).unique())    # 取所有唯一的元素
