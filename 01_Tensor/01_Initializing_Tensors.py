import torch

# ============================= #
#        Initialize Tensor      #
# ============================= #


device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                          device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size=(2, 3))    # 传入的是一个元组，未初始化，元素为随机值
x = torch.zeros((2, 3)) # 生成一个2*3的全0矩阵
x = torch.ones((3,3))   # 生成一个3*3的全1矩阵

x = torch.rand((3,3))   # 生成一个3*3的随机矩阵，元素在[0,1)之间
x = torch.empty(size=(2, 3)).uniform_(0,1)  # 生成一个2*3的随机矩阵，元素在[0,1)之间
x = torch.empty(size=(2, 3)).normal_(0,1)  # 生成一个2*3的正态分布矩阵，均值为0，标准差为1

x = torch.eye(5,5)      # 生成一个5*5的单位矩阵，只能创建对角线为1的矩阵
x = torch.diag(torch.ones(3))    # 生成一个3*3的对角矩阵，对角线元素为1，其他元素为0
x = torch.diag(torch.tensor([1, 2, 3]))  # 生成一个3*3的对角矩阵，对角线元素为1,2,3

x = torch.arange(start=0, end=5, step=1)        # tensor([0, 1, 2, 3, 4])
x = torch.linspace(start=0.1, end=1, steps=10)  # tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000,1.0000])


# How to initialize and convert tensor to other types(int, float, double)
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())   # dtype=torch.int16
print(tensor.long())    # dtype=torch.int64, 最常用
print(tensor.half())    # dtype=torch.float16
print(tensor.float())   # dtype=torch.float32, 最常用
print(tensor.double())  # dtype=torch.float64

# Convert tensor to numpy array
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array) # 将numpy数组转换为tensor
numpy_array = tensor.numpy()        # 将tensor转换为numpy数组
print(numpy_array)