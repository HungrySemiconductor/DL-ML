import torch

# ============================= #
#           Tensor Math         #
# ============================= #

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z1 = x + y

z2 = torch.add(x, y)

z3 = torch.empty(3)
torch.add(x, y, out=z3)

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)

# inplace operations
t = torch.zeros(3)
t.add_(x) # 带有下划线的函数表示在原 tensor 上进行操作，不会产生副本
t += x    # 等价于 t.add_(x), 但不等价于 t = t + x

# Exponentiation
z = x.pow(2)
z = x ** 2

# Simple comparison
z = x > 0
z = x < 0
# print(z)

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)   # 矩阵乘法, 输出为 (2, 3) 的 tensor
x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)

# element wise multiplication
z = x * y
# print(z)

# dot product
z = torch.dot(x, y) # 点积输出为标量，z = x1 * y1 + x2 * y2 + x3 * y3
# print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)   # 批量矩阵乘法, 输出为 (batch, n, p) 的 tensor
# print(out_bmm.shape)


# Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2  # x2 会自动广播成 (5, 5) 的 tensor
z = x1 * x2  # x2 会自动广播成 (5, 5) 的 tensor
# print(z)

# Other useful tensor operations
sum_x = torch.sum(x, dim=0) # 对第 0 维进行求和
# print(sum_x)    # x只有一个维度，输出第0维的和，即6

values, indices = torch.max(x, dim=0)    # 对第 0 维进行求最大值
values, indices = torch.min(x, dim=0)
# print(f"values: ", values, f"indices: ", indices)

abs_x = torch.abs(x)    # 对x进行绝对值计算，返回新的 tensor

z = torch.argmax(x, dim=0)  # 对第 0 维进行求最大值的索引
z = torch.argmin(x, dim=0)  # 对第 0 维进行求最小值的索引

mean_x = torch.mean(x.float(), dim=0)   # 对第 0 维进行求平均值，返回浮点数 tensor

z = torch.eq(x, y)  # 判断x和y逐个元素是否相等，返回布尔值 tensor

sorted_y, indices = torch.sort(y, dim=0, descending=False)  # 对第 0 维进行排序，返回排序后的 tensor 和索引 tensor

z = torch.clamp(x, min=0, max=10)   # 对x进行裁剪，将所有元素限制在[0, 10]范围内

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)    # 判断x是否有True元素
z = torch.all(x)    # 判断x是否所有元素都为True
