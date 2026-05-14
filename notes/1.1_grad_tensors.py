import torch
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
a = torch.add(w, x)
b = torch.add(w, 1)
y0 = torch.mul(a, b)
y1 = torch.add(a, b)

loss = torch.cat([y0, y1], dim=0)   # 沿维度 0 进行拼接
grad_t = torch.tensor([1., 2.])     # loss中每个元素的梯度权重
loss.backward(gradient=grad_t)      # 计算 loss 对 w 的梯度
print(w.grad)                       # 链式法则 输出 w 的梯度          