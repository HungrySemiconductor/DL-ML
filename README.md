# DL-ML

## 1. PyTorch自动求导模块

```python
torch.autograd.backward(tensors,
						grad_tensors=None,
						retain_graph=None,
						create_graph=False)
```

- tensors：用于求导的张量，例如 loss
- grade_tensors：多梯度权重，计算多个loss梯度时设置每个loss的权值

- retain_graph：保存计算图，PyTorch采用动态图机制，每次反向传播后计算图都会释放掉

- create_graph：创建导数计算图，用于高阶求导

```python
torch.autograd.grad(outputs,
					inputs,
					grad_outputs=None,
					retain_graph=None,
					create_graph=False)
```

- outputs：用于求导的张量，例如 loss
- inputs：需要梯度的张量，即组成 loss 的原始张量
- grade_tensors：多梯度权重，计算多个loss梯度时设置每个loss的权值

- retain_graph：保存计算图，PyTorch采用动态图机制，每次反向传播后计算图都会释放掉

- create_graph：创建导数计算图，用于高阶求导

## 2. PyTorch基本概念

1. 







[1] Deep Learning with PyTorch, Eli Stevens, Luca Antiga, Thomas Viehmann, 牟大恩 译, 人民邮电出版社

[2] 动手学PyTorch深度学习建模与应用, 王国平, 清华大学出版社