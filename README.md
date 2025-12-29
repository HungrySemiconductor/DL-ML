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

```
import torch
from torch import nn  # ✅ 神经网络模块
from torch.nn import functional as F  # ✅ 函数式接口
import torch.optim as optim  # ✅ 优化器
from torch.utils.data import DataLoader, Dataset  # ✅ 数据加载
```

### 2.1 张量

1. 张量：多维数组，一阶张量是标量，二阶张量是向量，三阶及以上统称为高阶张量

2. 张量主要属性：

   - data：被包装的张量
   - dtype：张量的数据类型
   - shape：张量的形状/维度
   - device：张量所在的设备，加速运算的关键，GPU/CPU
   - grad：data的梯度
   - grad_fn：创建张量的函数，自动求导的关键
   - requires_grad：是否需要计算梯度
   - is_lead：是否是叶子节点，叶子节点不可以执行in_place操作（在原地址中改变数据值）

3. 创建张量的方式

   - 数组直接创建

   - 概率分布创建：

     ```python
     # 从给定参数的离散正态分布中抽取随机数创建张量
     # mean或std维度不同时，会使用广播机制扩展成同型张量
     torch.normal(mean, std, size, out=None)
     
     # 从标准正态分布中抽取随机数创建张量，mean=0, std=1
     torch.randn(size, out=None, dtype=None, layout=torch.strided,device=None, required_grad=False)
     torch.randn_like(input, dtype=None, layout=None, device=None,required_grad=False)
     
     # 从[0,1)均匀分布中抽取随机数创建张量
     torch.randn(size, out=None, dtype=None, layout=torch.strided,device=None, required_grad=False)
     torch.rand_like(input, dtype=None, layout=torch.strided, device=None,required_grad=False)
     ```

     - mean：均值
     - std：标准差
     - size：仅在mean和std均为标量时使用，标识创建张量的形状
     - out：将结果直接写入已有张量
     - dtype：指定张量数据类型
     - layout：张量的内存布局方式（=torch.strided：**连续存储**访问效率高，=torch.sparse_coo：**稀疏存储**节省空间）
     - device：指定张量存储设备
     - required_grad：是否需要在张量上计算梯度

### 2.2 激活函数

1. 激活函数：在神经网络的神经元上运行的函数，将神经元的输入映射到输出端
   - 线性函数：无论有多少层神经网络，输出都是输入的线性函数，就相当于只有一个隐藏层，该情况相当于多层感知器
   - 非线性函数：非线性函数反复叠加，才使神经网络有足够能力抓取复杂特征
   
2. 常用激活函数

   | 激活函数                 | 计算公式                                                     | 说明                                                         |
   | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | Sigmoid                  | $f(x) = \dfrac{1}{1+e^{-x}}$                                 | 将变量映射到 (0,1)，输出均大于0<br>输入接近0时敏感（输入的小变化会引起输出的大变化）<br>其他大部分定义域内都饱和（通常\|𝑥\|>5），函数进入饱和区，梯度接近0，容易引发**梯度消失**问题 |
   | Tanh                     | $f(x) = \dfrac{sinh(x)}{cosh(x)} = \dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | 将变量映射到 (-1,1)，输出均匀地分布在y轴两侧                 |
   | ReLU                     | $f(x) = max(0, x)$                                           | 输入为负时输出始终为0，**神经元坏死（参数不再更新）**<br>函数再正无穷处梯度时一个常数，**解决了梯度消失问题** |
   | Leakly ReLU              | $f(x) = max(0.01x, x)$                                       | 负半轴引入一个泄露修正值（Leakly），对负输入有很小的坡度，使导数不总是为零，**解决了神经元不学习问题** |
   | 其他类型的激活函数...... |                                                              |                                                              |

### 2.3 损失函数

1. 损失函数：评估样本真实值与模型预测值之间的偏差，衡量模型的性能。

2. 损失函数选取：不同损失函数特性不同，用于不同场景，好的损失函数（模型预测精确度高）体现在越能扩大样本类间距离，减小类内距离

3. 常见的损失函数

   | 损失函数                       | 计算公式                                                     | 说明                                                         |
   | ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | L1范数损失函数                 | $loss(x,y)=\dfrac{1}{N}\sum_{i=1}^{N}|x-y|$                  | 预测值 与 真实值 的绝对误差取平均数                          |
   | 均方误差损失函数（MSE）        | $loss(x,y)=\dfrac{1}{N}\sum_{i=1}^{N}|x-y|^2$                | 预测值 与 真实值 的差的平方和的平均数                        |
   | 交叉熵损失函数（用于分类训练） | $H(p,q)=-\sum_{x}p(x)*logq(x)$                               | 实际输出概率与期望输出概率的距离，交叉熵的值越小，两个概率分布就越接近 |
   | 余弦相似度损失                 | $loss(x,y)=\begin{cases}     1-cos(x_1,x_2),& y==1\\     max(0,cos(x_1,x_2)-margin),&y==-1 \end{cases}$ | 让两个向量（有梯度）尽量相近                                 |
   | 其他类型的损失函数......       |                                                              |                                                              |

   - 模型调用方法

     ```python
     torch.nn.L1Loss(size_average=None,reduce=None,reduction='mean')
     torch.nn.MSELoss(reduce=True,size_average=True,reduction='mean')
     torch.nn.CrossEntropyLoss(weight=None,size_average=None,ignore_index=-100,reduce=None,reduction='mean')
     torch.nn.CosineEmbeddingLoss(margin=0.0,reduction='mean')
     ```

     - size_average：为True时返回loss的平均值，为False时返回个样本的loss数值之和
     - reduce：返回值是否为标量，默认为True
     - reduction：loss的参数，none返回一个向量（batch_size），mean返回均值，sum返回和
     - weight(tensor)：n个元素的一维张量，代表n类的权重
     - ignore_index：忽略某个类别，不计算器loss

### 2.4 优化器

1. 优化器：深度学习反向传播过程中，指引损失函数的各个参数往正确的方向更新合适的大小，使得更新后的各个参数让损失函数（目标函数）值不断逼近全局最小
2. 







## 3.深度神经网络

### 3.1 神经网络概述

> 神经元模型、多层感知机及MLP、前馈神经网络FNN

### 3.2 卷积神经网络CNN

1. 

2. 卷积神经网络类型

   | 卷积神经网络类型 |      |      |
   | ---------------- | ---- | ---- |
   | AlexNet          |      |      |
   | VGGNet           |      |      |
   | GoogLeNet        |      |      |
   | ResNet           |      |      |

   



## 4. 数据建模



[1] Deep Learning with PyTorch, Eli Stevens, Luca Antiga, Thomas Viehmann, 牟大恩 译, 人民邮电出版社

[2] 动手学PyTorch深度学习建模与应用, 王国平, 清华大学出版社