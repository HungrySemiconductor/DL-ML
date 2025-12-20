import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 准备拟合数据
x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y = np.cos(x)
X = np.expand_dims(x, axis=1)
Y = y.reshape(400, -1)
dataset = TensorDataset(torch.tensor(X, dtype=torch.float),torch.tensor(Y, dtype=torch.float))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 搭建神经网络，简单线性结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_features=1, out_features=10),nn.ReLU(),
            nn.Linear(10,100),nn.ReLU(),
            nn.Linear(100,10),nn.ReLU(),
            nn.Linear(10,1)
        )

    def forward(self, input:torch.FloatTensor):
        return self.net(input)
    
net = Net()

# 设置优化器和损失函数
optim = torch.optim.Adam(Net.parameters(net), lr=0.001)
Loss = nn.MSELoss()

# 开始训练模型，训练100次
for epoch in range(100):
    loss = None
    for batch_x, batch_y in dataloader:
        y_predict = net(batch_x)
        loss = Loss(y_predict, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    if (epoch+1) % 10 == 0:
        print("训练步骤：{0}, 模型损失：{1}".format(epoch+1, loss.item()))

# 训练完成后预测
predict = net(torch.tensor(X, dtype=torch.float))

# 绘制预测值和真实值之间的折线图
plt.figure(figsize=(12, 7), dpi=160)
plt.plot(x, Y, label="真实值", marker="X")
plt.plot(x, predict.detach().numpy(), label="预测值", marker="o")
plt.xlabel("x", size=15)
plt.ylabel("y", size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(fontsize=15)
plt.title("余弦函数拟合", size=20)
plt.show()
