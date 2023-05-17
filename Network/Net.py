import torch
import torch.nn as nn
# 自定义网络结构，包含三个全连接层和两个relu激活函数

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh()
        )
        self.block3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh()
        )
        self.block4 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
       # x = torch.dropout(x, 0.5, True)
        x = self.block3(x)
        #x = torch.dropout(x, 0.3, True)
        x = self.block4(x)
        x = torch.softmax(x, dim=1)
        x = self.fc5(x)
        return x