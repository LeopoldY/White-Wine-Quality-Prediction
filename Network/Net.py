import torch
import torch.nn as nn
# 自定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(11, 16),
            nn.Sigmoid()
        )
        self.block2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.Tanh()
        )
        self.block3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh()
        )
        self.block4 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(8, 7)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc5(x)
        x = torch.log_softmax(x, 1)
        return x
    
