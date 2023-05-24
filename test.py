import torch 
from Network.Net import Net
from dataloader import get_x_y

param_dic = torch.load('./output/params.ml')
net = Net()
net.load_state_dict(param_dic)

X, y = get_x_y('winequality-white.csv')
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long().add(-3)

if torch.cuda.is_available(): # 使用GPU，将数据载入cuda
    X, y = X.cuda(), y.cuda()
    net.cuda()

net.eval()
out = net(X)
pred = torch.max(out, 1)[1]
num_correct = (pred == y).sum()
print(f'Acc: {num_correct.item() / (len(y)) * 100:.2f}%')