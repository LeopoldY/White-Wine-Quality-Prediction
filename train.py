import os
import argparse
import math
import torch
import shutil
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable 

from Network.Net import Net
from dataloader import load_from_csv

# 运行参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512) # batch size 一批处理的数据量
parser.add_argument('--epochs', type=int, default=3000) # 总的迭代次数
parser.add_argument('--use_cuda', type=int, default=False) # 是否使用GPU训练

args = parser.parse_args()
args.cuda = bool(args.use_cuda) and torch.cuda.is_available()

train_dataset, test_dataset = load_from_csv('winequality-white.csv', 0.2)

# 创建可供训练的数据 DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

# 实例化网络
net = Net()

# 训练函数
def train():
    if args.cuda: # 使用GPU，将网络载入cuda
        print(f'training on {torch.cuda.get_device_name(0)}')
        net.cuda()
    else: # 使用CPU
        print('training with CPU')
    
    if not os.path.exists("./output"): # 创建输出文件夹
        os.mkdir("./output")
    else:
        shutil.rmtree("./output")
        os.mkdir("./output")

    loss_func = nn.CrossEntropyLoss() # 定义损失函数为交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.7, weight_decay=1e-4) # 定义优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500)

    avgLoss_per_epoch = [] # 每一次迭代的平均训练损失
    acc_per_epoch = [] # 每一次迭代的训练准确率
    evalLoss_per_epoch = [] # 每一次迭代的平均验证损失
    eval_acc_epoch = [] # 每一次迭代的验证准确率

    # --------------------------开始训练------------------------
    for epoch in range(args.epochs): 
        net.train() # 将网络设置为训练模式
        train_loss = 0 # 当前迭代中训练损失，初始化为0
        train_acc = 0 # 当前迭代中预测正确的数量

        for _, (batch_x, batch_y) in enumerate(train_loader): # 批处理
            if args.cuda: # 使用GPU，将数据载入cuda
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else: # 使用CPU
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = net(batch_x) # 将一批数据导入网络，获得输出out，注意：这个不是预测值，而是输出的7个类别的概率分布
            loss = loss_func(out, batch_y) # 计算一批数据的损失
            train_loss += loss.item() # 将损失加到当前迭代的总损失中
            pred = torch.max(out, 1)[1] # 获得预测值
            train_correct = (pred == batch_y).sum() # 计算预测值中预测正确的数量
            train_acc += train_correct.item() # 将正确值加到当前迭代的总正确值中
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() 
        scheduler.step() # 更新learning rate
        avgLoss_per_epoch.append(train_loss / (math.ceil(len(train_dataset)/args.batch_size)))
        acc_per_epoch.append(train_acc / (len(train_dataset)))
        if (epoch+1) % 50 == 0:
            print('epoch: %2d/%d Train Loss: %.6f, Acc: %.3f' 
            % (epoch + 1, args.epochs, avgLoss_per_epoch[epoch], acc_per_epoch[epoch]), end=' | ')

        # ------------------------验证------------------------
        net.eval() # 将网络设置为验证模式
        eval_loss = 0 # 当前验证中验证损失，初始化为0
        eval_acc = 0 # 当前验证中预测正确的数量
        
        for batch_x, batch_y in test_loader:
            if args.cuda: # 使用GPU，将数据载入cuda
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else: # 使用CPU
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = net(batch_x) # 将一批数据导入网络，获得输出out，注意：这个不是预测值，而是输出的7个类别的概率分布
            loss = loss_func(out, batch_y) # 计算一批数据的损失
            eval_loss += loss.item() # 将损失加到当前验证的总损失中
            pred = torch.max(out, 1)[1] # 获得预测值

            num_correct = (pred == batch_y).sum() # 计算预测值中预测正确的数量
            eval_acc += num_correct.item() # 将正确值加到当前验证的总正确值中
        evalLoss_per_epoch.append(eval_loss / (math.ceil(len(test_dataset)/args.batch_size)))
        eval_acc_epoch.append(eval_acc / (len(test_dataset)))
        if (epoch+1) % 50 == 0:
            print('Val Loss: %.6f, Acc: %.3f' 
                  % (evalLoss_per_epoch[epoch], eval_acc_epoch[epoch]))
        if (epoch+1) % 100 == 0:
            torch.save(net.state_dict(), f'./output/epoch_{epoch+1}.pt')

    torch.save(net.state_dict(), f'./output/Model.pt')
    print("Model saved at: ./output/Model.pt")

    plt.plot(avgLoss_per_epoch, label = 'Train Loss')
    plt.plot(acc_per_epoch, label = 'Train Accuracy')
    plt.plot(evalLoss_per_epoch, label = 'Evaluation Loss')
    plt.plot(eval_acc_epoch, label = 'Evaluation Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()

