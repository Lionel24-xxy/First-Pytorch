# -*-coding:gbk-*-
import torch
from torch import nn, no_grad
import torchvision
from torch.utils.data import DataLoader
from model import *
from torch.utils.tensorboard import SummaryWriter
import time


#定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#准备数据集
train_data = torchvision.datasets.CIFAR10('./data',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

#使用DataLoader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64)

#构建网络结构
LeNet = Model()
LeNet.to(device)

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#优化器
lr = 1e-3
optim = torch.optim.SGD(LeNet.parameters(),lr,momentum=0.9)

#训练次数
train_step = 0
test_step = 0

#训练轮数
epoch = 50

#Tensorboard可视化
writer = SummaryWriter('./log_train')

#开始计时
start_time = time.time()

for i in range(epoch):
    print(f'--------------第{i+1}轮训练开始--------------')
    
    LeNet.train()
    for data in train_dataloader:
        imgs , targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = LeNet(imgs)
        loss = loss_fn(outputs,targets)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_step += 1
        if (train_step % 100 == 0):
            end_time = time.time()
            print(end_time - start_time,'s')
            print(f"训练次数{train_step},Loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),train_step)
            
            
    LeNet.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs , targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = LeNet(imgs)
            loss = loss_fn(outputs,targets)
            test_loss += loss.item()
            accuracy += (outputs.argmax(1) == targets).sum()
            
    print(f"测试集上的Loss:{test_loss}")
    print(f"测试集正确率Accuracy:{accuracy / test_data_size}")
    writer.add_scalar("test_loss",test_loss,test_step)
    writer.add_scalar("test_accuracy",accuracy,test_step)
    test_step += 1
    
#模型保存
torch.save(LeNet,"LeNet.pth") 
print("模型已保存")   
    
writer.close()
            


