# -*-coding:gbk-*-
import torch
from torch import nn, no_grad
import torchvision
from torch.utils.data import DataLoader
from model import *
from torch.utils.tensorboard import SummaryWriter
import time


#����ѵ�����豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#׼�����ݼ�
train_data = torchvision.datasets.CIFAR10('./data',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

#ʹ��DataLoader�������ݼ�
train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64)

#��������ṹ
LeNet = Model()
LeNet.to(device)

#��ʧ����
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#�Ż���
lr = 1e-3
optim = torch.optim.SGD(LeNet.parameters(),lr,momentum=0.9)

#ѵ������
train_step = 0
test_step = 0

#ѵ������
epoch = 50

#Tensorboard���ӻ�
writer = SummaryWriter('./log_train')

#��ʼ��ʱ
start_time = time.time()

for i in range(epoch):
    print(f'--------------��{i+1}��ѵ����ʼ--------------')
    
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
            print(f"ѵ������{train_step},Loss:{loss.item()}")
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
            
    print(f"���Լ��ϵ�Loss:{test_loss}")
    print(f"���Լ���ȷ��Accuracy:{accuracy / test_data_size}")
    writer.add_scalar("test_loss",test_loss,test_step)
    writer.add_scalar("test_accuracy",accuracy,test_step)
    test_step += 1
    
#ģ�ͱ���
torch.save(LeNet,"LeNet.pth") 
print("ģ���ѱ���")   
    
writer.close()
            


