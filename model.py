# coding=gbk
import torch
from torch import nn

#构建LeNet-5网络
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=(5,5),stride=1,padding=0,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    
    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
     model = Model()
     input = torch.ones((64,3,32,32))
     output = model(input)
     print(output.shape)