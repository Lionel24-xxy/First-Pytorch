# -*- coding: gbk -*-
import imp
from PIL import Image
import torch
import torchvision
from model import *


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#image_path = "dog.jpg"
#image_path = "airplane.jpg"
image_path = "ship.jpg"
img = Image.open(image_path)
#print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(img)
#print(image.shape)

LeNet = torch.load("LeNet.pth",map_location=torch.device("cpu"))
image = torch.reshape(image,(1,3,32,32))
LeNet.eval()
with torch.no_grad():
    output = LeNet(image)
    
#print(output.argmax(1).item())
label = output.argmax(1).item()
print(classes[label])
img.show()