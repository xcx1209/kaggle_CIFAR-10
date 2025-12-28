import torch
import torchvision
from torchvision import transforms

#数据预处理
#图像转化为张量，并归一化到[-1,1]
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5,))])

trn_dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
tst_dataset = torchvision.datasets.ImageFolder(root='./data/test', transform=transform)
