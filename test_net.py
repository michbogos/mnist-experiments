import matplotlib
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch import load

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

class MuhNet(nn.Module):
    def __init__(self):
        super(MuhNet, self).__init__()
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.relu1 = nn.ReLU()

        #self.conv1 = nn.Conv2d(1, 1, 4, 1)
        self.linear1 = nn.Linear(28*28, 10*10)
        self.linear2 = nn.Linear(10*10, 10)
    
    def forward(self, x):
        #x = self.conv1(x)
        #x = self.sigmoid1(x)
        x = torch.flatten(x, 1)
        x = self.sigmoid1(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        return x

img = Image.open("two.png").convert("L")

toTensor = transforms.ToTensor()

tensor = toTensor(img)

print(tensor.shape)

plt.imshow(tensor[0])
plt.show()

mnt = MuhNet()
mnt.load_state_dict(load("pretrained/21_1000"))

mnt.eval()

print(mnt(tensor))
