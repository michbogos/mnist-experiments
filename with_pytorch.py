import matplotlib
from numpy import float32
from sklearn.utils import shuffle
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim import Adadelta
matplotlib.use("TkAgg")

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler

class MuhNet(nn.Module):
    def __init__(self):
        super(MuhNet, self).__init__()
        self.logsigmoid1 = nn.LogSigmoid()
        self.relu1 = nn.ReLU()
        #self.relu2 = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 1, 4, 1)
        self.linear1 = nn.Linear(625, 10*10)
        self.linear2 = nn.Linear(10*10, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        #x = self.sigmoid1(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.logsigmoid1(x)
        return x


def train(model : MuhNet, device, optimizer : torch.optim.Optimizer, epoch, dataloader):
    model.train(True)
    for batch, (data, target) in enumerate(dataloader):
        #target = torch.zeros(len(label), 10)
        #for i in range(len(target)):
        #    target[i][label[i]] = 1.0
        #print(label, target)
        #print(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()
        cross_entropy = nn.CrossEntropyLoss()
        loss = cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch % 1000 == 0:
            #print(output, target)
            torch.save(model.state_dict(), "./pretrained/{}_{}".format(epoch, batch))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(data), len(dataloader.dataset),
                100. * batch / len(dataloader), loss.item()))

def test(model, loader, device):
    loss = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))



dataset = MNIST("./MNIST_dataset", train=True, download=True, transform=
                                                            transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.1307,), (0.3081,))]))

dataset_test = MNIST("./MNIST_dataset", train=False, download=True, transform=
                                                            transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.1307,), (0.3081,))]))

train_dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=True)

#plt.imshow(dataset.train_data[0])

#plt.show()

cpu = torch.device("cpu")

nt = MuhNet().to(cpu)

anabelle = Adadelta(nt.parameters(),lr=1e-3)

print(dataset.data[0].float())

scheduler = lr_scheduler.StepLR(anabelle, step_size=1, gamma=0.01)

for epoch in range(11):
    train(nt, cpu, anabelle, epoch, dataloader=train_dataloader)
    test(nt, test_dataloader, cpu)
    scheduler.step()

