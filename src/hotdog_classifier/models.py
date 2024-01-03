import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

def block(input:int,output:list,enable_dropout=True,kernel_size=3,padding=1,dropout_rate=0.25):
    out = [
        nn.Conv2d(input,output[0],kernel_size=kernel_size,padding=padding),
        nn.ReLU(),
        nn.BatchNorm2d(output[0]),
        nn.Conv2d(output[0],output[1],kernel_size=kernel_size,padding=padding),
        nn.ReLU(),
        nn.BatchNorm2d(output[1]),
        nn.MaxPool2d(2, 2),
    ]
    if enable_dropout:
        out.append(nn.Dropout(dropout_rate))
    out = nn.Sequential(*out)
    return out

class BASIC_CNN(nn.Module):
    def __init__(self,n_classes=1):
        super().__init__()
        self.block1 = block(3,[256,128]) # 32
        self.block2 = block(128,[128,64]) # 16
        self.block3 = block(64,[64,32]) # 8
        self.block4 = block(32,[32,16]) # 4

        # # Linear Layers
        self.fc1 = nn.Linear(16 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x):

        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)

        # # Linear Layers
        x = torch.flatten(f4, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



