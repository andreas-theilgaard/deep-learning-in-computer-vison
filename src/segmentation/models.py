import os
import numpy as np
import glob
import PIL.Image as Image
import cv2
# pip install torchsummary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time

import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


def get_model(config):
    if config.model == 'EncDec':
        return EncDec(config)
    elif config.model == 'SimpleUNET':
        return SimpleUNET(config)
    elif config.model == 'UNET':
        return UNET()

class EncDec(nn.Module):
    """
    Encoder-Decoder network for image segmentation.

    Args:
        None

    Attributes:
        enc_conv0 (nn.Conv2d): Convolutional layer for the first encoder block.
        pool0 (nn.MaxPool2d): Max pooling layer for downsampling.
        enc_conv1 (nn.Conv2d): Convolutional layer for the second encoder block.
        pool1 (nn.MaxPool2d): Max pooling layer for downsampling.
        enc_conv2 (nn.Conv2d): Convolutional layer for the third encoder block.
        pool2 (nn.MaxPool2d): Max pooling layer for downsampling.
        enc_conv3 (nn.Conv2d): Convolutional layer for the fourth encoder block.
        pool3 (nn.MaxPool2d): Max pooling layer for downsampling.
        bottleneck_conv (nn.Conv2d): Convolutional layer for the bottleneck block.
        upsample0 (nn.Upsample): Upsampling layer for the first decoder block.
        dec_conv0 (nn.Conv2d): Convolutional layer for the first decoder block.
        upsample1 (nn.Upsample): Upsampling layer for the second decoder block.
        dec_conv1 (nn.Conv2d): Convolutional layer for the second decoder block.
        upsample2 (nn.Upsample): Upsampling layer for the third decoder block.
        dec_conv2 (nn.Conv2d): Convolutional layer for the third decoder block.
        upsample3 (nn.Upsample): Upsampling layer for the fourth decoder block.
        dec_conv3 (nn.Conv2d): Convolutional layer for the final decoder block.

    Methods:
        forward(x): Performs forward pass through the network.

    """

    def __init__(self,config):
        super().__init__()

        inp_size = config.img_size//2
        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, inp_size, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(inp_size, inp_size, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(inp_size, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3

class SimpleUNET(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        inp_size = config.img_size // 2

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, inp_size, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(inp_size, inp_size, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(inp_size, inp_size, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(inp_size * 2, inp_size, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(inp_size * 2, inp_size, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(inp_size * 2, inp_size, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(inp_size * 2, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(self.pool0(e0)))
        e2 = F.relu(self.enc_conv2(self.pool1(e1)))
        e3 = F.relu(self.enc_conv3(self.pool2(e2)))
        e4 = self.pool3(e3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e4))
        
        # decoder
        d0 = F.relu(self.dec_conv0(torch.cat([self.upsample0(b), e3], dim=1)))
        d1 = F.relu(self.dec_conv1(torch.cat([self.upsample1(d0), e2], dim=1)))
        d2 = F.relu(self.dec_conv2(torch.cat([self.upsample2(d1), e1], dim=1)))
        d3 = self.dec_conv3(torch.cat([self.upsample3(d2), e0], dim=1))
        return d3


import torch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchsummary import summary


def create_block(in_channels:int,out_channels,padding:int=0):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding,stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding,stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return block


class UNET(nn.Module):
    def __init__(self, in_ch:int=3, out_ch:int=1,upsample_type='transpose'):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.factor = 2 if upsample_type == 'bilinear' else 1

        self.d1 = create_block(in_channels=in_ch,out_channels=64,padding=1)
        self.d2 = create_block(in_channels=64,out_channels=128,padding=1)
        self.d3 = create_block(in_channels=128,out_channels=256,padding=1)
        self.d4 = create_block(in_channels=256,out_channels=512,padding=1)
        self.bottleneck = create_block(in_channels=512,out_channels=1024//self.factor,padding=1)


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.u1_conv = create_block(in_channels=1024,out_channels=512//self.factor,padding=1)
        self.u2_conv = create_block(in_channels=512,out_channels=256//self.factor,padding=1)
        self.u3_conv = create_block(in_channels=256,out_channels=128//self.factor,padding=1)
        self.u4_conv = create_block(in_channels=128,out_channels=64,padding=1)

        if upsample_type == 'transpose':
            self.u1 = nn.ConvTranspose2d(1024, 512//self.factor, kernel_size=2, stride=2)
            self.u2 = nn.ConvTranspose2d(512, 256//self.factor, kernel_size=2, stride=2)
            self.u3 = nn.ConvTranspose2d(256, 128//self.factor, kernel_size=2, stride=2)
            self.u4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        elif upsample_type == 'bilinear':
            self.u1 = nn.Upsample(scale_factor=2,mode='bilinear')
            self.u2 = nn.Upsample(scale_factor=2,mode='bilinear')
            self.u3 = nn.Upsample(scale_factor=2,mode='bilinear')
            self.u4 = nn.Upsample(scale_factor=2,mode='bilinear')

        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        down1 = self.d1(x) 
        x = self.maxpool(down1)  

        down2 = self.d2(x)  
        x = self.maxpool(down2) 
        x=self.dropout(x)

        down3 = self.d3(x)  
        x = self.maxpool(down3)
        x=self.dropout(x)  

        down4 = self.d4(x)  
        x = self.maxpool(down4)
        x=self.dropout(x)  

        bottleneck = self.bottleneck(x) 

        # Decoder
        up1 = self.u1(bottleneck) 
        x = torch.cat([up1, down4], dim=1)  
        x = self.u1_conv(x)
        x=self.dropout(x)  

        up2 = self.u2(x) 
        x = torch.cat([up2, down3], dim=1)  
        x = self.u2_conv(x)
        x=self.dropout(x)  

        up3 = self.u3(x)  
        x = torch.cat([up3, down2], dim=1)  
        x = self.u3_conv(x)
        x=self.dropout(x) 

        up4 = self.u4(x) 
        x = torch.cat([up4, down1], dim=1)  
        x = self.u4_conv(x) 

        # #output layer
        output = self.out(x)
        return output
