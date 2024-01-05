import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision



def block(input:int,output:list,enable_dropout=True,kernel_size=3,padding=1,dropout_rate=0.25,apply_batchnorm=True):
    out = [nn.Conv2d(input,output[0],kernel_size=kernel_size,padding=padding)]
    if apply_batchnorm:
        out.append(nn.BatchNorm2d(output[0]))
    out.append(nn.ReLU())
    out.append(nn.Conv2d(output[0],output[1],kernel_size=kernel_size,padding=padding))
    if apply_batchnorm:
        out.append(nn.BatchNorm2d(output[1]))
    out.append(nn.ReLU())
    out.append(nn.MaxPool2d(2, 2))

    if enable_dropout:
        out.append(nn.Dropout(dropout_rate))
    out = nn.Sequential(*out)
    return out

class BASIC_CNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        training_args = config.models.training
        self.block1 = block(3,[256,128],enable_dropout=training_args.enable_dropout,dropout_rate=training_args.dropout_rate,apply_batchnorm=training_args.apply_batchnorm) 
        self.block2 = block(128,[128,64],enable_dropout=training_args.enable_dropout,dropout_rate=training_args.dropout_rate,apply_batchnorm=training_args.apply_batchnorm) 
        self.block3 = block(64,[64,32],enable_dropout=training_args.enable_dropout,dropout_rate=training_args.dropout_rate,apply_batchnorm=training_args.apply_batchnorm) 
        self.block4 = block(32,[32,16],enable_dropout=training_args.enable_dropout,dropout_rate=training_args.dropout_rate,apply_batchnorm=training_args.apply_batchnorm) 
        # # Linear Layers
        img_dim_last_block_width = int(config.data_params.image_width/(2**4))
        img_dim_last_block_height= int(config.data_params.image_height/(2**4))
        self.fc1 = nn.Linear(16 * img_dim_last_block_width * img_dim_last_block_height, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, config.n_classes)

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

class Pretrained(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.training_args = config.models.training
        if self.config.models.name == 'efficientnet':
            self.model = timm.create_model('efficientnet_b4', pretrained=self.training_args.pretrained, num_classes=config.n_classes,drop_rate=self.training_args.dropout_rate if self.training_args.enable_dropout else 0)
            self.freeze_weights()
        elif self.config.models.name == 'googlenet':
            self.model = torchvision.models.googlenet(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc = nn.Linear(self.model.fc.in_features, config.n_classes)

    def forward(self,x):
        return self.model(x)

    def freeze_weights(self):
        if self.training_args.freeze_params == 1.0:
            for param_name,param in self.model.named_parameters():
                if 'classifier' not in param_name:
                    param.requires_grad = False
            
            self.model.classifier.requires_grad_()
        elif self.training_args.freeze_params < 1.0:
            total_params = sum(p.numel() for p in self.model.parameters())
            params_to_freeze = int(total_params * self.training_args.freeze_params)
            frozen_params = 0
            for param in self.model.parameters():
                if frozen_params < params_to_freeze:
                    param.requires_grad = False
                    frozen_params += param.numel()
                else:
                    param.requires_grad = True
            print(f" {frozen_params/total_params:2f}% of the model weights are frozen, corresponding to {frozen_params}/{total_params} parameters")