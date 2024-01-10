import os
import numpy as np
import glob
import PIL.Image as Image
import cv2
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
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.segmentation.lightning import Segmenter
from omegaconf import OmegaConf

config = OmegaConf.create({
    'img_size': 512, # org size mean 575 x 766
    'batch_size': 2, #6
    'seed': 42,
    'workers': 3, #3
    'lr': 1e-4,
    'epochs': 50,#50
    'loss': 'WeightedBCE', # BCE, WeightedBCE
    'set_seed': True,
    'device': 'cuda',
    'model': 'EncDec', #EncDec, SimpleUNET
    'use_wandb': True,
    'tag': 'REAL',
    'pos_total': None,
    'neg_total': None,
}) 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if config.set_seed:
    set_seed(config.seed)

class DRIVE(torch.utils.data.Dataset):
    def __init__(self,transform,idx_list):
        'Initialization'
        self.transform = transform
        self.data_path = "/dtu/datasets1/02516/DRIVE/training"
        self.image_paths = np.array(sorted(glob.glob(f"{self.data_path}/images/*.tif")))[idx_list]
        self.label_paths = np.array(sorted(glob.glob(f"{self.data_path}/1st_manual/*.gif")))[idx_list]

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = np.array(Image.open(image_path),dtype=np.float32)/255
        label = np.array(Image.open(label_path),dtype=np.float32)/255

        augmented_data = self.transform(image=image,mask=label)
        X,Y =augmented_data['image'],augmented_data['mask'].unsqueeze(0)
        return X,Y
    
def get_data():
    size = config.img_size
    train_transform = A.Compose([
                            A.Resize(size,size),
                            A.VerticalFlip(p=0.3), # 0.3
                            #A.HorizontalFlip(p=1.0),
                            #A.Rotate(p=1.0,limit=45),
                            ToTensorV2()
                        ], is_check_shapes=False) 

    val_test_transform = A.Compose([
                            A.Resize(size,size),
                            ToTensorV2()
                        ], is_check_shapes=False) 

    data_path = "/dtu/datasets1/02516/DRIVE/training"
    image_paths = sorted(glob.glob(f"{data_path}/images/*.tif"))
    train_idx,val_test_idx = train_test_split(list(range(len(image_paths))),train_size=0.6,random_state=config.seed)
    val_idx,test_idx = train_test_split(val_test_idx,train_size=0.5,random_state=42)

    trainset = DRIVE(transform=train_transform,idx_list=train_idx)
    valset = DRIVE(transform=val_test_transform,idx_list=val_idx)
    testset = DRIVE(transform=val_test_transform,idx_list=test_idx)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers,generator=torch.Generator().manual_seed(config.seed))
    val_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers,generator=torch.Generator().manual_seed(config.seed))
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers,generator=torch.Generator().manual_seed(config.seed))

    config.pos_total = sum([batch[1].sum().item() for batch in train_loader])
    config.neg_total = sum([(batch[1]==0).sum().item() for batch in train_loader])
    return train_loader,val_loader,test_loader
    
if __name__ == "__main__":
    train_loader,val_loader,test_loader = get_data()
    model = Segmenter(config)
    checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor=f"Val_dice",
            mode="max",
            filename="{epoch:02d}-{test_acc:.2f}",
        )
    WANDB = WandbLogger(
            name=f"Segmentation_{config.model}_{config.loss}",
            project='dtu_dlcv',
            tags=[config.tag],
            config=None
        )
    trainer = pl.Trainer(
        devices=-1, 
        accelerator=config.device, 
        max_epochs = config.epochs,
        log_every_n_steps =1,
        callbacks=checkpoint_callback,
        logger=WANDB if config.use_wandb else None,
    )

    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader) 
