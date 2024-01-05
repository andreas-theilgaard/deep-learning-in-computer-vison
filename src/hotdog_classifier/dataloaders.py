import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import PIL.Image as Image
from torchvision.datasets import ImageFolder
import glob 

import torch
from torchvision import transforms
from PIL import Image
class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02516/hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y

class normalize_data:
    def __init__(self,config):
        self.config = config

    def get_means_and_stds(self):
        width = self.config.image_width
        height = self.config.image_height

        train_transform = transforms.Compose([transforms.Resize((height, width)), 
                                            transforms.ToTensor()])
        batch_size = 1
        trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=self.config.workers)
        means = torch.stack([batch[0].mean(dim=[0,2,3]) for batch in train_loader]).mean(0)
        stds = torch.stack([batch[0][0].std(1).std(1) for batch in train_loader]).std(0)
        print("-----------------------------------------------")
        print(f"Calculating mean and std. dev. for train set")
        print(f"------Means: {means}------")
        print(f"------ Stds: {stds}------")
        print("-----------------------------------------------")
        self.means=means
        self.stds=stds
        return means,stds

    def denormalize(self,images):
        # norm = x-mean/std => x= norm*std + mean
        trans_inv = transforms.Compose([transforms.Normalize(mean=[0,0,0], std=1/self.stds),transforms.Normalize(mean=-self.means, std=[1,1,1])])
        return trans_inv(images)

def get_transformations(config):
    #size = config.image_size
    width = config.image_width
    height = config.image_height
    normalizer = None
    train_transformations = [transforms.Resize((height, width))]
    test_transformations = [transforms.Resize((height, width)),transforms.ToTensor()]

    if config.augment:
        train_transformations.append(transforms.RandomRotation(degrees=(10, 100)))
    if config.extra_augment:
        gaus_kernel = (5, 5)
        color_jit = [0.2, 0.15, 0.1, 0.15]
        train_transformations.append(transforms.GaussianBlur(gaus_kernel, sigma=(0.01, 2.0)))
        train_transformations.append(transforms.RandomHorizontalFlip(p=0.3))
        train_transformations.append(transforms.RandomVerticalFlip(p=0.3))
        train_transformations.append(transforms.ColorJitter(*color_jit))
    
    train_transformations.append(transforms.ToTensor())
    if config.normalize:
        normalizer = normalize_data(config=config)
        means,stds = normalizer.get_means_and_stds()
        train_transformations.append(transforms.Normalize(mean=means, std=stds))
        test_transformations.append(transforms.Normalize(mean=means, std=stds))

    return transforms.Compose(train_transformations),transforms.Compose(test_transformations),normalizer

def get_class_dist(loader):
    labels = torch.cat([labels for _, labels in loader], dim=0)
    labels_percent = (torch.bincount(labels)/len(labels)).tolist()
    return [f"{x:.2f}" for x in labels_percent]

def get_data(config):
    train_transform,test_transform,normalizer = get_transformations(config)

    batch_size = config.bs
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=config.workers)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=config.workers)
    print("-----------------------------------------------")
    print(f"Train dataset: {len(trainset)} samples, class distribution is {get_class_dist(train_loader)}")
    print(f"Test dataset: {len(testset)} samples class distribution is {get_class_dist(test_loader)}")
    print("-----------------------------------------------")
    return train_loader,test_loader,normalizer
