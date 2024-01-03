import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# This is should be replaced with the hotdog dataloader
# but still returning trainloader and testloader
def get_data(training_args):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if training_args.augment:
        augmentation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.3)
        ]
        )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=augmentation if training_args.augment else transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform,)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=training_args.bs,
                                            shuffle=True, num_workers=training_args.workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=training_args.bs,
                                            shuffle=False, num_workers=training_args.workers)

    return trainloader,testloader
