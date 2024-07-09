import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.transforms import (
    apply_transformations,
    get_transforms_list,
    RandomAugmentation,
)

import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


import matplotlib.pyplot as plt
import numpy as np

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(2)
    denormalized_tensor = tensor * std + mean
    denormalized_tensor.clamp_(0, 1)
    return denormalized_tensor


def plot_images_stacked(tensor1, tensor2):
    
    _, _, h, w = tensor1.shape
    
    tensor1 = tensor1.cpu()
    tensor2 = tensor2.cpu()
    # Check if the input tensors have the correct shape
    expected_shape = (3, h, h)
    if tensor1.shape[1:] != expected_shape or tensor2.shape[1:] != expected_shape:
        raise ValueError("Input tensors must have shape (N, 3, H, W)")

    # Denormalize tensors
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    tensor1 = denormalize(tensor1, mean, std)
    tensor2 = denormalize(tensor2, mean, std)

    # Set up the figure for subplots
    fig, axes = plt.subplots(2, tensor1.shape[0], figsize=(tensor1.shape[0]*4, 8))

    # Plot images from the first tensor
    for image_index in range(tensor1.shape[0]):
        image_data = tensor1[image_index].permute(1, 2, 0)  # Transpose to (32, 32, 3) for RGB
        axes[0, image_index].imshow(image_data)
        axes[0, image_index].axis('off')

    # Plot images from the second tensor
    for image_index in range(tensor2.shape[0]):
        image_data = tensor2[image_index].permute(1, 2, 0)  # Transpose to (32, 32, 3) for RGB
        axes[1, image_index].imshow(image_data)
        axes[1, image_index].axis('off')

    plt.show()


def collate_fn(img1_img2_y_details):
    imgs1, imgs2, targets, details = [], [], [], []
    for img1, img2, y, (img1_details, img2_details) in img1_img2_y_details:
        imgs1.append(img1.unsqueeze(0))
        imgs2.append(img2.unsqueeze(0))
        targets.append(y)
        details.append( (img1_details, img2_details) )
        
    imgs1 = torch.concat(imgs1)
    imgs2 = torch.concat(imgs2)
    targets = torch.tensor(targets, dtype=torch.long)
        
    return imgs1, imgs2, targets, details


class MyDataset(Dataset):
    def __init__(self, train_dataset, args, transform=True, normalize=None, random_resized_crop=None):
                
        self.train_dataset = train_dataset
        self.transform = transform
        
        self.random_policy = RandomAugmentation(N=2, pr=0.8)
        # self.random_policy = transforms.Compose([
        #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     # transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * 32))], p=0.5)
        # ])        
        
        self.last_transform = transforms.Compose([
            random_resized_crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        
        self.gray = transforms.RandomGrayscale(p=0.2)


    def __len__(self,):
        return len(self.train_dataset)
    
    def __getitem__(self, i):
        img, y = self.train_dataset[i][0], self.train_dataset[i][1]
        
        if not self.transform:
            return img, y
        
        img1, img2 = img.copy(), img.copy()
        
        img1, img1_details = self.random_policy(img1)
        img2, img2_details = self.random_policy(img2)  
        
        img1 = self.last_transform(img1)
        img2 = self.last_transform(img2)

        y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
        
        return img1, img2, y, (img1_details, img2_details)



def get_dataloader(args, batch_size, transform=True):        
    
    if args.dataset in ['cifar10', 'svhn', 'cifar100'] :
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        random_resized_crop = transforms.RandomResizedCrop(32, scale=(0.2, 1.))
        
        if args.dataset == 'cifar10':
            dataset = torchvision.datasets.CIFAR10('./dataset/cifar10/')
        elif args.dataset == 'cifar100':
            dataset = torchvision.datasets.CIFAR100('./dataset/cifar100/')
        elif args.dataset == 'svhn':
            dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='train')

    elif args.dataset in ['TinyImagenet', 'stl10']:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if args.dataset == 'TinyImagenet':
            random_resized_crop = transforms.RandomResizedCrop(64, scale=(0.2, 1.))
            dataset = torchvision.datasets.ImageFolder('dataset/tiny-imagenet-200/train')
        elif args.dataset == 'stl10':
            random_resized_crop = transforms.RandomResizedCrop(96, scale=(0.2, 1.))
            dataset = torchvision.datasets.STL10('./dataset/STL10/', split='train+unlabeled')
            # dataset = torchvision.datasets.STL10('./dataset/STL10/', split='train')

    dataset = MyDataset(
        train_dataset=dataset,
        args=args,
        transform=transform,
        normalize=normalize,
        random_resized_crop=random_resized_crop,
    )
    
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    
    return data_loader




def get_knn_evaluation_loader(dataset_name, batch_size=512):
    
    if dataset_name in ['cifar10', 'svhn', 'cifar100']:
        single_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        if dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10('./dataset/cifar10/', train=True, transform=single_transform)
            test_dataset = torchvision.datasets.CIFAR10('./dataset/cifar10/', train=False, transform=single_transform)
        elif dataset_name == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100('./dataset/cifar100', train=True, transform=single_transform)
            test_dataset = torchvision.datasets.CIFAR100('./dataset/cifar100', train=False, transform=single_transform)
        elif dataset_name == 'svhn':
            train_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='train', transform=single_transform)
            test_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='test', transform=single_transform)
    
    
    elif dataset_name in ['TinyImagenet', 'stl10']:
        single_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if dataset_name == 'TinyImagenet':
            train_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/train', transform=single_transform)
            test_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/val/images', transform=single_transform)
        elif dataset_name == 'stl10': 
            train_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='train', transform=single_transform)
            test_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='test', transform=single_transform)
        
        
        
        
    
    memory_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
        
    return memory_loader, test_loader



def get_linear_evaluation_loader(dataset_name, batch_size):
    
    if dataset_name in ['cifar10', 'svhn', 'cifar100']:

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(36, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        if dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10('./dataset/cifar10/', train=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR10('./dataset/cifar10/', train=False, transform=test_transform)
        elif dataset_name == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100('./dataset/cifar100', train=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR100('./dataset/cifar100', train=False, transform=test_transform)
        elif dataset_name == 'svhn':
            train_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='train', transform=train_transform)
            test_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='test', transform=test_transform)
    

    
    elif dataset_name == 'TinyImagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(72, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/train', transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/val/images', transform=test_transform)
    
    
    elif dataset_name == 'stl10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(108, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='train', transform=train_transform)
        test_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='test', transform=test_transform)
        
        
        
    
    memory_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
        
    return memory_loader, test_loader