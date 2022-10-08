import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
general_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def get_svhn_test_loader(path="/ssd1/dataset", batch_size=64, num_workers=2):
    test_data = SVHN(root=path, split='test', download=True, transform=general_test_transform)
    test_queue_svhn = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return test_queue_svhn


def get_cifar_test_loader(path="/ssd1/dataset", batch_size=64, num_workers=2):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
    test_queue_cifar10 = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return test_queue_cifar10


def svhn_dataloaders(batch_size=128, data_dir='/ssd1/dataset', num_workers=2, aug=False, flatten=False):

    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]
    if flatten:
        test_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    test_transform = transforms.Compose(test_transform_list)

    if aug:
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD) #  SVHN's mean/std
        ]
        if flatten:
            train_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    else:
        train_transform_list = list(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    print('Dataset information: SVHN\t 90% of 73257 images for training \t 10%% images for validation\t')
    print('Data augmentation = randomcrop(32) + randomhorizontalflip')

    indice = list(range(73257))
    random.shuffle(indice)
    train_set = Subset(SVHN(root=data_dir, split='train', transform=train_transform, download=True), indice[:int(73257*0.9)])
    val_set = Subset(SVHN(root=data_dir, split='train', transform=test_transform, download=True), indice[int(73257*0.9):])
    test_set = SVHN(root=data_dir, split='test', transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar10_dataloaders(batch_size=128, data_dir='/ssd1/dataset', num_workers=2, aug=False, flatten=False):

    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]
    if flatten:
        test_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    test_transform = transforms.Compose(test_transform_list)

    if aug:
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ]
        if flatten:
            train_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    else:
        train_transform_list = list(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    print('Dataset information: CIFAR-10\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(batch_size=128, data_dir='/ssd1/dataset', num_workers=2, val_size=5000, flatten=False, aug=False):

    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]
    if flatten:
        test_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    test_transform = transforms.Compose(test_transform_list)

    if aug:
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ]
        if flatten:
            train_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
        train_transform = transforms.Compose(train_transform_list)
    else:
        train_transform_list = list(test_transform_list)

    print('Dataset information: CIFAR-100\t %d images for training \t %d images for validation\t'%(50000-val_size, val_size))
    print('10000 images for testing')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(50000-val_size)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(50000-val_size, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def mnist_dataloaders(batch_size=128, data_dir='/ssd1/dataset', num_workers=2, flatten=False):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))

    train_transform = transforms.Compose(transform_list)
    test_transform = transforms.Compose(transform_list)

    print('Dataset information: MNIST\t 90% of 73257 images for training \t 10%% images for validation\t')
    indice = list(range(60000))
    random.shuffle(indice)
    train_set = Subset(MNIST(data_dir, train=True, transform=train_transform, download=True), indice[:int(60000*0.9)])
    val_set = Subset(MNIST(data_dir, train=True, transform=test_transform, download=True), indice[int(60000*0.9):])
    test_set = MNIST(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
