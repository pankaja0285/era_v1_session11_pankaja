import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from dataloader.albumentation import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import notebook
from PIL import Image
import os
import requests
import zipfile
from io import BytesIO
import glob
import csv
import numpy as np
import random

class Cifar10DataLoader:
    def __init__(self, config):
        self.config = config
        self.augmentation = config['data_augmentation']['type']
        
    def calculate_mean_std(self):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return mean, std

    def classes(self):
        return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], None
        
    def get_dataloader(self):
        train_batch_size = self.config['data_loader']['args']['train_batch_size']
        test_batch_size = self.config['data_loader']['args']['test_batch_size']
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
        return self.train_loader, self.test_loader
    
class Cifar10DataLoader_Alb:
    def __init__(self, config):
        self.config = config
        self.augmentation = config['data_augmentation']['type']
        
    def calculate_mean_std(self):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return mean, std

    def classes(self):
        return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], None
        
    def get_dataloader(self):
        cifar_albumentation = eval(self.augmentation)()
        mean,std = self.calculate_mean_std()
        
        train_batch_size = self.config['data_loader']['args']['train_batch_size']
        test_batch_size = self.config['data_loader']['args']['test_batch_size']
        
        horizontalflip_prob = self.config['data_augmentation']['args']['horizontalflip_prob']
        rotate_limit = self.config['data_augmentation']['args']['rotate_limit']
        shiftscalerotate_prob = self.config['data_augmentation']['args']['shiftscalerotate_prob']
        num_holes = self.config['data_augmentation']['args']['num_holes']
        cutout_prob = self.config['data_augmentation']['args']['cutout_prob']
        cutout_size = self.config['data_augmentation']['args']['cutout_size']
        cutout_h = cutout_size
        cutout_w = cutout_size
        
        # params mean, std, max_holes=1, min_holes=1, max_height=16, min_height=16,
        # max_width=16, min_width=16, cutout_prob=0.5
        transform_train = cifar_albumentation.train_transform(mean, std,
        max_holes=1, min_holes=1, max_height=cutout_h, min_height=cutout_h,
        max_width=cutout_w, min_width=cutout_w, cutout_prob=cutout_prob)
        # horizontalflip_prob, rotate_limit, shiftscalerotate_prob
        transform_test = cifar_albumentation.test_transform(mean, std)
                         
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
        return self.train_loader, self.test_loader
    