import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
# from torch_cv_wrapper.dataloader.albumentation import *
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
        
        # cifar_albumentation = eval(self.augmentation)()
        # mean,std = self.calculate_mean_std()
        
        # train_transforms, test_transforms = cifar_albumentation.train_transform(mean,std),cifar_albumentation.test_transform(mean,std)
                                                                              
        # trainset = datasets.CIFAR10(root='./data', train=True,
        #                                     download=True, transform=train_transforms)  
            
        # testset  = datasets.CIFAR10(root='./data', train=False,
        #                                      transform=test_transforms)

        # self.train_loader = torch.utils.data.DataLoader(trainset, 
        #                                               batch_size=self.config['data_loader']['args']['batch_size'], 
        #                                               shuffle=True,
        #                                               num_workers=self.config['data_loader']['args']['num_workers'], 
        #                                               pin_memory=self.config['data_loader']['args']['pin_memory'])
        # self.test_loader = torch.utils.data.DataLoader(testset, 
        #                                              batch_size=self.config['data_loader']['args']['batch_size'],  
        #                                              shuffle=False,
        #                                              num_workers=self.config['data_loader']['args']['num_workers'], 
        #                                              pin_memory=self.config['data_loader']['args']['pin_memory'])
        
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
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        return self.train_loader, self.test_loader
    