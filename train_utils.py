import matplotlib.pyplot as plt
import numpy as np
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as FP
from torchvision import transforms, models, datasets


class TrainUtils:
    """Contains methods for training"""

    def __init__(self, base_folder='./flower/'):
        self.base_folder = base_folder

    def data_loader(self):
        train_dir = self.base_folder + '/train'
        valid_dir = self.base_folder + '/valid'
        test_dir = self.base_folder + '/test'

        # TODO: Define your transforms for the training, validation, and testing sets

        train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomResizedCrop(244),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        valid_tranform = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(244),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

        test_transform = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(244),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

        train_data = datasets.ImageFolder(train_dir, transform=train_transform)

        valid_data = datasets.ImageFolder(valid_dir, transform=valid_tranform)

        test_data = datasets.ImageFolder(test_dir, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

        return train_loader, valid_loader, test_loader

    @staticmethod
    def names():
        with open('cat_to_name.json', 'r') as file:
            cat_to_name = json.load(file)

        return cat_to_name
