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

    def __init__(self, base_folder='./flower/', mode='gpu'):
        self.base_folder = base_folder
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.mode == 'gpu' else "cpu")

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

    def create_model(self, model='vgg16', hidden_layer=256, learning_rate=0.001):

        models = ['vgg16', 'densenet121', 'alexnet']

        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
        else:
            print(f'Please choose one of these models: {models}')

        classifier_input = model.classifier[0].in_features
        classifier_output = len(self.names)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(classifier_input, hidden_layer),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_layer, classifier_output),
                                         nn.LogSoftmax(dim=1))

        criterion = nn.NLLLoss()

        optimizer = optim.Adam(model.classifier.parameters(), learning_rate=0.001)

        model.to(self.device)

        return model, optimizer, criterion

    def train_validate(self, optimizer, model, criterion, train_loader, valid_loader, epochs=3):

        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            training_loss = round(running_loss / print_every, 3)

                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Valid loss: {valid_loss / len(valid_loader):.3f}.. "
                          f"Valid accuracy: {accuracy / len(valid_loader):.3f}")
                    running_loss = 0
                    model.train()

        print('Training Completed!!!')
