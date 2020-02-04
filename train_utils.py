import matplotlib.pyplot as plt
import numpy as np
import json
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as FP
from torchvision import transforms, models, datasets


class TrainUtils:
    """Contains methods for training"""

    def __init__(self, base_folder='./flower/', mode='gpu'):

        if not os.path.exists(base_folder):
            raise 'Target folder can not be found'
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

    def create_model(self, model_name='vgg16', hidden_layer=256, lr=0.001):

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

        optimizer = optim.Adam(model.classifier.parameters(), learning_rate=lr)

        model.to(self.device)

        return model, optimizer, criterion

    def validation(self, model, criterion, test_loader):

        valid_loss = 0
        accuracy = 0
                    
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

        return round(accuracy/len(test_loader), 3), round(valid_loss/len(test_loader), 3)

    def train_validate(self, optimizer, model, criterion, train_loader, valid_loader, epochs=3):

        steps = 0
        running_loss = 0
        print_every = 20
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

                    model.eval()

                    valid_loss = 0
                    accuracy = 0
                    
                    accuracy, test_loss = self.validation(model, criterion, test_loader)

                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Test loss: {test_loss}.. "
                          f"Test accuracy: {accuracy}")
                    running_loss = 0
                    model.train()

        print('Training Completed!!!')

        @staticmethod
        def save_model(model, train_loader, optimizer, file_path='./'):

            model.class_to_idx = train_data.class_to_idx

            print("Our model: \n\n", model, '\n')
            print("The state dict keys: \n\n", model.state_dict().keys())

            checkpoint = {'class_to_idx': model.class_to_idx,
                          'model':model,
                          'classifier': model.classifier,
                          'optimizer': optimizer.state_dict(),
                          'state_dict': model.state_dict()}

            torch.save(checkpoint, f'{file_path}/checkpoint.pth')

            print(f'Model successfully saved at {file_path}')

