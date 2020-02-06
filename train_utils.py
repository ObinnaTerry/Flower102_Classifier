import os

import torch
from torch import nn
from torch import optim
from torchvision import transforms, models, datasets


class TrainUtils(Exception):

    def __init__(self, mode='gpu'):
        """Provides methods for training models on any set of labeled images.

        :param mode: can be gpu or cpu
        """

        Exception.__init__(self)

        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.mode == 'gpu' else "cpu")

    @staticmethod
    def data_loader(base_folder='./flowers/'):
        """Load and transform train, validation and test data-sets.

        This method will raise a Exception if the base_folder argument provided does not exist.
        :param base_folder: folder containing train, validation and test data
        :return: dataloader for train, validation and test as well as class to index mapper
        """

        if not os.path.exists(base_folder):
            raise Exception('Target folder can not be found')  # raise exception is base_folder is not found

        train_dir = base_folder + '/train'
        valid_dir = base_folder + '/valid'
        test_dir = base_folder + '/test'

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

        return train_loader, valid_loader, test_loader, train_data.class_to_idx

    def create_model(self, cl_output, model_name='vgg16', hidden_layer=256, learning_rate=0.001):
        """Create model from input arguments.

        Supports only three pre-trained models: 'vgg16','densenet121' and'alexnet'.
        :param cl_output: number of output predictions from the model. (type: int)
        :param model_name: name of pre-trained model.
        :param hidden_layer: number of hidden layer. (type: int)
        :param learning_rate: learning rate for model. (type: float)
        :return: model, optimizer, criterion
        """

        models_list = ['vgg16', 'densenet121', 'alexnet']

        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
        else:
            print(f'Please choose one of these models: {models_list}')

        classifier_input = model.classifier[0].in_features

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(classifier_input, hidden_layer),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_layer, cl_output),
                                         nn.LogSoftmax(dim=1))

        criterion = nn.NLLLoss()

        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

        model.to(self.device)

        return model, optimizer, criterion

    def validation(self, model, criterion, test_loader):
        """Use for model validation.

        This method can be used for both validation and testing. When used for validation, the test_loader argument
        should point to a validation data. For testing, it should point to a test data.
        :param model: trained model to be used for validation or testing
        :param criterion: criterion to be used
        :param test_loader: folder containing test or validation data
        :return: accuracy and loss.
        """
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return round(accuracy / len(test_loader), 3), round(valid_loss / len(test_loader), 3)

    def train_validate(self, optimizer, model, criterion, train_loader, valid_loader, epochs=3):
        """Train model and performs validation.

        This method trains a model and performs validation after every 20 iteration of the train data set. Validation is
        performed using the validation method of this class. train loss, test loss and validation accuracy are printed
        after every 20 iteration of the train data.
        validation is performed is eval mode of the model.

        :param optimizer:
        :param model: model to be trained.
        :param criterion: criterion to be used.
        :param train_loader: folder containing training data set.
        :param valid_loader: folder containing validation data set
        :param epochs: number of training epochs. (default: 3; type: int)
        """
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

                    accuracy, validation_loss = self.validation(model, criterion, valid_loader)  # performs validation

                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Test loss: {validation_loss}.. "
                          f"Validation accuracy: {accuracy}")
                    running_loss = 0
                    model.train()

        print('Training Completed!!!')

    @staticmethod
    def save_model(model, class_to_idx, optimizer, file_path='./', model_name='checkpoint'):
        """Save a trained model.

        This method can invoked to save a trained model.
        :param model: Trained model to saved
        :param class_to_idx: class to index mapper of the model
        :param optimizer: optimizer used in training the model
        :param file_path: file path where moel should be saved. if no file path is provided, model will be saved in same
        folder as the python script that initiated it.
        :param model_name: named to be used in saving the model. (default:'checkpoint')
        """
        model.class_to_idx = class_to_idx

        print("Our model: \n\n", model, '\n')
        print("The state dict keys: \n\n", model.state_dict().keys())

        checkpoint = {'class_to_idx': model.class_to_idx,
                      'model': model,
                      'classifier': model.classifier,
                      'optimizer': optimizer.state_dict(),
                      'state_dict': model.state_dict()}

        torch.save(checkpoint, f'{file_path}//{model_name}.pth')

        print(f'Model successfully saved at {file_path}')
