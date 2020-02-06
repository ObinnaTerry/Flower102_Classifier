from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import json

import torch
from torchvision import transforms


class PredictUtils(Exception):

    def __init__(self):
        """Load model, process and predict image.

        The class provides load_checkpoint method for loading a trained model. process_image method for precessing an
        input image for prediction. imshow method for displaying an image. predict method for making a prediction on an image.
        """

        Exception.__init__(self)

    @staticmethod
    def load_checkpoint(filepath):
        """Load a trained model.

        :param filepath: file path to a checkpoint of the model
        :return: model
        """
        if not os.path.exists(filepath):
            raise Exception('Target model could not be found')

        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])

        for parameter in model.parameters():
            parameter.requires_grad = False

        # model.eval()

        return model

    @staticmethod
    def process_image(image):
        """Processes image to be input into a model for prediction.

        Scales, crops, and normalizes a PIL image for a PyTorch model.
            :param image: file path to the image file to be processed.
            :return: a tensor of the processed image.
        """
        if not os.path.exists(image):
            raise Exception('Target image could not be found')

        image = Image.open(image)

        image_norm = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(244),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

        return image_norm(image)

    @staticmethod
    def imshow(image, ax=None, title=None):

        """Display image tensor.

        :param image: image tensor
        :param ax:
        :param title:
        :return:
        """

        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        print(image.shape)

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        ax.imshow(image)

        return ax

    @staticmethod
    def predict(image, model, top_k=5, json_file=None, mode='cpu'):

        """ Predict the class of an image using input model.

        Takes and input image and produces a top top_k prediction of the image class. if a json file mapping categories
        to names is provided, the output prediction is name dictionary of name: probability pairs,
        else output is a dictionary of category: probability pair.

        prediction can be made on a cpu or gpu. To make predictions on gpu, change mode to gpu.
        :param top_k: number of top prediction to produce
        :param image: image to be predicted. (type: tensor)
        :param model: model to be used for prediction
        :param json_file: json file containing the mapping of categories to names.
        :param mode:
        :return: dictionary of result
        """
        if mode == 'gpu':
            if not torch.cuda.is_available():
                raise Exception('You have no available gpu')
            device = torch.device("cuda")
            model.to(device)
            image = image.to(device)
        else:
            model.cpu()

        model.eval()

        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model.forward(image)

            ps = torch.exp(output)
            top_prob, top_class = ps.topk(top_k, dim=1)
        class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}

        classes = []

        for index in top_class.cpu().numpy()[0]:
            classes.append(class_to_idx_rev[index])

        probs, classes = top_prob.cpu().numpy()[0], classes
        if json_file:
            if not os.path.exists(json_file):
                raise Exception('Target json file cant be found')

            with open(json_file, 'r') as f:
                cat_to_name = json.load(f)

                if len(class_to_idx_rev) != len(cat_to_name):
                    raise Exception('Number of items in json file does not match the number of output from model')
                # length of items in json file should match number of network output

            names = []

            for cl in classes:
                names.append(cat_to_name[str(cl)])

            return dict(zip(names, probs))
        else:
            return dict(zip(classes, probs))
