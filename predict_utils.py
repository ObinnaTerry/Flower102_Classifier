from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import json

import torch 
from torchvision import transforms



class PredictUtils(Exception):

	def __init__(self):
		Exception.__init__(self)
		

	@staticmethod
	def load_checkpoint(filepath):

		if not os.path.exists(filepath):
			print('hi')
			raise Exception('Target model could not be found')
    
		checkpoint = torch.load(filepath, map_location=lambda storage, loc:storage)

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

		''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        	returns an Numpy array
		'''

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

		"""Imshow for Tensor."""

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
	def predict(image, model, topk=5, map_to_names=True, mode='cpu'):
    
		''' Predict the class (or classes) of an image using a trained deep learning model.
		'''
		if mode == 'gpu':
			model.gpu()
		else:
			model.cuda()

		model.eval()

		image = image.unsqueeze(0)
	    
		with open('cat_to_name.json', 'r') as f:
			cat_to_name = json.load(f)
	    
		with torch.no_grad():
			output = model.forward(image)
	        
			ps = torch.exp(output)
			top_prob, top_class = ps.topk(topk, dim=1)
	        
		class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}
	    
		classes = []
	    
		for index in top_class.numpy()[0]:
			classes.append(class_to_idx_rev[index])
	    
		probs, classes = top_prob.numpy()[0], classes
	    
		if map_to_names:

			names = []

			for cl in classes:
				names.append(cat_to_name[str(cl)])
	        
			return dict(zip(names, probs))
		else:	    	
			return dict(zip(classes,probs))