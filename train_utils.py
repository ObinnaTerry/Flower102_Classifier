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

	def __init__(self, base_folder = './flower/'):
		self.base_folder = base_folder