# Flower102_Classifier

This project uses pytorch pretrained deep learning models for the classification of 102 different species of flowers, the dataset can be found here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. <code>train.py</code> and <code>predict.py</code> provides command line application that can be trained on any set of labeled images and make predictions respectively. 

**train.py**
- trains a model based on input parameters and saves the trained model to a destination folder. 
	**input variables:**
	- model_name: name of model to be used. - only 3 pretrained models are suppported, they include:  <bdi>'vgg16'</bdi>, <bdi>'densenet121'</bdi>, and  <bdi>'alexnet'</bdi>. 
	- base_folder: path to folder containing dataset
	- mode: training mode, can be cpu or gpu
	- hidden_layer: number of hidden layer
	- learning rate
	- epochs
	- model_save_path: location to save trained model
	- model_save_name: name to use in saving trained model
- all input variables for train.py have predefined default values. To run train.py using default parameters, simply type <code>python train.py</code> on your command line
- to change a default input variable, e.g., to change the number of epochs to 5, type <code>python train.py --epochs 5</code>. type <code>python train.py -h</code> to see all input variables
- train.py also provides abilty to train on a gpu, use <code>--mode gpu</code> to train on gpu

**predict.py**
- takes input image and provides a prediction. <code>predict.py</code> is a command line application and can be executed in same way as <code>train.py</code> above
	**input variables:** 
	- image_path: path to image be used predicted
	- model_path: path to model to used for prediction
	- top_k: number of top predictions to display
	- map_to_name: maps classes to image names. set to <code>False</code> if you prefer <i>classes</i> as output
	- mode: prediction mode, can be cpu or gpu. default is cpu
- two input variables <i>model_path</i> and <i>image_path</i> must be provides during execution. 

**Dependencies**
- pytorch
- PIL
- numpy
- matplotlib
