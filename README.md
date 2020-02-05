# Flower102_Classifier

This project uses pytorch pretrained deep learning models for the classification of 102 different species of flowers. train.py is a command line application can be trained on any set of labeled images. 

** train.py **
- trains a model based on input parameters and saves the trainee model to a destination folder. 
- all input variables for train.py have predefined default values. To run train.py using default parameters, simply type <code>python train.py</code> on your command line
- to change a default input variable, e.g., to change the number of epochs to 5, type <code>python train.py --epochs 5</code>. type <code>python train.py -h</code> to see all input variables
- only 3 pretrained models are suppported, they include:  'vgg16', 'densenet121', and  'alexnet'
- train.py also provides abilty to train on a gpu use <code>--mode gpu</code> to train on gpu