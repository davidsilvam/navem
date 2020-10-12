# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from libnavem.cnn_models import create_cnn, resnet8, vgg16

def getModel(img_width, img_height, img_channels, output_dim, weights_path):
    """
    Initialize model.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       weights_path: Path to pre-trained model.

    # Returns
       model: A Model instance.
    """
    model = resnet8(img_width, img_height, img_channels, output_dim=output_dim) #change model from train resnet or vgg16
    print("\n\n\n" + weights_path + "\n\n\n")
    a = input("ante de get")
    if weights_path:
        a = input("GET MODEL")
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model

def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))

	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))

	# return our model
	return model
