from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten, Input, Activation, Dropout, Lambda
import numpy as np
import os
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50

exp_name = "exp_098"
path_model_json = os.path.join("..\\..\\experiments", exp_name, "model_struct.json")
path_model = os.path.join("..\\..\\experiments", exp_name, "weights_dronet.h5") # model_weights_149 exp_066

# load json and create model
json_file = open(path_model_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path_model)
print("Loaded model from disk")

# load model without output layer
# model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3)) #, include_top=False, input_shape=(224,224,3)
model = loaded_model
##model = ResNet50()

# print(len(model.layers))
# for layer in model.layers:
#     print(layer)
#
model.summary()

# print(len(model.layers))
# for layer in model.layers:
#     print(layer)

############# new dronet ################

input = Input(shape=(None, 200, 200, 3), name='input_1')
print("---------")

model._layers[1].batch_input_shape = (None, 200, 200, 1)


# model = Model(input=input, output=model.layers[1])
# model.layers[0] = input
# model.summary()

# newOutputs = model(input)
# newModel = Model(input, newOutputs)
#
# print("=======")
#
# model.summary()

print(model.input)

print(len(model.layers))
x = model.layers[0](input)
for layer in model.layers[:]:
    x = layer(x)

x.summary()

############33 dronet ###############

# model.layers.pop()
# model.layers.pop()
# model.layers.pop()
# model.layers.pop()
# model.layers.pop()
# model.layers.pop()
#
# ###input = Input(shape=(200, 200, 3), name='image_input')
#
# output_dim = 5
# model.get_layer('activation_1').name = 'activation_01'
# x = Flatten(name='flatten')(model.layers[-1].output)
# x = Activation('relu')(x)
# x = Dropout(0.5)(x)
#
# # #Steering channel
# steer = Dense(output_dim, activation="softmax")(x)
#
# model = Model(inputs=[model.input], outputs=steer)
#
# print(len(model.layers))
# for layer in model.layers:
#     print(layer)
#
#
# model.summary()

############### vgg16 ################
##output_dim = 5
##
##x = Flatten(name='flatten')(model.layers[-1].output)
##x = Dense(4096, activation='relu', name='fc1')(x)
##x = Dense(4096, activation='relu', name='fc2')(x)
##steer = Dense(output_dim, activation="softmax")(x)
##
##model = Model(inputs=[model.input], outputs=steer)
##
##model.summary()

########## resnet50 ################
##output_dim = 5
##
##x = Flatten(name='flatten')(model.layers[-1].output)
####x = Dense(4096, activation='relu', name='fc1')(x)
####x = Dense(4096, activation='relu', name='fc2')(x)
##steer = Dense(output_dim, activation="softmax")(x)
####
##model = Model(inputs=[model.input], outputs=steer)
####
##model.summary()

##Save model
model_json = model.to_json()
with open("model_struct.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("weights_dronet.h5")
print("Saved model to disk")

# freeze layers dronet

# model.get_layer('conv2d_1').trainable = False
# model.get_layer('conv2d_2').trainable = False
# model.get_layer('conv2d_4').trainable = False
# model.get_layer('conv2d_3').trainable = False

# freeze layers resnet50

##model.get_layer('res2a_branch2b').trainable = False
##model.get_layer('res2b_branch2b').trainable = False
##model.get_layer('res2c_branch2b').trainable = False
##model.get_layer('res3a_branch2b').trainable = False
##model.get_layer('res3b_branch2b').trainable = False
##model.get_layer('res3c_branch2b').trainable = False
##model.get_layer('res3d_branch2b').trainable = False
##model.get_layer('res3a_branch2b').trainable = False
##model.get_layer('res4a_branch2b').trainable = False

# freeze layers vgg16

##model.get_layer('block1_conv1').trainable = False
##model.get_layer('block1_conv2').trainable = False
##model.get_layer('block2_conv1').trainable = False
##model.get_layer('block2_conv2').trainable = False
##model.get_layer('block3_conv1').trainable = False
##model.get_layer('block3_conv2').trainable = False
##model.get_layer('block3_conv3').trainable = False
##
##initial_weights_layer_block1_conv1 = model.get_layer('block1_conv1').get_weights()
##initial_weights_layer_block1_conv2 = model.get_layer('block1_conv2').get_weights()
##initial_weights_layer_block2_conv1 = model.get_layer('block2_conv1').get_weights()
##initial_weights_layer_block2_conv2 = model.get_layer('block2_conv2').get_weights()
##initial_weights_layer_block3_conv1 = model.get_layer('block3_conv1').get_weights()
##initial_weights_layer_block3_conv2 = model.get_layer('block3_conv2').get_weights()
##initial_weights_layer_block3_conv3 = model.get_layer('block3_conv3').get_weights()
##
### fit
##
##final_weights_layer_block1_conv1 = model.get_layer('block1_conv1').get_weights()
##final_weights_layer_block1_conv2 = model.get_layer('block1_conv2').get_weights()
##final_weights_layer_block2_conv1 = model.get_layer('block2_conv1').get_weights()
##final_weights_layer_block2_conv2 = model.get_layer('block2_conv2').get_weights()
##final_weights_layer_block3_conv1 = model.get_layer('block3_conv1').get_weights()
##final_weights_layer_block3_conv2 = model.get_layer('block3_conv2').get_weights()
##final_weights_layer_block3_conv3 = model.get_layer('block3_conv3').get_weights()
##
### block1_conv1
##
##np.testing.assert_allclose(
##    initial_weights_layer_block1_conv1[0], final_weights_layer_block1_conv1[0]
##)
##np.testing.assert_allclose(
##    initial_weights_layer_block1_conv1[1], final_weights_layer_block1_conv1[1]
##)
##
### block1_conv2
##
##np.testing.assert_allclose(
##    initial_weights_layer_block1_conv2[0], final_weights_layer_block1_conv2[0]
##)
##np.testing.assert_allclose(
##    initial_weights_layer_block1_conv2[1], final_weights_layer_block1_conv2[1]
##)
##
### block2_conv1
##
##np.testing.assert_allclose(
##    initial_weights_layer_block2_conv1[0], final_weights_layer_block2_conv1[0]
##)
##np.testing.assert_allclose(
##    initial_weights_layer_block2_conv1[1], final_weights_layer_block2_conv1[1]
##)
##
### block2_conv2
##
##np.testing.assert_allclose(
##    initial_weights_layer_block2_conv2[0], final_weights_layer_block2_conv2[0]
##)
##np.testing.assert_allclose(
##    initial_weights_layer_block2_conv2[1], final_weights_layer_block2_conv2[1]
##)
##
### block3_conv1
##
##np.testing.assert_allclose(
##    initial_weights_layer_block3_conv1[0], final_weights_layer_block3_conv1[0]
##)
##np.testing.assert_allclose(
##    initial_weights_layer_block3_conv1[1], final_weights_layer_block3_conv1[1]
##)
##
### block3_conv2
##
##np.testing.assert_allclose(
##    initial_weights_layer_block3_conv2[0], final_weights_layer_block3_conv2[0]
##)
##np.testing.assert_allclose(
##    initial_weights_layer_block3_conv2[1], final_weights_layer_block3_conv2[1]
##)
##
### block3_conv3
##
##np.testing.assert_allclose(
##    initial_weights_layer_block3_conv3[0], final_weights_layer_block3_conv3[0]
##)
##np.testing.assert_allclose(
##    initial_weights_layer_block3_conv3[1], final_weights_layer_block3_conv3[1]
##)






