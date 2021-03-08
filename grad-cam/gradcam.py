from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
import sys
import os
from keras.models import model_from_json

all_img = False
exp_name = "exp_094"
path_model_json = os.path.join("..\\..\\experiments", exp_name, "model_struct.json")
path_model = os.path.join("..\\..\\experiments", exp_name, "model_weights_199.h5") # model_weights_149 exp_066

# load json and create model
json_file = open(path_model_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path_model)
print("Loaded model from disk")

# model = VGG16(weights="imagenet")
model = loaded_model
#img_path = sys.argv[1]
print(model.summary())

if(all_img == True):
    directory = r'vgg\\train\\images'
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print(filename.split('\\')[0])

            set_n = "vgg\\train\\images"
            img_name = "000000.jpg"

            img_path = os.path.join(set_n, filename.split('\\')[0])
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            class_idx = np.argmax(preds[0])
            class_output = model.output[:, class_idx]
            last_conv_layer = model.get_layer("block5_conv3")

            grads = K.gradients(class_output, last_conv_layer.output)[0]
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

            pooled_grads_value, conv_layer_output_value = iterate([x])

            for i in range(512):
                conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

            heatmap = np.mean(conv_layer_output_value, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)

            img = cv2.imread(img_path)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            cv2.imshow("Original", img)
            cv2.imshow("GradCam", superimposed_img)
            cv2.waitKey(0)

        else:
            continue
else:
    set_n = "vgg\\train\\images"
    img_name = "000000.jpg"

    img_path = os.path.join(set_n, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("block5_conv3")

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imshow("Original", img)
    cv2.imshow("GradCam", superimposed_img)
    cv2.waitKey(0)