import os
import sys
from libnavem.common_flags import FLAGS
import libnavem.utils as utils
from libnavem import datasets

from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image

from absl import app
from absl import flags


import numpy as np

def main(argv):
    print("asdfas")
    weights_name = "model_weights_99.h5"
    exp_name = "exp_313"
    weights_load_path = os.path.join(".\..", "experiments", exp_name, weights_name)

    # Load json and create model
    json_model_path = os.path.join(".\..\experiments", exp_name, "model_struct.json")
    print(json_model_path)
    model = utils.jsonToModel(json_model_path)

    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    img_width, img_height = 224, 224
    dataset_name = "indoor_accy_flipped_232_pc_dataset_00"
    path_img = os.path.join(".\..\datasets", "vgg16", dataset_name, dataset_name, "train", dataset_name, "images", "000007.jpg")

    img = image.load_img(path_img, target_size=(img_width, img_height), color_mode='rgb')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    np.set_printoptions(suppress=True)
    images = np.vstack([x])
    classes = model.predict(x, batch_size=64)

    print(np.argmax(classes))

if __name__ == "__main__":
    app.run(main)