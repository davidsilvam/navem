import os
import sys
from libnavem.common_flags import FLAGS
import libnavem.utils as utils
from libnavem import datasets

from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json

from absl import app
from absl import flags

import numpy as np

def main(argv):
    print("asdfas")
    weights_name = "model_weights_259.h5"
    exp_name = "exp_029"
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
    model.compile(loss='mse', optimizer='adam')

if __name__ == "__main__":
    app.run(main)