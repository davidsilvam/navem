import os
import sys
from libnavem.common_flags import FLAGS
import libnavem.utils as utils
from libnavem import datasets

from keras import backend as K

from absl import app
from absl import flags

import numpy as np

def main(argv):
    #python3 evaluation.py --exp_name="exp_005" --dataset="dataset_navem_224_224" --weights_fname="weights_050.h5" --batch_size=64
    #python3 evaluation.py --exp_name="exp_029" --dataset="\vgg16\sidewalk_accx" --weights_fname="model_weights_259.h5" --batch_size=64
    # Set testing mode (dropout/batchnormalization)

    ########### command aws #############
    #python3 evaluation.py --exp_name="exp_029" --dataset="sidewalk_accy_proportion_classes_20" --weights_fname="model_weights_259.h5" --batch_size=64

    TEST_PHASE = 0
    TRAIN_PHASE = 1

    K.set_learning_phase(TEST_PHASE)

    # Generate testing data
    test_datagen = utils.NavemDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(os.path.sep.join([".\\..\\datasets", "vgg16", FLAGS.dataset, FLAGS.dataset, "train"]),
                          shuffle=False,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                          crop_size=(FLAGS.crop_img_height, FLAGS.crop_img_width),
                          batch_size=FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(".\\..\\experiments", FLAGS.exp_name, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.join("./experiments/" + FLAGS.exp_name, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='mse', optimizer='adam') # change to "categorical_crossentropy"

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))

    predictions, ground_truth, t = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)

    print(predictions[1])
    print(ground_truth[1])

    # Param t. t=1 steering, t=0 collision
    #t_mask = t==1

    # Predicted and real steerings
    #pred_steerings = predictions[t_mask]
    #real_steerings = ground_truth[t_mask]


    ##print("[INFO] loading attributes...")
    ##inputPath = os.path.sep.join(["./datasets/labels", FLAGS.dataset + ".txt"])
    ##df = datasets.load_navem_attributes(inputPath)

    # load the house images and then scale the pixel intensities to the range [0, 1]
    ##print("[INFO] loading images...")
    ##images = datasets.load_navem_images(df)
    ##images = images / 255.0

    #print(np.array([images[0]]).shape)
    #programPause = input("Press the <ENTER> key to continue save train and test...")
    ##img = 406
    ##score = model.predict(np.array([images[img]]))
    ##print(df["path_images"][img], df["gyro_z"][img], score)

if __name__ == "__main__":
    app.run(main)

# python3 evaluation.py --exp_name="exp_081" --dataset="sidewalk_accy_proportion_classes_20" --weights_fname="model_weights_299.h5" --batch_size=64