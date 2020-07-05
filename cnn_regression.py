# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
import tensorflow as tf
from keras import optimizers
#inicio alteração
from keras import optimizers
#fim alteração
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from libnavem import datasets
from libnavem import models
import libnavem.utils as utils
import numpy as np
import pandas as pd
#import argparse
import locale
import os

import sys
from absl import app
from absl import flags

from libnavem.common_flags import FLAGS
import libnavem.log_utils as log_utils
import libnavem.logz as logz


#from libnavem.cnn_models import create_cnn

#python3 cnn_regression.py --exp_name="exp_001" --dataset="dataset_cel_log_resized_200_200_gray" --img_mode="rgb" --batch_size=64
#python3 cnn_regression.py --exp_name="exp_004" --dataset="dataset_navem_224_224" --img_mode="rgb" --batch_size=64
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64

def trainModel(train_data_generator, val_data_generator, model, initial_epoch):

    # Initialize loss weights
	#model.alpha = tf.Variable(1, trainable=False, name='alpha', dtype=tf.float32)
	#model.beta = tf.Variable(0, trainable=False, name='beta', dtype=tf.float32)

	# Initialize number of samples for hard-mining
	model.k_mse = tf.Variable(FLAGS.batch_size, trainable=False, name='k_mse', dtype=tf.int32)
	#model.k_entropy = tf.Variable(FLAGS.batch_size, trainable=False, name='k_entropy', dtype=tf.int32)

	##opt = Adam(lr=1e-3, decay=1e-3 / 200)
	##model.compile(loss="mean_squared_error", optimizer=opt)
	optimizer = optimizers.Adam(decay=1e-5)
	# Configure training process
	model.compile(loss="mean_squared_error", optimizer=optimizer)

	# Save model with the lowest validation loss
	weights_path = os.path.join("./experiments/" + FLAGS.exp_name, 'weights_{epoch:03d}.h5')#Parou aqui
	writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True, period=50)

	logz.configure_output_dir("./experiments/" + FLAGS.exp_name)
	saveModelAndLoss = log_utils.MyCallback(filepath="./experiments/" + FLAGS.exp_name, period=FLAGS.log_rate, batch_size=FLAGS.batch_size)

	# train the model
	print("[INFO] training model...")
#	print(len(train_data_generator[0]), len(train_data_generator[1]), train_data_generator[1][0])
	#history = model.fit(train_data_generator[0], train_data_generator[1],
			##validation_data=val_data_generator,
			##epochs=FLAGS.epochs, batch_size=8,
			##callbacks=[writeBestModel, saveModelAndLoss],
			##initial_epoch=initial_epoch)

#	convert the history.history dict to a pandas DataFrame:
	##hist_df = pd.DataFrame(history.history)

	##hist_csv_file = os.path.join("./experiments/" + FLAGS.exp_name, 'history.csv')#Parou aqui
	##with open(hist_csv_file, mode='w') as f:
		##hist_df.to_csv(f)

#Test trainig with bath
	steps_per_epoch = int(np.ceil(train_data_generator.samples / FLAGS.batch_size))
	validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))
	history = model.fit_generator(train_data_generator,epochs=FLAGS.epochs,steps_per_epoch = steps_per_epoch,callbacks=[writeBestModel, saveModelAndLoss],validation_data=val_data_generator,validation_steps = validation_steps,initial_epoch=initial_epoch)

	##hist_df = pd.DataFrame(history.history)

	hist_csv_file = os.path.join("./experiments/" + FLAGS.exp_name, 'history.csv')#Parou aqui
	with open(hist_csv_file, mode='w') as f:
		hist_df.to_csv(f)

def main(argv):
	if not os.path.exists("./experiments/" + FLAGS.exp_name):
		os.makedirs("./experiments/" + FLAGS.exp_name)

    # Image mode
	if FLAGS.img_mode=='rgb':
		img_channels = 3
	elif FLAGS.img_mode == 'grayscale':
		img_channels = 1
	else:
		raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")

	# Output dimension (one for steering and one for collision)
	output_dim = 1

	print("[INFO] loading attributes...")
	##inputPath = os.path.sep.join(["./datasets/labels", FLAGS.dataset + ".txt"])
	##df = datasets.load_navem_attributes(inputPath)

	# load the house images and then scale the pixel intensities to the range [0, 1]
	print("[INFO] loading images...")
	##images = datasets.load_navem_images(df)
	##images = images / 255.0

	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	#print("\n\n\n" + str(len(df)) + "| " + str(len(images)) + "\n\n\n")
	##split = train_test_split(df, images, test_size=0.20, random_state=42)
	#programPause = input("Press the <ENTER> key to continue save train and test...")
	##(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

	#Save train and test state
	##utils.write_divided_dataset(os.path.join("./experiments/" + FLAGS.exp_name), trainAttrX, testAttrX)

	#Nomalization
	#trainY = (trainAttrX["gyro_z"] - trainAttrX["gyro_z"].min()) / (trainAttrX["gyro_z"].max() - trainAttrX["gyro_z"].min())
	#testY = (testAttrX["gyro_z"] - testAttrX["gyro_z"].min()) / (testAttrX["gyro_z"].max() - testAttrX["gyro_z"].min())

	##train_generator = (trainImagesX, trainAttrX["gyro_z"])
	##val_generator = (testImagesX, testAttrX["gyro_z"])

	#Data augmentation FLAGS
	#rotation_range = 0.2
	#width_shift_range = 0.2
	#height_shift_range=0.2

	train_datagen = utils.NavemDataGenerator(rescale = 1./255, rotation_range = 0.2, width_shift_range = 0.2, height_shift_range=0.2)
	train_generator = train_datagen.flow_from_directory(os.path.sep.join(["./datasets", FLAGS.dataset, "train"]),
                                                        shuffle = True,
                                                        color_mode=FLAGS.img_mode,
                                                        target_size=(FLAGS.img_width, FLAGS.img_height),
														crop_size=(FLAGS.crop_img_width, FLAGS.crop_img_height),
                                                        batch_size = FLAGS.batch_size)
	#print(train_generator.samples)



	val_datagen = utils.NavemDataGenerator(rescale=1./255, rotation_range = 0.2, width_shift_range = 0.2, height_shift_range=0.2)
	#val_datagen.fit(testImagesX)
	val_generator = val_datagen.flow_from_directory(os.path.sep.join(["./datasets", FLAGS.dataset, "val"]),
                                                        shuffle = True,
                                                        color_mode=FLAGS.img_mode,
                                                        target_size=(FLAGS.img_width, FLAGS.img_height),
														crop_size=(FLAGS.crop_img_width, FLAGS.crop_img_height),
                                                        batch_size = FLAGS.batch_size)

	# Weights to restore
	weights_path = os.path.join("./experiments/" + FLAGS.exp_name, FLAGS.weights_fname)
	initial_epoch = 0

	if not FLAGS.restore_model:
		# In this case weights will start from random
		weights_path = None
	else:
        # In this case weigths will start from the specified model
		initial_epoch = FLAGS.initial_epoch

	# Define model
	model = models.getModel(FLAGS.img_width, FLAGS.img_height, img_channels, output_dim, weights_path)
	#model = models.create_cnn(64, 64, 3, regress=True)

    # Serialize model into json
	json_model_path = os.path.join("./experiments/" + FLAGS.exp_name, FLAGS.json_model_fname)
	utils.modelToJson(model, json_model_path)

    # Train model
	trainModel(train_generator, val_generator, model, initial_epoch)

if __name__ == "__main__":
    app.run(main)
