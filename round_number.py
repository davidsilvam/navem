import os
import pandas as pd
import numpy as np

labels_name = "gyro"

round_sample = 4

dataset_name = "all_dataset_cn"
labels_dataset_name = "gyro_round"

set = "train"

video_directory_train = "./../AWS/datasets/dataset_sidewalk_224_224_cn/train/sidewalk"
video_directory_val = "./../AWS/datasets/dataset_sidewalk_224_224_cn/val/sidewalk"
images_directory = "./../AWS/datasets/dataset_sidewalk_224_224_cn"
dataset_directory = "./../datasets/vgg16"

#f = open(os.path.join(video_directory, labels_name + ".txt"))
data = pd.read_csv(os.path.join(dataset_directory, dataset_name, dataset_name, set, dataset_name, labels_name + ".txt"), sep=" ", engine="python", encoding="ISO-8859-1", header=None).values

dataset_file = open(os.path.join(dataset_directory, dataset_name, dataset_name, set, dataset_name, labels_dataset_name + ".txt"), "w+")

for i in data:
    #print(str(i[0]) + " " + str(round(i[1], round_sample)) + "\n")
    dataset_file.write(str(i[0]) + " " + str(i[1]).split(".")[0] + "." + str(i[1]).split(".")[1][:round_sample] + "\n")
dataset_file.close()
print('Finalizou')
