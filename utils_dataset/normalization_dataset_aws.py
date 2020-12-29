import os
import pandas as pd
import numpy as np

labels_name = "gyro"

round_number = False

dataset_name = "_gyro"
video_name = "images"
video_directory_train = "./../AWS/datasets/dataset_sidewalk_224_224_cn/train/sidewalk"
video_directory_val = "./../AWS/datasets/dataset_sidewalk_224_224_cn/val/sidewalk"
images_directory = "./../AWS/datasets/dataset_sidewalk_224_224_cn"
dataset_directory = "./../datasets"

#f = open(os.path.join(video_directory, labels_name + ".txt"))
data = pd.read_csv(os.path.join(video_directory_train, labels_name + ".txt"), sep=" ", engine="python", encoding="ISO-8859-1", header=None)
data_val = pd.read_csv(os.path.join(video_directory_val, labels_name + ".txt"), sep=" ", engine="python", encoding="ISO-8859-1", header=None)

data = data.append(data_val)
data = data.values
col = 1
data[:,col] = (data[:, col] - np.mean(data[:, col]))/np.std(data[:, col])#normalize (x - media) / desvio padrao

dataset_file = open(os.path.join(images_directory, dataset_name + ".txt"), "w+")

for i in data:
    #print(str(i[0]) + " " + str(i[1]) + "\n")
    dataset_file.write(str(i[0]) + " " + str(i[1]) + "\n")
dataset_file.close()
print('Finalizou')
