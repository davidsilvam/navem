import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

phase = "test" 
exp = "exp_067"
var = 'accy'
name_weights = "model_weights_149.h5"
name_dataset = "sidewalk_" + var + "_proportion_classes_19"

error = 0.4

set = "val"

flag_classification = True
classe = 2

#load val
if(phase == "train"):
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_1_" + '.txt')
else:
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_0_" + '.txt')

df_r_val = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None, names=['pred', 'real'])

name_gyro = os.path.join('./../datasets', 'vgg16', name_dataset, name_dataset, set, name_dataset, 'gyro.txt')
df_gyro = pd.read_csv(name_gyro, sep=" ", engine="python", encoding="ISO-8859-1", names=['image', 'accx'])

# print(df_r_val.shape[0])
# print(len(df_r_val), len(df_gyro))

wrong = []
if(flag_classification):
    for c in range(5):
        for i in range(df_r_val.shape[0]):
            if c == df_r_val['real'][i]:
                if df_r_val['real'][i] != df_r_val['pred'][i]:
                    wrong.append([df_gyro['image'][i], int(df_r_val['real'][i]), df_r_val['pred'][i]])

wrong = pd.DataFrame(wrong)
wrong.columns = ['image', 'real', 'pred']
# wrong = wrong[wrong['real'] == classe]
# print(wrong)

f = open(os.path.join('./../experiments', exp, "images_error.txt"), "w")
for sample in range(len(wrong)):
    f.write(wrong['image'][sample] + "," + str(int(wrong['real'][sample])) + "," + str(wrong['pred'][sample]) + "\n")
f.close()
print("Finalizou")
