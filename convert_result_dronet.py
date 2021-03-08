import os
import pandas as pd
import numpy as np

phase = "test"
set = "val"
exp = "exp_115"
name_weights = "weights_dronet.h5" # weights_500.h5 model_weights_499.h5

if phase == "train":
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_1_" + '.txt')
else:
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_0_" + '.txt')

df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", names=['stear', 'col', 'classify_real'])

# f = open(os.path.join('./../../datasets', exp, "predict_truth_" + set + "_" + name_weights + "_1_new" + '.txt'), "w")

def ConvertLabelsRegress2Classify(maximun, value):
    if (value > 0) and (value <= maximun / 5):
        return 0
    elif (value > maximun / 5) and (value <= (maximun / 5) * 2):
        return 1
    elif (value > (maximun / 5) * 2) and (value <= (maximun / 5) * 3):
        return 2
    elif (value > (maximun / 5) * 3) and (value <= (maximun / 5) * 4):
        return 3
    else:
        return 4

for sample in range(df.shape[0]):
    print(str(ConvertLabelsRegress2Classify(1, float(df['stear'][sample][1:]))) + " " + str(df['classify_real'][sample]) + "\n")
    # f.write(str(df['img_dataset'][i]) + " " + str(0) + "\n")

# f,close()