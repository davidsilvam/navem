import os
import pandas as pd
import numpy as np

flag_classification = True
phase = "test"
set = "val"
exp = "exp_049"
name_weights = "model_weights_149.h5" # weights_500.h5 model_weights_499.h5

max = 1

num_classes = 5
matrix = np.zeros((5, 5))

def getClass(acc, max):
    if (acc > 0) and (acc <= max / 5):
        return 0
    elif (acc > max / 5) and (acc <= (max / 5) * 2):
        return 1
    elif (acc > (max / 5) * 2) and (acc <= (max / 5) * 3):
        return 2
    elif (acc > (max / 5) * 3) and (acc <= (max / 5) * 4):
        return 3
    else:
        return 4

if phase == "train":
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_1_" + '.txt')
else:
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_0_" + '.txt')

df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])
if flag_classification:
    for i in range(len(df)):
        matrix[int(df['real'][i])][int(df['pred'][i])] += 1
else:
    for i in range(len(df)):
        matrix[getClass(df['real'][i], max)][getClass(df['pred'][i], max)] += 1

def getMetrics(m):
    print("Accuracy => ", sum(np.diag(m)) / np.sum(m))
    print("Precision")
    for c in range(len(m)):
        print("Class ", c, " => ", m[c][c] / np.sum(m[c, :]))
    print("Recall")
    for c in range(len(m)):
        print("Class ", c, " => ", m[c][c] / np.sum(m[:, c]))

print(matrix)
getMetrics(matrix)
