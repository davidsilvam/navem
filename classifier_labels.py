import os
import pandas as pd

dataset_name = 'sidewalk_accx_proportion'
data_type = 'val'
name = os.path.join('./../datasets', 'vgg16', dataset_name, dataset_name, data_type, dataset_name, 'gyro.txt')

df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None)

f = open(os.path.join('./../datasets', 'vgg16', dataset_name, dataset_name, data_type, dataset_name,
                      'gyro_classifier.txt'), "w")

max = 1
for i in range(len(df)):
    #print(str(df.at[i, 0]) + " " + str(df.at[i, 1]))
    if (df.at[i, 1] > 0) and (df.at[i, 1] <= max/5):
        # f.write(str(df.at[i, 0]) + " " + str(1) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + "\n")
        f.write(str(df.at[i, 0]) + " " + str(0) + "\n")
    elif (df.at[i, 1] > max/5) and (df.at[i, 1] <= (max/5)*2):
        # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(1) + " " + str(0) + " " + str(0) + " " + str(0) + "\n")
        f.write(str(df.at[i, 0]) + " " + str(1) + "\n")
    elif(df.at[i, 1] > (max/5)*2) and (df.at[i, 1] <= (max/5)*3):
        # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(1) + " " + str(0) + " " + str(0) + "\n")
        f.write(str(df.at[i, 0]) + " " + str(2) + "\n")
    elif(df.at[i, 1] > (max/5)*3) and (df.at[i, 1] <= (max/5)*4):
        # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(1) + " " + str(0) + "\n")
        f.write(str(df.at[i, 0]) + " " + str(3) + "\n")
    else:
        # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(1) + "\n")
        f.write(str(df.at[i, 0]) + " " + str(4) + "\n")

print('Finalizou')
f.close()
