import os
import pandas as pd

dataset_name = 'sidewalk_accy_proportion_classes'
data_type = 'train'
name = os.path.join('./../datasets', 'vgg16', dataset_name, dataset_name, data_type, dataset_name, 'gyro.txt')
# name = os.path.join('./../datasets', dataset_name + ".txt")

# df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None)
df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None, names=['img_dataset', 'accx'])

f = open(os.path.join('./../datasets', 'vgg16', dataset_name, dataset_name, data_type, dataset_name,
                      'gyro_classifier.txt'), "w")
# f = open(os.path.join('./../datasets', dataset_name + "_classes.txt"), "w")

max = 1
for i in range(len(df)):
    #print(str(df.at[i, 0]) + " " + str(df.at[i, 1]))
    if (df['accx'][i] > 0) and (df['accx'][i] <= max/5):
        # f.write(str(df.at[i, 0]) + " " + str(1) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + "\n")
        f.write(str(df['img_dataset'][i]) + " " + str(0) + "\n")
        # f.write(str(df['img_dataset'][i]) + " " + str(df['img_original'][i]) + " " + str(df['folder'][i]) + " " + str(0) + "\n")
    elif (df['accx'][i] > max/5) and (df['accx'][i] <= (max/5)*2):
        # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(1) + " " + str(0) + " " + str(0) + " " + str(0) + "\n")
        f.write(str(df['img_dataset'][i]) + " " + str(1) + "\n")
        # f.write(str(df['img_dataset'][i]) + " " + str(df['img_original'][i]) + " " + str(df['folder'][i]) + " " + str(1) + "\n")
    elif(df['accx'][i] > (max/5)*2) and (df['accx'][i] <= (max/5)*3):
        # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(1) + " " + str(0) + " " + str(0) + "\n")
        f.write(str(df['img_dataset'][i]) + " " + str(2) + "\n")
        # f.write(str(df['img_dataset'][i]) + " " + str(df['img_original'][i]) + " " + str(df['folder'][i]) + " " + str(2) + "\n")
    elif(df['accx'][i] > (max/5)*3) and (df['accx'][i] <= (max/5)*4):
        # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(1) + " " + str(0) + "\n")
        f.write(str(df['img_dataset'][i]) + " " + str(3) + "\n")
        # f.write(str(df['img_dataset'][i]) + " " + str(df['img_original'][i]) + " " + str(df['folder'][i]) + " " + str(3) + "\n")
    else:
        # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(1) + "\n")
        f.write(str(df['img_dataset'][i]) + " " + str(4) + "\n")
        # f.write(str(df['img_dataset'][i]) + " " + str(df['img_original'][i]) + " " + str(df['folder'][i]) + " " + str(4) + "\n")

# for i in range(len(df)):
#     #print(str(df.at[i, 0]) + " " + str(df.at[i, 1]))
#     if (df.at[i, 1] > 0) and (df.at[i, 1] <= max/5):
#         # f.write(str(df.at[i, 0]) + " " + str(1) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + "\n")
#         f.write(str(df.at[i, 0]) + " " + str(0) + "\n")
#     elif (df.at[i, 1] > max/5) and (df.at[i, 1] <= (max/5)*2):
#         # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(1) + " " + str(0) + " " + str(0) + " " + str(0) + "\n")
#         f.write(str(df.at[i, 0]) + " " + str(1) + "\n")
#     elif(df.at[i, 1] > (max/5)*2) and (df.at[i, 1] <= (max/5)*3):
#         # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(1) + " " + str(0) + " " + str(0) + "\n")
#         f.write(str(df.at[i, 0]) + " " + str(2) + "\n")
#     elif(df.at[i, 1] > (max/5)*3) and (df.at[i, 1] <= (max/5)*4):
#         # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(1) + " " + str(0) + "\n")
#         f.write(str(df.at[i, 0]) + " " + str(3) + "\n")
#     else:
#         # f.write(str(df.at[i, 0]) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(1) + "\n")
#         f.write(str(df.at[i, 0]) + " " + str(4) + "\n")


print('Finalizou')
f.close()
