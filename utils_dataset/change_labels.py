import os
import pandas as pd
import numpy as np
import cv2 as cv

dataset_name = 'sidewalk_accy_proportion_classes'
data_type = 'val'
name = os.path.join('./../datasets', 'vgg16', dataset_name, dataset_name, data_type, dataset_name, 'gyro_psi.txt')
# name = os.path.join('./../datasets', dataset_name + ".txt")

# df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None)
df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None, names=['img_dataset', 'accx'])
##
##f = open(os.path.join('./../datasets', 'vgg16', dataset_name, dataset_name, data_type, dataset_name,
##                      'gyro_psi.txt'), "w")
##
##ori_name = 'sidewalk_psi'
##df_ori = pd.read_csv(os.path.join('./../datasets', ori_name + '.txt'), sep=" ", engine="python", encoding="ISO-8859-1", header=None, names=['img_dataset', 'img_original', 'folder', 'accx'])
##df_ori['accx'] = (df_ori['accx'] - df_ori['accx'].min()) / (df_ori['accx'].max() - df_ori['accx'].min())
##
##for sample in range(len(df)):
##    m = df_ori['img_dataset'] == df['img_dataset'][sample].split('/')[1]
##    res = df_ori[m]
####    print(df['img_dataset'][sample] + ' ' + str(res.iloc[0].accx) + '\n')
##    f.write(df['img_dataset'][sample] + ' ' + str(res.iloc[0].accx) + '\n')
##f.close()
##print('Finalizou')

######################

for sample in range(len(df)):
    if os.path.isfile(os.path.join('./../datasets', 'vgg16', dataset_name, dataset_name, data_type, dataset_name, df['img_dataset'][sample])):
        print('exists')






