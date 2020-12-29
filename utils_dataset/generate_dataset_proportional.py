import os
import pandas as pd
import random
import numpy as np

name_full = os.path.join('./../datasets', 'sidewalk_psi_proportion_classes_10.txt')# Full dataset
f = open(os.path.join('./../datasets', 'sidewalk_psi_proportion_classes_11.txt'), "w")

df_full = pd.read_csv(name_full, sep=" ", engine="python", encoding="ISO-8859-1", names=['img_dataset', 'img_original', 'folder', 'accx'])

cla = 1 # class
percent = 0.20

def proportion():
    count_default = 0
    count_deceleration = 0

    for i in range(df_full.shape[0]):
        # if (df_full['accx'][i] > 1):  # Full dataset accx
        # if (df_full['accx'][i] < 2 and df_full['accx'][i] > -2):  # Full dataset psi
        if (df_full['accx'][i] < 0.08 and df_full['accx'][i] > -0.08):  # Full dataset psi
            # if(df.at[i, 1] > 0.2 and df.at[i, 1] < 0.4):
            count_default += 1
            # print(df.at[i, 0], df.at[i, 1])
        else:
            count_deceleration += 1
    return count_default / df_full.shape[0]# count_default if psi, count_desacelaration if accx

def make_proportion():
    while(proportion() > 0.50):# Logic in function of proportion if psi > 50, if accx < 50
        val = random.randint(1, df_full.shape[0] - 1)
        #(df_r.at[i, 1] == df_gyro['accx'])
        #print(df_full['accx'][val])
        # if (df_full['accx'][val] > 1):# Full dataset accx
        if (df_full['accx'][val] < 0.08 and df_full['accx'][val] > -0.08):  # Full dataset psi
            df_full = df_full.drop([val])
        df_full.reset_index(drop=True, inplace=True)
        print(proportion(), df_full.shape[0])

    for i in range(df_full.shape[0]):
        f.write(df_full['img_dataset'][i] + " " + df_full['img_original'][i] + " " + df_full["folder"][i] + " " + str(df_full["accx"][i]) + "\n")
    f.close()
    print("Terminou")

def proportion_classes(array, col, cla):
    c = [0, 0, 0, 0, 0]
    max = 1
    array_ = (array[col] - array[col].min()) / (array[col].max() - array[col].min())
    for i in range(len(array)):
        if (array_[i] > 0) and (array_[i] <= max / 5):
            c[0] += 1
        elif (array_[i] > max / 5) and (array_[i] <= (max / 5) * 2):
            c[1] += 1
        elif (array_[i] > (max / 5) * 2) and (array_[i] <= (max / 5) * 3):
            c[2] += 1
        elif (array_[i] > (max / 5) * 3) and (array_[i] <= (max / 5) * 4):
            c[3] += 1
        else:
            c[4] += 1
    return c[cla]/len(array)


def decode(num, min, max):
    return num * (max - min) + min

def make_proportion_classes(arr, col, file, cla):
    max = 1
    arr[col] = (arr[col] - arr[col].min()) / (arr[col].max() - arr[col].min())
    min = arr[col].min()
    max = arr[col].max()
    while(proportion_classes(arr, 'accx', cla) > percent):# Logic in function of proportion if psi > 50, if accx < 50
        m = {}
        if(cla == 1):
            m = (arr['accx'] > max / 5) & (arr['accx'] < (max / 5) * 2)
        elif(cla == 2):
            m = (arr['accx'] > (max / 5) * 2) & (arr['accx'] < (max / 5) * 3) # (arr['accx'] > (max / 5) * 2) & (arr['accx'] < (max / 5) * 3)   (arr['accx'] > (max / 5) * 4)
        elif(cla == 3):
            m = (arr['accx'] > (max / 5) * 3) & (arr['accx'] < (max / 5) * 4)
        elif(cla == 4):
            m = (arr['accx'] > (max / 5) * 4) # (arr['accx'] > (max / 5) * 2) & (arr['accx'] < (max / 5) * 3)
        a = arr[m]
        val = random.randint(1, len(a) - 1)
        rem = a.index[val]
        #(df_r.at[i, 1] == df_gyro['accx'])
        #print(df_full['accx'][val])
        # if (df_full['accx'][val] > 1):# Full dataset accx
        if(cla == 1):
            if (arr['accx'][rem] > max / 5) and (arr['accx'][rem] <= (max / 5) * 2):  # Full dataset psi arr['accx'][rem] > (max / 5) * 2) and (arr['accx'][rem] <= (max / 5) * 3
                arr = arr.drop([rem])
        elif(cla == 2):
            if (arr['accx'][rem] > (max / 5) * 2) and (arr['accx'][rem] <= (max / 5) * 3):  # Full dataset psi arr['accx'][rem] > (max / 5) * 2) and (arr['accx'][rem] <= (max / 5) * 3
                arr = arr.drop([rem])
        elif(cla == 3):
            if (arr['accx'][rem] > (max / 5) * 3) and (arr['accx'][rem] <= (max / 5) * 4):  # Full dataset psi arr['accx'][rem] > (max / 5) * 2) and (arr['accx'][rem] <= (max / 5) * 3
                arr = arr.drop([rem])
        elif(cla == 4):
            if (arr['accx'][rem] > (max / 5) * 4):  # Full dataset psi arr['accx'][rem] > (max / 5) * 2) and (arr['accx'][rem] <= (max / 5) * 3
                arr = arr.drop([rem])
        arr.reset_index(drop=True, inplace=True)
        print(proportion_classes(arr, 'accx', cla), arr.shape[0])
    for i in range(len(arr)):
        file.write(str(arr['img_dataset'][i]) + " " + str(arr['img_original'][i]) + " " + str(arr["folder"][i]) + " " + str(arr["accx"][i]) + "\n")
    file.close()
    print("Terminou")

# make_proportion()
make_proportion_classes(df_full, 'accx', f, cla)
