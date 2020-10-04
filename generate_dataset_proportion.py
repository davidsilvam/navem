import os
import pandas as pd
import random
import numpy as np

name_full = os.path.join('./../datasets', 'sidewalk_accy.txt')# Full dataset
f = open(os.path.join('./../datasets', 'sidewalk_accy_proportion.txt'), "w")

df_full = pd.read_csv(name_full, sep=" ", engine="python", encoding="ISO-8859-1", names=['img_dataset', 'img_original', 'folder', 'accx'])

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
