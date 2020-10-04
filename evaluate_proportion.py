import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from sklearn.metrics import r2_score

#load full dataset
name_full = os.path.join('./../datasets', 'sidewalk_accy.txt')# Full dataset
# name = os.path.join('./../datasets','vgg16/sidewalk_accx/sidewalk_accx/val/sidewalk_accx' ,'gyro.txt')# vgg16 dataset
df_full = pd.read_csv(name_full, sep=" ", engine="python", encoding="ISO-8859-1", names=['img_dataset', 'img_original', 'folder', 'accx'])

count_default = 0
count_deceleration = 0

for i in range(df_full.shape[0]):
    if(df_full['accx'][i] < 0.08 and df_full['accx'][i] > -0.08):#Full dataset psi
    #if(df_full['accx'][i] < 2 and df_full['accx'][i] > -2):#Full dataset psi
    #if(df_full['accx'][i] > 1):#Full dataset accx   
    # if(df.at[i, 1] > 0.2 and df.at[i, 1] < 0.4):
        count_default+=1
        # print(df.at[i, 0], df.at[i, 1])
    else:
        count_deceleration+=1

print("============ proportion ===========")
print("% walking accx, psi- ", count_default / df_full.shape[0], "%")
print("% deceleration or desvio - ", count_deceleration / df_full.shape[0], "%")



########### begin real #############
phase = "test"
exp = "exp_035"
name_weights = "weights_500.h5"
name_dataset = "sidewalk_accx_proportion"

error = 0.4

set = "val"
#load val
if(phase == "train"):
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_1_" + '.txt')
else:
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_0_" + '.txt')

df_r = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None)
df_r_val = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None, names=['pred', 'real'])

name_gyro = os.path.join('./../datasets', 'vgg16', name_dataset, name_dataset, set, name_dataset, 'gyro.txt')
df_gyro = pd.read_csv(name_gyro, sep=" ", engine="python", encoding="ISO-8859-1", names=['image', 'accx'])

set = "train"
#load val
if(phase == "train"):
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_1_" + '.txt')
else:
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_" + name_weights + "_0_" + '.txt')

df_r_train = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None, names=['pred', 'real'])

name_gyro = os.path.join('./../datasets', 'vgg16', name_dataset, name_dataset, set, name_dataset, 'gyro.txt')
df_gyro_train = pd.read_csv(name_gyro, sep=" ", engine="python", encoding="ISO-8859-1", names=['image', 'accx'])


#truncate values
df_gyro['accx'] = np.trunc(100000000 * df_gyro['accx']) / 100000000

df_r_val_aux = df_r_val.copy()
df_r_train_aux = df_r_train.copy()
df_r_full =  df_r_val_aux.append(df_r_train_aux).copy()

print("============ Images out of error limit ===========")
for i in range(len(df_r_full)):
    if(abs(df_r_full.iloc[i]['pred'] - df_r_full.iloc[i]['real']) > error):#[i, 0]
        print(df_r_full.iloc[i]['pred'], df_r_full.iloc[i]['real'])
##        a = 1
        m = (df_r_full.iloc[i]['real'] == df_gyro['accx'])        
        a = df_gyro.loc[m]
        if(a.shape[0] == 0):            
            m = ((df_r_full.iloc[i]['real'] - 0.00000001) == df_gyro['accx'])# and (0.4 > (abs(df_r.at[i, 0] - df_gyro['accx'])))
            a = df_gyro.loc[m]
            #print(abs(df_r.at[i, 0] - a['accx']))
            #m = (0.4 >= (abs(df_r.at[i, 0] - a['accx'])))
            #a = a.loc[m]
##            print('como assim')
        if(a.shape[0] == 0):
            aux = str(df_r_full.iloc[i]['real'] - 0.00000001)
            w = aux.split('.')[0] + '.' + aux.split('.')[1][:7]
            m = (float(w) == df_gyro['accx'])
            a = df_gyro.loc[m]
##            print('legal')
##        print('asdf', df_r.at[i, 1], a['image'])
        if(a.shape[0] == 0):
            aux_a = df_gyro['accx'].copy()
            aux_a['accx'] = np.trunc(10000000 * df_gyro['accx']) / 10000000            
            aux = str(df_r_full.iloc[i]['real'])
            w = aux.split('.')[0] + '.' + aux.split('.')[1][:7]
##            print(w)
            m = (float(w) == aux_a['accx'])        
            a = df_gyro.loc[m]
##            print("o loko")
        if(a.shape[0] == 0):
            aux_a = df_gyro['accx'].copy()
            aux_a['accx'] = np.trunc(1000000 * df_gyro['accx']) / 1000000            
            aux = str(df_r_full.iloc[i]['real'] - 0.00000001)
            w = aux.split('.')[0] + '.' + aux.split('.')[1][:6]
##            print(w)
            m = (float(w) == aux_a['accx'])        
            a = df_gyro.loc[m]
##            print("o loko")
        for i in range(a.shape[0]):
            #print(abs(df_r.at[a.iloc[i].name, 0] - a.iloc[i]['accx']))
            if(abs(df_r_full.iloc[a.iloc[i].name]['pred'] - a.iloc[i]['accx']) > error):
                print(df_r_full.iloc[i]['real'], a.iloc[i]['image'])
            #print(a)
        print('------------')
        #print(a['image'][a['image'].index[0]])
        

#m = m = (0.2 > df['accx']) & (0.15 < df['accx'])
#a = df.loc[m]

########### end #################


def evaluate_img(img, folder, qtd_b, qtd_a, df_r_v, df_gyro_v, df_r_t, df_gyro_t, df_accx, plot_r2 = True):
    df_real = df_r_v.append( df_r_t).copy()
    df_gyro = df_gyro_v.append(df_gyro_t).copy()

    img_res_m = (img == df_accx['img_dataset'])    
    
    res = df_accx.loc[img_res_m]

    mask_folder = (res['folder'].item() == df_accx['folder'])   
    folder = df_accx.loc[mask_folder]

    #print((res.index[0] - qtd_b), (res.index[0] + qtd_a + 1))
    #print(folder.index)
    filter_range = folder.ix[(res.index[0] - qtd_b):(res.index[0] + qtd_a + 1)]
    #filter_range = folder.iloc[(res.index[0] - qtd_b):(res.index[0] + qtd_a + 1)]

    #print(filter_range)

    print("============ images in dataset ===========")
    array = []
    for item in filter_range.values:
        a = (str('val/' + item[0]) == df_gyro_v['image'])
        try:
            if(df_gyro_v.loc[a].shape[0] == 0):#train
                a = ('train/' + item[0] == df_gyro_t['image'])
                #print(df_gyro_t.loc[a])#print index
                #print(df_r_t.iloc[df_gyro_t.loc[a].index[0]])#print values pred and real            
                pred = df_r_t.iloc[df_gyro_t.loc[a].index[0]]['pred'].item()
                real = df_r_t.iloc[df_gyro_t.loc[a].index[0]]['real'].item()
                t = df_gyro_t.loc[a]['image'].item().split('/')[0]
                image = df_gyro_t.loc[a]['image'].item().split('/')[1]
                array.append([t, image, pred, real])
            else:#val
                #print(df_gyro_v.loc[a])
                #print(df_r_v.iloc[df_gyro_v.loc[a].index[0]])#print values pred and real
                pred = df_r_v.iloc[df_gyro_v.loc[a].index[0]]['pred'].item()
                real = df_r_v.iloc[df_gyro_v.loc[a].index[0]]['real'].item()
                t = df_gyro_v.loc[a]['image'].item().split('/')[0]
                image = df_gyro_v.loc[a]['image'].item().split('/')[1]
                array.append([t, image, pred, real])
            print(t, image, pred, real)
        except:
            print("Image out of dataset")
    df = pd.DataFrame(array, columns=['type', 'image', 'pred', 'real'])

    #r2 = r2_score(df['real'], df['pred'])

    plt.style.use('default')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 4)) 

    a_r = df[df['image'] == img]['real']

    ax.scatter(df[df['type'] == 'val']['real'], df[df['type'] == 'val']['pred'], edgecolor='k', facecolor='green', alpha=0.7, label='val')
    ax.scatter(df[df['type']=='train']['real'], df[df['type']=='train']['pred'], edgecolor='k', facecolor='blue', alpha=0.7, label='train')
    ax.scatter(df[df['image'] == img]['real'], df[df['image'] == img]['pred'], edgecolor='k', facecolor='red', alpha=0.7, label='val')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_ylabel('predict', fontsize=14)
    ax.set_xlabel('real', fontsize=14)

    ax.text(0.8, 0.1, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
             transform=ax.transAxes, color='grey', alpha=0.5)
    ax.legend(facecolor='white', fontsize=11)
    #ax.set_title('$%s -> R^2= %.2f$' % (set, r2), fontsize=18)

    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    plt.show()

    return pd.DataFrame(array, columns=['type', 'image', 'pred', 'real'])

##f = evaluate_img('004718.jpg', '2020_06_25-14_14_59', 5, 5, df_r_val, df_gyro, df_r_train, df_gyro_train, df_full)

