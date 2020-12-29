import os
import pandas as pd
import numpy as np

exp = ["exp_067"]

flag_create_txt = False

images_error_exps = []
for experiments in exp:
    name = os.path.join('./../experiments', experiments, "images_error" + '.txt')
    df = pd.read_csv(name, sep=",", engine="python", encoding="ISO-8859-1", header=None,
                     names=['image', 'real', 'pred'])
    # images_error_exps.append(df)
    # images_error_exps = df.copy()


# print('asdf', images_error_exps.shape)
images_out = []
# for df in images_error_exps:
for index in range(len(df)):
    if(abs(df['real'][index] - df['pred'][index]) > 1):
        images_out.append(df['image'][index])


imgs = pd.DataFrame(images_out, columns=["image"])
imgs.drop_duplicates(subset ="image", keep = False, inplace = True)
# print(imgs)
dataset_name = "sidewalk_accy_proportion_classes_full"
new_dataset_name = "sidewalk_accx_proportion_classes"

df_dataset = pd.read_csv(os.path.join('./../datasets', dataset_name + '.txt'), sep=" ", engine="python", encoding="ISO-8859-1", names=['img_dataset', 'img_original', 'folder', 'accx'])


a = len(df_dataset)
# Remove images wrongs
for sample in range(len(imgs)):
##    print(imgs['image'][sample].split('/')[1])
    m = df_dataset['img_dataset'] == imgs['image'][sample].split('/')[1]
    res = df_dataset[m]
    # print(res)
    if(len(res.index) != 0):
        df_dataset = df_dataset.drop([res.index[0]])
        df_dataset.reset_index(drop=True, inplace=True)
        print(imgs['image'][sample].split('/')[1])
    # else:
    #     print(imgs['image'][sample].split('/')[1])

print(len(imgs), a, len(df_dataset))

if flag_create_txt:
    # Create dataset with images wrongs removed
    f = open(os.path.join('./../datasets', new_dataset_name + '.txt'), 'w')
    for sample in range(len(df_dataset)):
        f.write(str(df_dataset['img_dataset'][sample]) + ' ' + str(df_dataset['img_original'][sample]) + ' ' + str(df_dataset['folder'][sample]) + ' ' + str(df_dataset['accx'][sample]) + '\n')
    f.close()
print('Finalizou')
    




        
