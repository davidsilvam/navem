import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
from datetime import date
import cv2 as cv
import os

dataset_directory = "./../datasets"
o_dataset_name = "sidewalk"
network = "vgg16"

dim = (224, 224)

repeted = False
round_sample = -1
dataset_name = "sidewalk_cn"
dataset_name_case = "sidewalk"

if not os.path.exists(os.path.join(dataset_directory, network, o_dataset_name)):
    os.makedirs(os.path.join(dataset_directory, network, o_dataset_name))
    print("Path", os.path.join(dataset_directory, network, o_dataset_name), "created.")
else:
    print("Directory already exist.")


#Load dataset
data = pd.read_csv(os.path.join(dataset_directory, o_dataset_name + ".txt"), sep=" ", engine="python", encoding="ISO-8859-1", header=None).values

min_max_array = []
def getMinMax(array, col):
    return [data[:,col].min(), data[:,col].max()]

min_max_array.append(getMinMax(data, 2))#acc_y

#Normalization
#data[:,1] = (data[:,1]-data[:,1].min())/(data[:,1].max()-data[:,1].min())#Nomalize de 0 a 1
##data[:,1] = data[:,1].astype(float)
data[:,2] = (data[:,2] - np.mean(data[:,2]))/np.std(data[:,2])#normalize (x - media) / desvio padrao
##data[:,2] = data[:,2].astype(float)
##data[:,2] = (data[:,2]-data[:,2].min())/(data[:,2].max()-data[:,2].min())#Nomalize
trainingSet, testSet = train_test_split(data, test_size=0.3)

src = os.path.join(dataset_directory, o_dataset_name)
dst = os.path.join(dataset_directory, network, o_dataset_name)

folders = ["train", "val"]

trainingSet = trainingSet[trainingSet[:,0].argsort()]
testSet = testSet[testSet[:,0].argsort()]

sets = [trainingSet, testSet]

#Legenda: nmp = noamalização média desvio padrão
dst_dataset_name = os.path.join(dst, dataset_name)

dataset_name_flag = True

print("Started")

for folder in folders:
    if not os.path.exists(os.path.join(dst_dataset_name, folder, dataset_name_case, "images")):
        os.makedirs(os.path.join(dst_dataset_name, folder, dataset_name_case, "images"))
        dataset_name_flag = True

trainFile = open(os.path.join(dst_dataset_name, "train", dataset_name_case, "gyro.txt"), "w")
valFile = open(os.path.join(dst_dataset_name, "val", dataset_name_case, "gyro.txt"), "w")

##trainFileGP = open(dst_dataset_name + "/" + "train" + "/" + dataset_name_case + "/" + "gyro_gp.txt", "w")
##valFileGP = open(dst_dataset_name + "/" + "val" + "/" + dataset_name_case + "/" + "gyro_gp.txt", "w")

def check(list, elem):
    for i in list:
        if(elem == i):
            return True
    return False

def elemEqualsSet(name, dataset, lis):
    today = date.today()
    log = open(name + "_" + str(today) + ".txt", "w")
    for i in lis:
        print("-----------")
        for j in range(len(dataset)):
            if(dataset[j][1] == i):
                print(dataset[j][0], j, i)
                log.write(dataset[j][0] + " " + str(j) + " " + str(i) + " " + "\n")
    for col in ["gyro"]:
        for sample in min_max_array:
                log.write(col + ":" + str(sample))
                        
##      log.write("min pedo" + " " + str(data[:,2].min()))
##      log.write("max pedo" + " " + str(data[:,2].max()))
    log.close()

print(trainingSet)
print(testSet)

programPause = input("Press the <ENTER> key to continue save train and test...")
for folder in zip(folders, sets):
    elem = []
    alreadyExist = []
    if(dataset_name_flag):
        print(folder[0])
        for sample in folder[1]:
            if(not check(elem, sample[1]) or repeted):
                img = cv.imread(os.path.join(src, sample[0][:]), cv.IMREAD_UNCHANGED)
                resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
                cv.imwrite(os.path.join(dst_dataset_name, folder[0], dataset_name_case, "images/", sample[0]), resized)
#                               print(dst_dataset_name + "/" + folder[0] + "/" + sample[0])
                if(folder[0] == "train"):
                    if(round_sample == -1):
                        trainFile.write(folder[0] + "/" + sample[0][1:] + " " + str(sample[2]) + "\n")
##                                              trainFileGP.write(folder[0] + "/" + sample[0][1:] + " " + str(sample[1]) + " " + str(sample[2]) + "\n")
                    else:
                        trainFile.write(folder[0] + "/" + sample[0][1:] + " " + str(round(sample[2], round_sample)) + "\n")
##                                              trainFileGP.write(folder[0] + "/" + sample[0][1:] + " " + str(round(sample[1], round_sample)) + " " + str(round(sample[2], round_sample)) + "\n")
                    elem.append(sample[1])
                if(folder[0] == "val"):
                    if(round_sample == -1):
                        valFile.write(folder[0] + "/" + sample[0][1:] + " " + str(sample[2]) + "\n")
##                                              valFileGP.write(folder[0] + "/" + sample[0][1:] + " " + str(sample[1]) + " " + str(sample[2]) + "\n")
                    else:
                        valFile.write(folder[0] + "/" + sample[0][1:] + " " + str(round(sample[2], round_sample)) + "\n")
##                                              valFileGP.write(folder[0] + "/" + sample[0][1:] + " " + str(round(sample[1], round_sample)) + " " + str(round(sample[2], round_sample)) + "\n")
                    elem.append(sample[1])
            else:
                alreadyExist.append(sample[1])
        print("Qtd samples", len(elem), "| All samples", len(folder[1]))
        print("Already exist in " + folder[0] + " : ", alreadyExist)
        elemEqualsSet(dst_dataset_name + "/log_" + folder[0], folder[1], alreadyExist)

trainFile.close()
valFile.close()
