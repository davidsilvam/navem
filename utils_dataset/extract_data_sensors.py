import os
import json
import math
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

begin = 50
end = 3501

video_name = "2020_06_25-16_49_23"
video_directory = "./../raw_datasets_videos"
images_directory = "./../raw_datasets_images"

accJson = json.load(open(os.path.join(video_directory, video_name, "accelerations.json"),"r"))
camJson = json.load(open(os.path.join(video_directory, video_name, "frames.json"),"r"))
gyroJson = json.load(open(os.path.join(video_directory, video_name, "rotations.json"),"r"))

def fill(array, key):    
    data = []
    if(key == 'frames'):
        for sample in array[key]:
            data.append([sample['frame_id'], sample['sensor_timestamp'], sample['time_usec']])
        return pd.DataFrame(data, columns=['frame_id', 'sensor_timestamp', 'time_usec'])
    else:    
        for sample in array[key]:
            data.append([sample['x'], sample['y'], sample['z'], sample['time_usec']])
        return pd.DataFrame(data, columns=['x', 'y', 'z', 'time_usec'])

def moving_avg(array, n, axis="y"):
    aux = []
    for i in range(array.index.min(), array.index.max() - (n - 2)):
        res = 0
        for j in range(n):
            res+=array[axis][i + j]
        aux.append(res/n)
    arrayOut = array.drop(array.index[:n - 1])
    arrayOut[axis] = np.array(aux)
    return arrayOut

def turnReferenceTime(array_, ref = []):
    aux = []
    array = array_.copy()
    for i in range(len(array)):
        if(len(ref) == 0):
            aux.append((array["time_usec"][i] - array["time_usec"][0])*10**-6)
        elif(len(ref) == 4):
            aux.append((array["time_usec"][i] - ref["time_usec"])*10**-6)
    array["time_usec"] = np.array(aux)
    return array

def getTimeArray(arrayIn, arrayOut, ref = []):
    for i in range(len(arrayIn)):
        if(len(ref) == 0):
            arrayOut.append((arrayIn[i]["time_usec"] - arrayIn[0]["time_usec"])*10**-6)
        elif(len(ref) == 4):
            arrayOut.append((arrayIn[i]["time_usec"] - ref["time_usec"])*10**-6)
            
##def stepsBySeconds(array, sec):
##    indexes = getIndexSec(array, sec + 1)
##    steps = 0
##    flag = False
##    for sample in range(indexes[0], indexes[1]):
##        if(array['y'][sample] > 10.5):
##            flag = True      
##        elif array['y'][sample] < 10.5 and flag == True:
##            steps += 1
##            flag = False
##        if(indexes[1] - 1 == sample and array['y'][sample] > 10.5):
##            steps+=1
##    return steps

def stepsBySeconds(array, sec, time):    
    mask = (array['time_usec'] >= sec - time) & (array['time_usec'] <= sec + time)
    r = array.loc[mask]
    steps = 0
    flag = False    
    for sample in range(r.index.min(), r.index.max()):
        if(r['y'][sample] > 10.5):
            flag = True      
        elif r['y'][sample] < 10.5 and flag == True:
            steps += 1
            flag = False    
    return steps/(2*time)

def qtdSteps(array):     
    steps = 0
    flag = False
    for sample in range(array.index.min(), array.index.max()):
        if(array['y'][sample] > 10.5):
            flag = True      
        elif array['y'][sample] < 10.5 and flag == True:
            steps += 1
            flag = False
    return steps

def getIndexSec(array, sec, axis="time_usec"):
    if(sec == 1):
        for i in range(t.index.min(), len(array[axis])):
            if(array[axis][i] > 1):
                return (t.index.min(), i - 1)
    else:
        interval = 0
        flag = True
        for i in range(array.index.min(), len(array[axis])):
            if(array[axis][i] > sec):                
                return (interval, i - 1)
            if(array[axis][i] > sec - 1 and flag):
                flag = False
                interval = i - 1
                if(interval < array.index.min()):
                    interval = array.index.min()                
    return (interval, i)

def getValuesSensorEachFrameOld(arrayCam, arraySensor, begin, end):
    camAux = arrayCam[begin:end + 1]
    for t in range(camAux.index.min() + 1, camAux.index.max()):
        mask = (arraySensor['time_usec'] >= camAux["time_usec"][t - 1]) & (arraySensor['time_usec'] <= camAux["time_usec"][t])
    return arraySensor.loc[mask]

def getValuesSensorEachFrame(arrayCam, arraySensor, sample):
    mask = (arrayCam["time_usec"][sample - 1] <= arraySensor['time_usec']) & (arraySensor['time_usec'] <= arrayCam["time_usec"][sample])
    return arraySensor.loc[mask]

def getLabels(arrayGyro, arrayAcc, arrayCam, begin, end, n_avg_gyro, n_avg_acc, step_fps = 1):
    labels = open(os.path.join(video_directory, video_name, "labels" + "_" + str(begin) + "_" +  str(end) + ".txt"), "w")    
    acc = moving_avg(turnReferenceTime(arrayAcc), n_avg_acc)
##    gyro = turnReferenceTime(arrayGyro)
##    cam = turnReferenceTime(arrayCam)
    gyro = arrayGyro
    cam = arrayCam
    cam_sec = turnReferenceTime(arrayCam)
    for sample in range(begin, end, step_fps):
        #acc_aux = moving_avg(getValuesSensorEachFrame(cam, acc, sample), n_avg_acc)
        time_begin = acc.loc[acc['time_usec'] > cam_sec['time_usec'][sample]].index.min()
##        print(time_begin)
        gyro_aux = getValuesSensorEachFrame(cam, gyro, sample)
        acc_aux = getValuesSensorEachFrame(cam, arrayAcc, sample)
##        if(sample == 2022):
##            print(gyro_aux)
##            a = input()
##        print(gyro_aux.shape[0])
        if(gyro_aux.shape[0] > n_avg_gyro - 1):
            gyro_aux = moving_avg(gyro_aux, n_avg_gyro)
            acc_aux = moving_avg(acc_aux, n_avg_gyro, axis="x")
            #print(sample, max(gyro_aux['y'], key=abs)*-1, stepsBySeconds(acc, acc['time_usec'][time_begin], 1.5), max(gyro_aux['x'], key=abs)*-1)#Terminou
#            print(sample, max(gyro_aux['y'], key=abs)*-1)
            print(str(sample) + ".jpg" + " " + str(max(gyro_aux['y'], key=abs)*-1) + " " + str(stepsBySeconds(acc, acc['time_usec'][time_begin], 1.5)) + " " +  str(max(gyro_aux['x'], key=abs)*-1))
            labels.write(str(sample) + ".jpg" + " " + str(max(gyro_aux['y'], key=abs)*-1) + " " + str(stepsBySeconds(acc, acc['time_usec'][time_begin], 1.5)) + " " +  str(max(gyro_aux['x'], key=abs)*-1) + "\n")
    labels.close()
    print("Finish")

df_acc = fill(accJson, "accelerations")
df_gyro = fill(gyroJson, "rotations")
df_cam = fill(camJson, "frames")

getLabels(df_gyro, df_acc, df_cam, begin, end, 8, 50, 1)#step=3


##acc = moving_avg(turnReferenceTime(df_acc), 50)
##gyro = turnReferenceTime(df_gyro)
##cam = turnReferenceTime(df_cam)

##acc_aux = getValuesSensorEachFrameOld(cam, acc, 550, 695)

##print(getIndexSec(cam, 1))

##p = []
##print(qtdSteps(acc[4237:5875]))
##for i in range(550, 695, 3):
##    time_begin = acc.loc[acc['time_usec'] > cam['time_usec'][i]].index.min()
##    time_end = acc.loc[acc['time_usec'] > cam['time_usec'][i] + 1].index.min()
##    print(time_begin, time_end)
##    print(stepsBySeconds(acc, acc['time_usec'][time_begin]))
##    p.append(stepsBySeconds(acc, acc['time_usec'][i]))

##print(np.mean(p), np.std(p))

##plt.plot(acc['y'][:])
##plt.show()






