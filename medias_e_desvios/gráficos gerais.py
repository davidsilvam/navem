import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

timeExpArray = [] 
timeCam = []
timeIniArray = []
timeFimArray = []
timeFimIniArray = []
timeUsecSense = []
timeGyr_X = []
timeGyr_Y = []
timeGyr_Z = []

def difference2by2(exp):
    diff = []
    for time in range(1, len(exp)):
        diff.append(exp[time] - exp[time- 1])
    return diff

def abrirCam(cam):
    aberturaCam = open(cam, "r")
    timeCam = json.load(aberturaCam)
    for time in timeCam["dados"]:
        timeIniArray.append(time["time_ini_usec"])
    for time in timeCam["dados"]:
        timeFimArray.append(time["time_fim_usec"])
    for time in timeCam["dados"]:
        timeFimIniArray.append(time["time_fim-ini_usec"])

def abrirGyr(sensor):
    aberturaGyr = open(sensor,"r")
    timeGyr = json.load(aberturaGyr)
    for time in timeGyr["dados"]:
        timeUsecSense.append(time["time_usec"])
    for time in timeGyr["dados"]:
        timeGyr_X.append(time["x"])
    for time in timeGyr["dados"]:
        timeGyr_Y.append(time["y"])
    for time in timeGyr["dados"]:
        timeGyr_Z.append(time["z"])

def abrirExp(exp):
    aberturaExp = open(exp,"r")
    timeExp = json.load(aberturaExp)
    for time in timeExp["dados"]:
        timeExpArray.append(time["tempo_foto"])
        
def inteiro(exp):
    Inteiro = np.array(exp).astype(np.int64)
    exp = Inteiro
    return exp

def retirarPrimeiroNum(exp):
    Inteiro = np.array(exp).astype(np.int64)
    exp = Inteiro - Inteiro[0]
    return exp

def retirarNano(exp):
    for i in range(len(exp)):
        exp[i] = str(exp[i])
        exp[i] = exp[i][8:]
    return exp

abrirExp("/home/pi/Desktop/Mar  4 2020:17:05:2366/exp11.json")
abrirCam("/home/pi/Desktop/Mar  4 2020:17:05:2366/camera.json")


# inteiro(timeExpArray)

# retirarNano(timeExpArray)
# retirarNano(timeIniArray)
# retirarNano(timeFimArray)
timeIniArray = inteiro(timeIniArray)
timeFimArray = inteiro(timeFimArray)
timeExpArray = inteiro(timeExpArray)
# inteiro(timeFimIniArray)
# timeIniArray = retirarPrimeiroNum(timeIniArray)


# print(timeExpArray)
# print("*"*30)
# print(timeIniArray)
# print("*"*30)
# print(timeFimArray)

#SENSOR
# X = []
# Y = []
# Z = []
# Time = []

# X= np.array(timeGyr3_X).astype(float)
# Y = np.array(timeGyr3_Y).astype(float)
# Z = np.array(timeGyr3_Z).astype(float)
# 
# Time = np.array(timeUsecSense).astype(int)

# intevalos = []
# 
# timeExp3Array = np.array(timeExp3Array).astype(int)
# 
# timeIniArray3 = np.array(timeIniArray3).astype(int)
# timeFimArray3 = np.array(timeFimArray3).astype(int)
# timeFimIniArray3 = np.array(timeFimIniArray3).astype(int)
# 
# timeIniArray3 = timeIniArray3[1:] - timeIniArray3[1]
# timeFimArray3 = timeIniArray3 + timeFimIniArray3[1:]

# for i in range(-1,len(timeFimArray3)):
#     for j in range(len(Time)):
#         if timeIniArray3[i] < Time[j] <= timeFimArray3[i] :
#             print("Intervalo: ", timeIniArray3[i],"e ", timeFimArray3[i], "=", Time[j], "X: ",X[j], "Y: ",Y[j], "Z: ",Z[j]) 
        
# timeExp3ArrayZ = timeExp3Array[1:] - timeExp3Array[1]
# timeExp3ArrayZ = timeExp3Array - timeExp3Array[1]
# print("DIFERENÇA: ",difference2by2(timeExp3ArrayZ[1:]))


# print("INI: ",timeIniArray3[1:])

## print("FIM: ",timeFimArray3[1:])

#PLOT

#TRÊS:
# axis_x = np.ones(len(valuesExp3))*1.3
# plt.plot(valuesExp3, axis_x, "ro", color = 'Red', marker = 'o', label = 'RealEXP3')
# axis_x = np.ones(len(valuesIni3))*1.01
# plt.plot(valuesIni3, axis_x, "ro", color = 'Green', marker = 'x', label = 'INI3')
# axis_x = np.ones(len(valuesFim3))*1.2
# plt.plot(valuesFim3, axis_x, "ro", color = 'Blue', marker = '+', label = 'FIM3')
# 
#DOIS:
# # axis_x = np.ones(len(valuesExp2))*1.1
# # plt.plot(valuesExp2, axis_x, "ro", color = 'Black', marker = 'o', label = 'RealEXP2')
# # axis_x = np.ones(len(valuesIni2))
# # plt.plot(valuesIni2, axis_x, "ro", color = 'Yellow', marker = 'x', label = 'INI2')
# # axis_x = np.ones(len(valuesFim2))*1.19
# # plt.plot(valuesFim2, axis_x, "ro", color = 'Brown', marker = '.', label = 'FIM2')
# # 
# # 
# # plt.legend()
# # plt.show()

#LINHA DO TEMPO

axis_x = np.ones(len(timeExpArray))
plt.plot(inteiro(timeExpArray), axis_x, "ro", color = 'Black', marker = 'o', label = 'ENVIO3')
axis_x = np.ones(len(timeIniArray))
plt.plot(inteiro(timeIniArray),axis_x, "ro", color = 'Green', marker = 'x', label = 'INI3')
axis_x = np.ones(len(timeFimArray))
plt.plot(inteiro(timeFimArray),axis_x, "ro", color = 'Blue', marker = '.', label = 'FIM3')

plt.axis([min(timeIniArray), max(timeFimArray), 0.99, 1.004])
# plt.plot(timeExp3ArrayZ, axis_x, "ro", color = 'Red', marker = 'o', label = 'RealEXP3')

plt.legend()
plt.show()

#Linha do tempo em linhas diferentes
# axis_x = np.ones(len(timeExpArray))*1.001
# plt.plot(inteiro(timeExpArray), axis_x, "ro", color = 'Black', marker = 'o', label = 'ENVIO3')
# axis_x = np.ones(len(timeIniArray))*1.002
# plt.plot(inteiro(timeIniArray),axis_x, "ro", color = 'Green', marker = 'x', label = 'INI3')
# axis_x = np.ones(len(timeFimArray))
# plt.plot(inteiro(timeFimArray),axis_x, "ro", color = 'Blue', marker = '.', label = 'FIM3')
# 
# plt.axis([min(timeIniArray), max(timeFimArray), 0.99, 1.004])
# # plt.plot(timeExp3ArrayZ, axis_x, "ro", color = 'Red', marker = 'o', label = 'RealEXP3')
# 
# plt.legend()
# plt.show()
# 
