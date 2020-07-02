import os
import json
import cv2 as cv
import shutil

labels_name = "labels_50_100"
dataset_name = "sidewalk"
video_name = "2020_05_31-17_44_57"
video_directory = "./../raw_datasets_videos"
images_directory = "./../raw_datasets_images"
dataset_directory = "./../datasets"

f = open(os.path.join(video_directory, video_name, labels_name + ".txt"))
##f = open(str(directory[:-2]) + "/" + "_labels_n.txt")
##dataset_file = open("/home/david/Área de Trabalho/navem_keras/arquivos/Bando de dados Mestrado/dataset_navem_novo/all_dataset.txt","r+")

if not os.path.exists(os.path.join(dataset_directory, dataset_name)):
    os.makedirs(os.path.join(dataset_directory, dataset_name))
    print("Path", os.path.join(dataset_directory, dataset_name), "created.")
else:
    print("Directory already exist.")

if not os.path.isfile(os.path.join(dataset_directory, dataset_name + ".txt")):
    dataset_file = open(os.path.join(dataset_directory, dataset_name + ".txt"), "w+")
    print("File", os.path.join(dataset_directory, dataset_name + ".txt"), "created.")
else:
    dataset_file = open(os.path.join(dataset_directory, dataset_name + ".txt"), "r+")
    print("File", os.path.join(dataset_directory, dataset_name + ".txt"), "was successfully opened.")

##for i in labels_file:
##    print(i)
#image_name image_name_original acc_y pedometer acc_x stop/continue
f_ini = len(dataset_file.readlines())
for i in f:
    if(f_ini < 10):
        shutil.copyfile(os.path.join(images_directory, video_name, i.split(" ")[0][:]), os.path.join(dataset_directory, dataset_name, "00000" + str(f_ini) + ".jpg" ))
        dataset_file.write("00000" + str(f_ini) + ".jpg" + " " + i.split(" ")[0][:] + " " + " ".join(i.split(" ")[1:]))
    elif(f_ini < 100):
        shutil.copyfile(os.path.join(images_directory, video_name, i.split(" ")[0][:]), os.path.join(dataset_directory, dataset_name, "0000" + str(f_ini) + ".jpg" ))
        dataset_file.write("0000" + str(f_ini) + ".jpg" + " " + i.split(" ")[0][:] + " " + " ".join(i.split(" ")[1:]))
    elif(f_ini < 1000):
        shutil.copyfile(os.path.join(images_directory, video_name, i.split(" ")[0][:]), os.path.join(dataset_directory, dataset_name, "000" + str(f_ini) + ".jpg" ))
        dataset_file.write("000" + str(f_ini) + ".jpg" + " " + i.split(" ")[0][:] + " " + " ".join(i.split(" ")[1:]))
    elif(f_ini < 10000):
        shutil.copyfile(os.path.join(images_directory, video_name, i.split(" ")[0][:]), os.path.join(dataset_directory, dataset_name, "00" + str(f_ini) + ".jpg" ))
        dataset_file.write("00" + str(f_ini) + ".jpg" + " " + i.split(" ")[0][:] + " " + " ".join(i.split(" ")[1:]))
    elif(f_ini < 100000):
        shutil.copyfile(os.path.join(images_directory, video_name, i.split(" ")[0][:]), os.path.join(dataset_directory, dataset_name, "0" + str(f_ini) + ".jpg" ))
        dataset_file.write("0" + str(f_ini) + ".jpg" + " " + i.split(" ")[0][:] + " " + " ".join(i.split(" ")[1:]))
    else:
        shutil.copyfile(os.path.join(images_directory, video_name, i.split(" ")[0][:]), os.path.join(dataset_directory, dataset_name, str(f_ini) + ".jpg" ))
        dataset_file.write("" + str(f_ini) + ".jpg" + " " + i.split(" ")[0][:] + " " + " ".join(i.split(" ")[1:]))
    f_ini+=1
dataset_file.close()
print('Finalizou')

##for i in range(len(gyro_y)):
##    gyro_y[i] = float(gyro_y[i])*-1
##
##fn = open(str(directory_dataset) + "/" + "labels" + "_" + str(begin) + "_" + str(end) + "_n" + ".txt", "w")
##for i in zip(frames, gyro_y):
##    print(i)
##    fn.write(str(i[0]) + "|" + str(i[1]) + "/n")
##fn.close()
