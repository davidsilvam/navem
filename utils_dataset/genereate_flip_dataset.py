from dataset import Dataset
from file import File
import cv2 as cv
import os
import pandas as pd
from shutil import copyfile

class Generate(object):

    def __init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name, network_name, dimension,
                 resize, repeted, round_sample, joined_dataset=False, copy_flag=False):
        self.network_name = network_name
        self.dimension = dimension
        self.resize = resize
        self.repeted = repeted
        self.round_sample = round_sample
        self.dimension = dimension
        self.folders = ["train", "val", "test"]
        self.output_dataset_name = output_dataset_name
        self.dataset_dir = dataset_dir
        self.src = os.path.join(dataset_dir, dataset_name)
        self.dst = os.path.join(dataset_dir, network_name, output_dataset_name)
        self.copy_flag = copy_flag
        Dataset.__init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name, joined_dataset)
        Dataset.loadDataset(self)

    def makeDirDataset(self):
        if not os.path.exists(os.path.join(self.dataset_dir, self.network_name, self.output_dataset_name)):
            os.makedirs(os.path.join(self.dataset_dir, self.network_name, self.output_dataset_name))
            print("Path", os.path.join(self.dataset_dir, self.network_name, self.output_dataset_name), "created.")
        else:
            print("Directory already exist.")

        for folder in self.folders:
            if not os.path.exists(os.path.join(self.dst, self.output_dataset_name, folder, self.output_dataset_name, "images")):
                os.makedirs(os.path.join(self.dst, self.output_dataset_name, folder, self.output_dataset_name, "images"))
                self.dataset_name_flag = True

    def check(self, list, elem):
        for i in list:
            if (elem == i[0]):
                return True
        return False

    def generateAllDatasets(self):
        self.makeDirDataset()
        Dataset.splitDataset(self, 0.7, 0.15, 0.15)
        for folder in zip(self.folders, self.sets):
            aux_dataset = []
            print("Initialized -> ", folder[0] + " -> " + str(folder[1].shape[0]))
            dataset_file = File(os.path.join(self.dst, self.output_dataset_name, folder[0], self.output_dataset_name))
            if True:
                for sample in range(len(folder[1])):#len(folder[1])
                    # print(folder[1]['img_dataset'].iloc[sample])
                    if not self.check(aux_dataset, folder[1]['accx'].iloc[sample]) or repeted:
                        path_img = os.path.join(self.dataset_dir, folder[1]['dataset'].iloc[sample],
                                                folder[1]['img_dataset'].iloc[sample])
                        # print(path_img, os.path.exists(path_img))
                        # os.system('pause')
                        if os.path.exists(path_img or self.copy_flag):
                            # os.system('pause')
                            # a = 1
                            out_path = os.path.join(self.dst, self.output_dataset_name, folder[0],
                                                    self.output_dataset_name,
                                                    "images/", folder[1]['new_img_datset'].iloc[sample])
                            if self.resize:
                                if not self.copy_flag:
                                    img = cv.imread(path_img, cv.IMREAD_UNCHANGED)
                                    resized = cv.resize(img, self.dimension, interpolation=cv.INTER_AREA)
                                    # out_path = os.path.join(self.dst, self.output_dataset_name, folder[0],
                                    #                         self.output_dataset_name,
                                    #                         "images/", folder[1]['new_img_datset'].iloc[sample])
                                    cv.imwrite(out_path, resized)
                                else:
                                    src_all_dataset = os.path.join(self.dataset_dir, self.dataset_name)
                                    copyfile(
                                        os.path.join(src_all_dataset, folder[1]['new_img_datset'].iloc[sample]),
                                        os.path.join(out_path))
                                    # print(os.path.join(src_all_dataset, folder[1]['new_img_datset'].iloc[sample]))

                            aux_dataset.append(folder[1].iloc[sample])
            else:
                print('pass train')
            aux_dataset = pd.DataFrame(aux_dataset)
            aux_dataset = aux_dataset.sort_index()
            aux_dataset = aux_dataset.reset_index(drop=True)
            # print(aux_dataset)
            # os.system('pause')
            dataset_file.saveFile2(aux_dataset, folder)

    def generate(self):
        self.makeDirDataset()
        Dataset.splitDataset(self, 0.7, 0.15, 0.15)
        for folder in zip(self.folders, self.sets):
            aux_dataset = []
            print("Initialized -> ", folder[0])
            dataset_file = File(os.path.join(self.dst, self.output_dataset_name, folder[0], self.output_dataset_name))
            for sample in folder[1].values:
                # print(sample)
                if (not self.check(aux_dataset, sample[1]) or repeted):
                    # print(os.path.join(self.src, sample[0][:]))
                    img = cv.imread(os.path.join(self.src, sample[0][:]), cv.IMREAD_UNCHANGED)
                    if (self.resize):
                        # print(folder[0], sample[0])
                        resized = cv.resize(img, self.dimension, interpolation=cv.INTER_AREA)
                        cv.imwrite(
                            os.path.join(self.dst, self.output_dataset_name, folder[0], self.output_dataset_name, "images/", sample[0]),
                            resized)
                    else:
                        cv.imwrite(
                            os.path.join(self.dst, self.output_dataset_name, self.folder[0], self.output_dataset_name, "images/", sample[0]), img)
                    aux_dataset.append(sample)
                    # print(folder)
                    # os.system("pause")
            dataset_file.saveFile(aux_dataset, folder)

for i in range(0, 1):
#dataset accx -> sidewalk_accx_all_out_classes -> sidewalk_accx_184_pc_dataset_" + str(i)
#dataset accy -> sidewalk_accy_proportion_classes_fliped_3 -> sidewalk_accy_315_pc_dataset_" #+ str(i)
#dataset accy no flipped -> sidewalk_accy_all_out_classes -> sidewalk_accy_158_pc_dataset_" + str(i)
    dataset_directory = "../../datasets"
    dataset_name = "sidewalk_accy_all_datasets_classes_new_900"
    # datasets_names_files = ['sidewalk_accx_all_out_classes', 'sidewalk_dataset_x_out_pc_classes', 'plus_sidewalk_dataset_x_out_pc_classes']
    if i < 10:
        output_dataset_name = "sidewalk_accy_all_datasets_classes_new_900_0" + str(i)
    else:
        output_dataset_name = "sidewalk_accy_all_datasets_classes_new_900_" + str(i)
    network_name = "vgg16"

    dimension = (224, 224)

    resize = True#Must be True always, False is exception
    repeted = True
    round_sample = False

    gen = Generate(dataset_directory, dataset_name, dataset_directory, output_dataset_name, network_name, dimension,
                   resize, repeted, round_sample, joined_dataset=True, copy_flag=False)
    # gen.generate()
    gen.generateAllDatasets()