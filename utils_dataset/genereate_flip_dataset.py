from dataset import Dataset
from file import File
import cv2 as cv
import os

class Generate(object):

    def __init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name, network_name, dimension, resize, repeted, round_sample):
        self.network_name = network_name
        self.dimension = dimension
        self.resize = resize
        self.repeted = repeted
        self.round_sample = round_sample
        self.dimension = dimension
        self.folders = ["train", "val", "test"]
        self.output_dataset_name = output_dataset_name
        self.src = os.path.join(dataset_dir, output_dataset_name)
        self.dst = os.path.join(dataset_dir, network_name, output_dataset_name)
        Dataset.__init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name)
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

    def generate(self):
        self.makeDirDataset()
        Dataset.splitDataset(self, 0.7, 0.15, 0.15)
        for folder in zip(self.folders, self.sets):
            aux_dataset = []
            print("Initialize -> ", folder[0])
            dataset_file = File(os.path.join(self.dst, self.dataset_name, folder[0], self.output_dataset_name))
            for sample in folder[1].values:
                # print(sample)
                if (not self.check(aux_dataset, sample[1]) or repeted):
                    # print(os.path.join(self.src, sample[0][:]))
                    img = cv.imread(os.path.join(self.src, sample[0][:]), cv.IMREAD_UNCHANGED)
                    if (self.resize):
                        # print(folder[0], sample[0])
                        resized = cv.resize(img, self.dimension, interpolation=cv.INTER_AREA)
                        cv.imwrite(
                            os.path.join(self.dst, self.dataset_name, folder[0], self.output_dataset_name, "images/", sample[0]),
                            resized)
                    else:
                        cv.imwrite(
                            os.path.join(self.dst, self.dataset_name, self.folder[0], self.output_dataset_name, "images/", sample[0]), img)
                    aux_dataset.append(sample)
            dataset_file.saveFile(aux_dataset, folder)

dataset_directory = "../../datasets"
dataset_name = "sidewalk_accx_all_out_classes"
output_dataset_name = "sidewalk_accx_all_out_classes"
network_name = "vgg16"

dimension = (224, 224)

resize = True#Must be True always, False is exception
repeted = True
round_sample = False

gen = Generate(dataset_directory, dataset_name, dataset_directory, output_dataset_name, network_name, dimension, resize, repeted, round_sample)
gen.generate()