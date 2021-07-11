import cv2 as cv
import os
import pandas as pd
from shutil import copyfile

class UtilDataset(object):
    def __init__(self, img_name, path_dataset, dimension, out_path_resize, dataset_dir, output_dataset_name, dataset_name, network):
        self.img_name = img_name
        self.path_dataset = path_dataset
        self.path_img = os.path.join("../../datasets", self.path_dataset, self.img_name)
        self.dimension = dimension
        self.out_path_resize = out_path_resize
        self.dataset_dir = dataset_dir
        self.output_dataset_name = output_dataset_name
        self.dataset_name = dataset_name
        self.network = network
        self.folders = ["train", "val", "test"]

        self.dataset_loaded = None

    def resize(self):
        out_path = os.path.join(self.out_path_resize, self.img_name)

        img = cv.imread(self.path_img, cv.IMREAD_UNCHANGED)
        resized = cv.resize(img, dimension, interpolation=cv.INTER_AREA)
        cv.imwrite(out_path, resized)

    def makeDirDataset(self):
        if not os.path.exists(os.path.join(self.dataset_dir, self.output_dataset_name)):
            os.makedirs(os.path.join(self.dataset_dir, self.output_dataset_name))
            print("Path", os.path.join(self.dataset_dir, self.output_dataset_name), "created.")
        else:
            print("Directory already exist.")

    def makeDatasetImagesFirstResize(self):
        self.dataset_loaded = pd.read_csv(os.path.join(self.dataset_dir, self.path_dataset + ".txt"),
                               sep=" ", engine="python", encoding="ISO-8859-1",
                               names=['img_dataset', 'img_original', 'folder', 'accx', 'dataset',
                                      'new_img_datset'])
        self.makeDirDataset()
        # print(self.dataset_loaded)
        # path = os.path.join(self.dataset_dir, self.network)
        dst = os.path.join(self.dataset_dir, self.output_dataset_name)
        print('Initialized')
        for folder in self.folders:
            a = os.path.join(self.dataset_dir, self.network, self.dataset_name, self.dataset_name, folder, self.dataset_name)
            gyro_data = pd.read_csv(os.path.join(a, "gyro.txt"), sep=" ", engine="python", encoding="ISO-8859-1", names=['img_dataset', 'accx'])
            print(folder)
            # print(gyro_data)
            for sample in range(len(gyro_data)):
                # print(os.path.join(a, "images", gyro_data['img_dataset'].iloc[sample].split('/')[1]))
                # print(dst)
                copyfile(os.path.join(a, "images", gyro_data['img_dataset'].iloc[sample].split('/')[1]),
                         os.path.join(dst, gyro_data['img_dataset'].iloc[sample].split('/')[1]))
        print('Finished')



dataset_dir = "../../datasets"
output_dataset_name = "sidewalk_accy_all_datasets_classes" # file in
img_name = "006249.jpg"
out_path_resize = r"C:\Users\david\OneDrive\Mestrado\SIBGRAPI"
path_dataset = "indoor_dataset_velx_out_pc_classes" # path out save images
dimension = (300, 300)


dataset_name = "sidewalk_accy_all_datasets_classes_00" # path where images is
network = "vgg16"

a = UtilDataset(img_name, path_dataset, dimension, out_path_resize, dataset_dir, output_dataset_name, dataset_name, network)
# a.resize()
a.makeDatasetImagesFirstResize() # Create folder with all images dataset