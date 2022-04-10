from file import File
import cv2 as cv
import os
import pandas as pd
from shutil import copyfile

class Dataset:
    def __init__(self, path, dataset_file_name):
        self.path = path
        self.datset_file_name = dataset_file_name
        self.dataset = None
        pass

    def load_dataset(self, dataset_type='full'):
        if dataset_type == 'set':
            self.dataset = pd.read_csv(os.path.join(self.path, self.datset_file_name + ".txt"),
                                       sep=" ", engine="python", encoding="ISO-8859-1",
                                       names=['new_img_dataset', 'accx'])
        elif dataset_type == 'full':
            self.dataset = pd.read_csv(os.path.join( self.path, self.datset_file_name + ".txt"),
                                   sep=" ", engine="python", encoding="ISO-8859-1",
                                   names=['img_dataset', 'img_original', 'folder', 'accx', 'dataset',
                                          'new_img_dataset', 'flipped'])
        else:
            self.dataset = pd.read_csv(os.path.join( self.path, self.datset_file_name + ".txt"),
                                   sep=" ", engine="python", encoding="ISO-8859-1",
                                   names=['img_dataset', 'img_original', 'folder', 'accx'])

class ModifyDataset:
    def __init__(self, path, dataset_name, exp_dataset_name, network, new_dataset_name, full_dataset_classes):
        self.path = path
        self.dataset_name = dataset_name
        self.exp_dataset_name = exp_dataset_name
        self.network = network
        self.new_dataset_name = new_dataset_name
        self.full_dataset_classes = full_dataset_classes
        self.folders = ["train", "val", "test"]
        if self.network == 'dronet':
            self.dimension = (200, 200)
        pass

    def modify(self):
        full_dataset = Dataset(self.path, self.dataset_name)
        full_dataset.load_dataset()

        # exp_dataset = Dataset(self.path, self.exp_dataset_name)
        # exp_dataset.load_dataset()
        # print(exp_dataset.dataset)
        for folder in self.folders:
            # print(os.path.join(self.path, self.network, self.exp_dataset_name, self.exp_dataset_name, folder, self.exp_dataset_name))
            path = os.path.join(self.path, self.network, self.exp_dataset_name, self.exp_dataset_name, folder, self.exp_dataset_name)
            out_path = os.path.join(self.path, self.network, self.exp_dataset_name, self.new_dataset_name, folder, self.new_dataset_name)
            exp_dataset = Dataset(path, 'gyro')
            exp_dataset.load_dataset('set')

            print(folder)
            os.makedirs(out_path, exist_ok=True)
            copyfile(os.path.join(path, 'gyro.txt'), os.path.join(out_path, 'gyro.txt'))

            for i, row in exp_dataset.dataset.iterrows():
                extract_img = row['new_img_dataset'].split('/')
                img_filter = (full_dataset.dataset['new_img_dataset'] == extract_img[1])

                img = cv.imread(os.path.join(self.path, full_dataset.dataset[img_filter]['dataset'].item(), full_dataset.dataset[img_filter]['img_dataset'].item()), cv.IMREAD_UNCHANGED)
                # print(img.shape[0], img.shape[1])
                h = img.shape[0]
                w = img.shape[1]
                x = 0
                y = int(img.shape[0] * 0.3)
                crop_img = img[y:y + h, x:x + w]
                resized = cv.resize(crop_img, self.dimension, interpolation=cv.INTER_AREA)

                os.makedirs(os.path.join(out_path, "images"), exist_ok=True)
                # cv.imshow('asdf', cv.resize(img, (480, 720), interpolation = cv.INTER_AREA))
                # new_path = os.path.join()
                # out_path = os.path.join()
                cv.imwrite(os.path.join(out_path, "images", full_dataset.dataset[img_filter]['new_img_dataset'].item()), resized)
                # os.system('pause')
        pass

    def modifyClassToRegress(self):
        full_dataset = Dataset(self.path, self.dataset_name)
        full_dataset.load_dataset(dataset_type='regress')
        full_dataset_classes = Dataset(self.path, self.full_dataset_classes)
        full_dataset_classes.load_dataset('full')
        # exp_dataset = Dataset(self.path, self.exp_dataset_name)
        # exp_dataset.load_dataset()
        # print(exp_dataset.dataset)
        for folder in self.folders:
            # print(os.path.join(self.path, self.network, self.exp_dataset_name, self.exp_dataset_name, folder, self.exp_dataset_name))
            path = os.path.join(self.path, self.network, self.exp_dataset_name, self.exp_dataset_name, folder, self.exp_dataset_name)
            out_path = os.path.join(self.path, self.network, self.exp_dataset_name, self.new_dataset_name, folder, self.new_dataset_name)
            exp_dataset = Dataset(path, 'gyro')
            exp_dataset.load_dataset('set')
            newDatasetContent = pd.DataFrame(columns=["new_img_datset",  "accx"])
            print(folder)
            os.makedirs(out_path, exist_ok=True)
            #copyfile(os.path.join(path, 'gyro.txt'), os.path.join(out_path, 'gyro.txt'))
            newDatasetRegress = File(out_path)
            for i, row in exp_dataset.dataset.iterrows():
                extract_img = row['new_img_dataset'].split('/')
                img_filter = (full_dataset_classes.dataset['new_img_dataset'] == extract_img[1])

                img_filter_original = (full_dataset.dataset['img_dataset'] == full_dataset_classes.dataset[img_filter]['img_dataset'].values[0])

                #print(full_dataset.dataset[img_filter]['accx'])
                newSample = {'new_img_datset': row['new_img_dataset'], 'accx': full_dataset.dataset[img_filter_original]['accx']}
                #print(newSample)
                #os.system('pause')
                newDatasetContent = newDatasetContent.append(newSample, ignore_index=True)
                #print(os.path.join(path, 'images', extract_img[1]))
                #img = cv.imread(os.path.join(path, 'images', extract_img[1]), cv.IMREAD_UNCHANGED)
                #print(img)
                #os.system('pause')
                # print(row)
                os.makedirs(os.path.join(out_path, "images"), exist_ok=True)
                copyfile(os.path.join(path, 'images', extract_img[1]), os.path.join(out_path, 'images', extract_img[1]))
                #os.system('pause')
                #print(newSample)
                #os.system('pause')
                #img = cv.imread(os.path.join(self.path, full_dataset.dataset[img_filter]['dataset'].item(), full_dataset.dataset[img_filter]['img_dataset'].item()), cv.IMREAD_UNCHANGED)
                # print(img.shape[0], img.shape[1])
                #h = img.shape[0]
                #w = img.shape[1]
                #x = 0
                #y = int(img.shape[0] * 0.3)
                #crop_img = img[y:y + h, x:x + w]
                #resized = cv.resize(crop_img, self.dimension, interpolation=cv.INTER_AREA)

                #os.makedirs(os.path.join(out_path, "images"), exist_ok=True)
                # cv.imshow('asdf', cv.resize(img, (480, 720), interpolation = cv.INTER_AREA))
                # new_path = os.path.join()
                # out_path = os.path.join()
                #cv.imwrite(os.path.join(out_path, "images", full_dataset.dataset[img_filter]['new_img_dataset'].item()), resized)
                # os.system('pause')
            newDatasetRegress.saveRegress(newDatasetContent, folder)
        pass

dataset_directory = "../../datasets"

network = 'dronet'

new_dataset_name = 'market_dataset_2_x_proportional_classes_all_00_regress'

full_dataset_name = "market_dataset_2_x_proportional"
full_dataset_name_classes = "market_dataset_2_x_proportional_classes_all"
exp_dataset = "market_dataset_2_x_proportional_classes_all_00"
dataset_modified = ModifyDataset(dataset_directory, full_dataset_name, exp_dataset, network, new_dataset_name, full_dataset_name_classes)
dataset_modified.modifyClassToRegress()
