from dataset import Dataset
from file import File
import os
import cv2
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import numpy as np

class GenerateSimalirity(object):
    def __init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name, datasets_list, threshold, method_sim, joined_dataset = True):
        self.datasets_list = datasets_list
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_output_dir = dataset_output_dir
        self.output_dataset_name = output_dataset_name
        self.dataset_all = None
        self.count = 0
        self.dimension = (224, 224)
        self.threshold = threshold
        self.method_sim = method_sim
        self.dataset_all = pd.DataFrame(
            columns=["img_dataset", "img_original", "folder", "accx", 'dataset', 'new_img_datset'])
        self.dataset_all_ignore = pd.DataFrame(
            columns=["img_dataset", "img_original", "folder", "accx", 'dataset', 'new_img_datset'])
        Dataset.__init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name, joined_dataset)
        Dataset.loadDataset(self)
        self.class_map = np.zeros(5)

    def generate(self):
        #a = self.dataset.groupby('folder').apply(pd.DataFrame.sort_values, 'img_dataset') # funciona
        b = self.dataset.groupby('folder')
        dataset_file = File(os.path.join(self.dataset_dir))
        #print(os.path.join(self.dataset_dir, self.output_dataset_name))
        #os.system('pause')
        for k, g in b:
            print(k)
            group = g.sort_values(['img_dataset'])
            self.window(group)
            print(self.class_map)
        dataset_file.saveFileAllDataset(self.dataset_all, self.output_dataset_name)
        print(self.output_dataset_name)

    def window(self, group):
        pivo = 5
        # group.shape[0]
        for w in range(5, group.shape[0] - pivo):
            print(w)
            #print(len(group.iloc[w]))
            #print('antes')
            #print(len(group[w - pivo:w]))
            #print('depois')
            #print(len(group[w + 1:(w + 1) + pivo]))
            # os.system('pause')
            self.apply_similarity(group[w - pivo:w], group.iloc[w], group[w + 1:(w + 1) + pivo], self.method_sim)

    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def checkIfExist(self, original, obj):
        # print(pd.any(self.dataset_all == obj))
        # os.system('pause')
        #print('antes')
        #a = (self.dataset_all == obj).any(axis=0).any()
        #print(a)
        #print('depois')
        #os.system('pause')
        #print((original == obj).all(1).any())
        #os.system('pause')
        return (original == obj).all(1).any()

    def apply_similarity(self, before, current, after, method='ssim'):
        #print(len(before))
        #print(before)
        #print(before.iloc[4])
        #os.system('pause')
        for img_before in range(len(before)):
            # print(before['accx'].iloc[img_before])
            #print(after)
            #os.system('pause')
            if before['accx'].iloc[img_before] != current['accx']:
                if not self.checkIfExist(self.dataset_all, before.iloc[img_before]):
                    path_img_before = os.path.join(self.dataset_dir, before['dataset'].iloc[img_before], before['img_dataset'].iloc[img_before])
                    path_img_current = os.path.join( self.dataset_dir, current['dataset'], current['img_dataset'])
                    #print(path_img_before)
                    #print(path_img_current)
                    img1 = cv2.cvtColor(cv2.imread(path_img_before), cv2.COLOR_BGR2GRAY)
                    img2 = cv2.cvtColor(cv2.imread(path_img_current), cv2.COLOR_BGR2GRAY)
                    resized_img1 = cv2.resize(img1, self.dimension, interpolation=cv2.INTER_AREA)
                    resized_img2 = cv2.resize(img2, self.dimension, interpolation=cv2.INTER_AREA)
                    if method == 'ssim':
                        res_sim = ssim(resized_img1, resized_img2)
                    elif method == 'mse':
                        res_sim = self.mse(resized_img1, resized_img2)
                    if self.method_sim == 'mse':
                        if res_sim > self.threshold:
                            if not self.checkIfExist(self.dataset_all, before.iloc[img_before]) and not self.checkIfExist(self.dataset_all_ignore, before.iloc[img_before]):
                                self.dataset_all = self.dataset_all.append(before.iloc[img_before])
                                self.class_map[before['accx'].iloc[img_before]] = self.class_map[
                                                                                      before['accx'].iloc[img_before]] + 1
                                #print('add before different class')
                        else:
                            if not self.checkIfExist(self.dataset_all_ignore, before.iloc[img_before]): # parou aqui
                                print('exception before')
                                self.dataset_all_ignore = self.dataset_all_ignore.append(before.iloc[img_before])
                    elif self.method_sim == 'ssim':
                        if res_sim < self.threshold:
                            if not self.checkIfExist(self.dataset_all, before.iloc[img_before]) and not self.checkIfExist(self.dataset_all_ignore, before.iloc[img_before]):
                                self.dataset_all = self.dataset_all.append(before.iloc[img_before])
                                self.class_map[before['accx'].iloc[img_before]] = self.class_map[
                                                                                      before['accx'].iloc[img_before]] + 1
                                #print('add before different class')
                        else:
                            if not self.checkIfExist(self.dataset_all_ignore, before.iloc[img_before]): # parou aqui
                                print('exception before')
                                self.dataset_all_ignore = self.dataset_all_ignore.append(before.iloc[img_before])
            else:

                if not self.checkIfExist(self.dataset_all, before.iloc[img_before]) and not self.checkIfExist(self.dataset_all_ignore, before.iloc[img_before]):
                    self.dataset_all = self.dataset_all.append(before.iloc[img_before])
                    self.class_map[before['accx'].iloc[img_before]] = self.class_map[before['accx'].iloc[img_before]] + 1
                    #print('add before equals class')
        for img_after in range(len(after)):
            # print(before['accx'].iloc[img_before])
            # os.system('pause')
            if after['accx'].iloc[img_after] != current['accx']:
                if not self.checkIfExist(self.dataset_all, before.iloc[img_before]):
                    path_img_after = os.path.join(self.dataset_dir, after['dataset'].iloc[img_after], after['img_dataset'].iloc[img_after])
                    path_img_current = os.path.join( self.dataset_dir, current['dataset'], current['img_dataset'])
                    #print(path_img_before)
                    #print(path_img_current)
                    img1 = cv2.cvtColor(cv2.imread(path_img_after), cv2.COLOR_BGR2GRAY)
                    img2 = cv2.cvtColor(cv2.imread(path_img_current), cv2.COLOR_BGR2GRAY)
                    resized_img1 = cv2.resize(img1, self.dimension, interpolation=cv2.INTER_AREA)
                    resized_img2 = cv2.resize(img2, self.dimension, interpolation=cv2.INTER_AREA)
                    if method == 'ssim':
                        res_sim = ssim(resized_img1, resized_img2)
                    elif method == 'mse':
                        res_sim = self.mse(resized_img1, resized_img2)
                    # print(res_sim)
                    if self.method_sim == 'mse':
                        if res_sim > self.threshold:
                            if not self.checkIfExist(self.dataset_all, after.iloc[img_after]) and not self.checkIfExist(self.dataset_all_ignore, after.iloc[img_after]):
                                self.dataset_all = self.dataset_all.append(after.iloc[img_after])
                                self.class_map[after['accx'].iloc[img_after]] = self.class_map[
                                                                                      after['accx'].iloc[img_after]] + 1
                        else:
                            if not self.checkIfExist(self.dataset_all_ignore, after.iloc[img_after]): # parou aqui
                                self.dataset_all_ignore = self.dataset_all_ignore.append(after.iloc[img_after])
                                print('exception after')
                    elif self.method_sim == 'ssim':
                        if res_sim < self.threshold:
                            if not self.checkIfExist(self.dataset_all, after.iloc[img_after]) and not self.checkIfExist(self.dataset_all_ignore, after.iloc[img_after]):
                                self.dataset_all = self.dataset_all.append(after.iloc[img_after])
                                self.class_map[after['accx'].iloc[img_after]] = self.class_map[
                                                                                      after['accx'].iloc[img_after]] + 1
                        else:
                            if not self.checkIfExist(self.dataset_all_ignore, after.iloc[img_after]): # parou aqui
                                self.dataset_all_ignore = self.dataset_all_ignore.append(after.iloc[img_after])
                                print('exception after')
            else:

                if not self.checkIfExist(self.dataset_all, after.iloc[img_after]) and not self.checkIfExist(self.dataset_all_ignore, after.iloc[img_after]):
                    self.dataset_all = self.dataset_all.append(after.iloc[img_after])
                    self.class_map[after['accx'].iloc[img_after]] = self.class_map[after['accx'].iloc[img_after]] + 1
                    #print('asdf')
        #print(self.dataset_all[['img_dataset', 'accx']])
        #os.system('pause')

dataset_directory = "../../datasets"
dataset_name = "market_accx_all_datasets_classes"
datasets_names_files = ['market_accx_all_datasets_classes']
output_dataset_name = "market_accx_all_datasets_classes_rem_sim_whiskers_bottom" # Not used
threshold = 0.26423958723819224
method_sim = 'ssim'

join = GenerateSimalirity(dataset_directory, dataset_name, dataset_directory, output_dataset_name, datasets_names_files, threshold, method_sim)
join.generate()
