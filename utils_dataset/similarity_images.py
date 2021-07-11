from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from dataset import Dataset
from numpy import unravel_index
from file import  File
from itertools import combinations

class Similarity(object):
    def __init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_output_dir = dataset_output_dir
        self.output_dataset_name = output_dataset_name

    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def getSamplesByClasses(self, dataset, clas):
        filter = (dataset['accx'] == clas)
        return filter

    def bestsSamples(self, subset_in, add):
        # print(subset)
        subset = subset_in.copy()
        dinamic = np.zeros((len(subset), len(subset)))
        result_array = []
        # print(dinamic)
        for sample in range(len(subset)):
            img1 = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir,
                                                        self.dataset_name,
                                                        subset['new_img_datset'].iloc[sample])), cv2.COLOR_BGR2GRAY)
            # img1 = os.path.join(self.dataset_dir, self.dataset_name, subset['new_img_datset'].iloc[sample])
            for sample2 in range(len(subset)):
                if dinamic[sample][sample2] == 0:
                    if not sample == sample2:
                        img2 = cv2.cvtColor(cv2.imread(
                            os.path.join(self.dataset_dir, self.dataset_name,
                                         subset['new_img_datset'].iloc[sample2])), cv2.COLOR_BGR2GRAY)
                        # img2 = os.path.join(self.dataset_dir, self.dataset_name, subset['new_img_datset'].iloc[sample2])
                        # print(img1)
                        # print(img2)
                        dinamic[sample][sample2] = dinamic[sample2][sample] = ssim(img1, img2)
                        result_array.append(ssim(img1, img2))
        # img11 = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir,
        #                                             self.dataset_name,
        #                                             subset['new_img_datset'].iloc[0])), cv2.COLOR_BGR2GRAY)
        # img21 = cv2.cvtColor(cv2.imread(
        #     os.path.join(self.dataset_dir, self.dataset_name,
        #                  subset['new_img_datset'].iloc[1])), cv2.COLOR_BGR2GRAY)
        # img31 = cv2.cvtColor(cv2.imread(
        #     os.path.join(self.dataset_dir, self.dataset_name,
        #                  subset['new_img_datset'].iloc[2])), cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img1', img11)
        # cv2.imshow('img2', img21)
        # cv2.imshow('img3', img31)
        # cv2.waitKey()
        # print(ssim(img11, img21))
        # print(ssim(img11, img31))
        # print(ssim(img21, img31))


        # print(result_array)
        # print(max(result_array))
        # print(result_array.index(max(result_array)))
        # print(dinamic)
        # print(unravel_index(dinamic.argmax(), dinamic.shape))

        comb = combinations(range(len(subset_in)), 2)
        combinations_ = []
        for combination in comb:
            combinations_.append(combination)

        index_result = [-2] * len(combinations_)
        for index, combination in enumerate(combinations_):
            index_result[index] = dinamic[combination[0]][combination[1]]

        # print(index_result)
        count = 0
        final_result = []
        for results in range(len(index_result)):
            # print(max(index_result), index_result.index(max(index_result)))
            if count < add:
                index_result[index_result.index(max(index_result))] = -2
                count+=1

        mem = []
        for sample in range(len(index_result)):
            if index_result[sample] != -2:
                if not combinations_[sample][0] in mem:
                    final_result.append(subset.iloc[combinations_[sample][0]])
                    mem.append(combinations_[sample][0])
                elif not combinations_[sample][1] in mem:
                    final_result.append(subset.iloc[combinations_[sample][1]])
                    mem.append(combinations_[sample][1])
                if len(mem) >= add:
                    break
        # print(index_result)
        # print(combinations_)
        # print(mem)
        # print(len(final_result), len(mem))
        # os.system('pause')
        return final_result



    def makeNewDataset(self):
        dataset = Dataset(self.dataset_dir, self.dataset_name, self.dataset_output_dir, self.output_dataset_name, joined_datasaet = True)
        dataset.loadDataset()
        new_dataset = pd.DataFrame(columns=dataset.dataset.columns)
        # print(dataset.dataset[self.getSamplesByClasses(dataset.dataset, 0)][0:10])
        # print('asdf')
        multiple = 5
        add = 2
        for classes in range(5):
            filter_class = dataset.dataset[self.getSamplesByClasses(dataset.dataset, classes)]
            # new_dataset.append(filter_class.iloc[0])
            # print(filter_class[0:2])
            for sample in range(0, len(filter_class), multiple):
                # print(dataset.dataset['img_dataset'].iloc[sample], dataset.dataset['img_dataset'].iloc[sample + 1],
                #      dataset.dataset['img_dataset'].iloc[sample + 2])
                # self.bestsSamples(dataset.dataset.iloc[sample],
                #                   dataset.dataset.iloc[sample + 1],
                #                   dataset.dataset.iloc[sample + 2])
                print('Class {0} sample {1}'.format(classes, sample))
                subs = None
                if sample == 0:
                    subs = filter_class[sample: multiple]
                else:
                    subs = filter_class[sample: sample + multiple]
                # print(subs)
                result_samples = self.bestsSamples(subs, add)
                for result in result_samples:
                    new_dataset = new_dataset.append(result)
            # print(len(new_dataset))
            # os.system('pause')
        f = File(self.dataset_dir)
        f.saveFileAllDataset(new_dataset, self.output_dataset_name)






dataset_dir = '../../datasets'
dataset_name = 'sidewalk_accy_all_datasets_classes'
output_dataset_name = 'sidewalk_accy_all_datasets_classes_new_900'
img1_name = '000041.jpg'
img2_name = '000091.jpg'
img1 = cv2.imread(os.path.join(dataset_dir, dataset_name, img1_name))
img2 = cv2.imread(os.path.join(dataset_dir, dataset_name, img2_name))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
test = Similarity(dataset_dir, dataset_name, dataset_dir, output_dataset_name)
print(test.mse(img1, img2))
print(ssim(img1, img2))

test.makeNewDataset()
