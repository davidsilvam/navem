import os
import pandas as pd
import numpy as np
import random

class RawDataset(object):

    def __init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_output_dir = dataset_output_dir
        self.output_dataset_name = output_dataset_name
        self.dataset = None
        self.dataset_normalized = None

    def loadDataset(self):
        self.dataset = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + ".txt"), sep=" ", engine="python",
                           encoding="ISO-8859-1", names=['img_dataset', 'img_original', 'folder', 'accx'])

    def normalizeDataset(self, method, data, position):
        dataset_temp = self.dataset.copy()
        if method == 'zero_one':
            dataset_temp['accx'] = (data['accx'] - data['accx'].min()) / (data['accx'].max() - data['accx'].min())
        elif method == 'std_mean':
            dataset_temp['accx'] = (data['accx'] - np.mean(data['accx']))/np.std(data['accx'])
        self.dataset_normalized = dataset_temp

    def saveDatasetProportion(self, original):
        print("Saving new dataset proportion...")
        file = open(os.path.join(self.dataset_dir, self.output_dataset_name + ".txt"), "w")
        for i in range(len(self.dataset_normalized)):
            if original:
                temp = self.dataset.loc[self.dataset['img_dataset'] == self.dataset_normalized['img_dataset'][i]]
                # print(str(temp['img_dataset'].item()) + " " + str(temp['img_original'].item()) + " " + str(temp['folder'].item()) + " " + str(temp['accx'].item()))
                file.write(str(temp['img_dataset'].item()) + " " + str(temp['img_original'].item()) + " " + str(temp['folder'].item()) + " " + str(temp['accx'].item()) + "\n")
            else:
                file.write(str(self.dataset_normalized['img_dataset'][i]) + " " + str(self.dataset_normalized['img_original'][i]) + " " + str(
                    self.dataset_normalized["folder"][i]) + " " + str(self.dataset_normalized["accx"][i]) + "\n")
        file.close()
        print("Saved.")


class Proportion(object):

    def __init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name, quantity):
        self.classes = [0,1,2,3,4]
        self.quantity = quantity
        self.max = 1
        RawDataset.__init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name)

    def printProportionAllClasses(self, array):
        for c in self.classes:
            print("Class {0} -> {1}".format(c, self.getProportionByClass(self.dataset_normalized, c, False)))

    def printProportional(self):
        RawDataset.loadDataset(self)
        RawDataset.normalizeDataset(self, 'zero_one', self.dataset, 3)
        for c in self.classes:
            print("Class {0} -> {1}".format(c, self.getProportionByClass(self.dataset, c, False)))

    def getProportionByClass(self, array, cla, percent):
        c = [0, 0, 0, 0, 0]
        max = 1
        for i in range(len(array)):
            if isinstance(array.loc[i, 'accx'], float):
                if (array.loc[i, 'accx'] > 0) and (array.loc[i, 'accx'] <= max / 5):
                    c[0] += 1
                elif (array.loc[i, 'accx'] > max / 5) and (array.loc[i, 'accx'] <= (max / 5) * 2):
                    c[1] += 1
                elif (array.loc[i, 'accx'] > (max / 5) * 2) and (array.loc[i, 'accx'] <= (max / 5) * 3):
                    c[2] += 1
                elif (array.loc[i, 'accx'] > (max / 5) * 3) and (array.loc[i, 'accx'] <= (max / 5) * 4):
                    c[3] += 1
                else:
                    c[4] += 1
            else:
                if (array.loc[i, 'accx']  == 0):
                    c[0] += 1
                elif (array.loc[i, 'accx'] == 1):
                    c[1] += 1
                elif (array.loc[i, 'accx'] == 2):
                    c[2] += 1
                elif (array.loc[i, 'accx'] == 3):
                    c[3] += 1
                else:
                    c[4] += 1
        if (percent):
            return c[cla] / len(array)
        else:
            return c[cla]

    def makeProportion(self):
        RawDataset.loadDataset(self)
        RawDataset.normalizeDataset(self, 'zero_one', self.dataset, 3)
        # print(self.getProportionByClass(self.dataset_normalized, 4, False))
        # self.printProportionAllClasses(self.dataset_normalized)
        for c in self.classes:
            while(self.getProportionByClass(self.dataset_normalized, c, False) > self.quantity):
                m = {}
                if(c == 0):
                    m = (self.dataset_normalized['accx'] > 0) & (self.dataset_normalized['accx'] <= (self.max / 5))
                elif(c == 1):
                    m = (self.dataset_normalized['accx'] > self.max / 5) & (self.dataset_normalized['accx'] <= (self.max / 5) * 2)
                elif(c == 2):
                    m = (self.dataset_normalized['accx'] > (self.max / 5) * 2) & (self.dataset_normalized['accx'] <= (self.max / 5) * 3)
                elif(c == 3):
                    m = (self.dataset_normalized['accx'] > (self.max / 5) * 3) & (self.dataset_normalized['accx'] <= (self.max / 5) * 4)
                elif(c == 4):
                    m = (self.dataset_normalized['accx'] > (self.max / 5) * 4)
                a = self.dataset_normalized[m]
                val = random.randint(1, len(a) - 1)
                rem = a.index[val]
                self.dataset_normalized = self.dataset_normalized.drop([rem])
                self.dataset_normalized.reset_index(drop=True, inplace=True)
                print("Classe {0} -> {1}".format(c, self.getProportionByClass(self.dataset_normalized, c, False)))
            print("==============================")
            self.printProportionAllClasses(self.dataset_normalized)
        RawDataset.saveDatasetProportion(self, original=False)

dataset_directory = "../../datasets"
dataset_name = "sidewalk_accy_flipped_all_out_classes"
output_dataset_name = "sidewalk_accx_all_out"
quantity_per_class = 184

proportion = Proportion(dataset_directory, dataset_name, dataset_directory, output_dataset_name, quantity_per_class)
# proportion.makeProportion()
proportion.printProportional()