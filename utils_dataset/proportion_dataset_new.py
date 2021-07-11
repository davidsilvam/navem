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
        self.dataset_normalized_added = None

    def loadDataset(self):
        self.dataset = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + ".txt"), sep=" ", engine="python",
                           encoding="ISO-8859-1", names=['img_dataset', 'img_original', 'folder', 'accx', 'dataset', 'new_img_datset'])

    def normalizeDataset(self, method, data, position):
        dataset_temp = self.dataset.copy()
        if method == 'zero_one':
            dataset_temp['accx'] = (data['accx'] - data['accx'].min()) / (data['accx'].max() - data['accx'].min())
        elif method == 'std_mean':
            dataset_temp['accx'] = (data['accx'] - np.mean(data['accx']))/np.std(data['accx'])
        self.dataset_normalized = dataset_temp

    def saveDatasetProportion(self, array, original):
        print("Saving new dataset proportion...")
        file = open(os.path.join(self.dataset_dir, self.output_dataset_name + ".txt"), "w")
        for i in range(len(array)):
            if original:
                temp = self.dataset.loc[self.dataset['img_dataset'] == array['img_dataset'][i]]
                # print(str(temp['img_dataset'].item()) + " " + str(temp['img_original'].item()) + " " + str(temp['folder'].item()) + " " + str(temp['accx'].item()))
                file.write(str(temp['img_dataset'].item()) + " " + str(temp['img_original'].item()) + " " + str(temp['folder'].item()) + " " + str(temp['accx'].item()) + "\n")
            else:
                file.write(str(array['img_dataset'][i]) + " " + str(array['img_original'][i]) + " " + str(
                    array["folder"][i]) + " " + str(array["accx"][i]) + "\n")
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
            print("Class {0} -> {1}".format(c, self.getProportionByClass(array, c, False)))

    def printProportional(self, normalize=False, print_normalized=False):
        RawDataset.loadDataset(self)
        if normalize:
            RawDataset.normalizeDataset(self, 'zero_one', self.dataset, 3)
        if print_normalized:
            for c in self.classes:
                print("Class {0} -> {1}".format(c, self.getProportionByClass(self.dataset_normalized, c, False)))
        else:
            for c in self.classes:
                print("Class {0} -> {1}".format(c, self.getProportionByClass(self.dataset, c, False)))

    def getProportionByClass(self, array, cla, percent):
        c = [0, 0, 0, 0, 0]
        max = 1
        for i in range(len(array)):
            # print(array)
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
                if not array.empty:
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

    def getFilterByClass(self, array, c):
        m = {}
        if (c == 0):
            m = (array['accx'] > 0) & (array['accx'] <= (self.max / 5))
        elif (c == 1):
            m = (array['accx'] > self.max / 5) & (
                    array['accx'] <= (self.max / 5) * 2)
        elif (c == 2):
            m = (array['accx'] > (self.max / 5) * 2) & (
                    array['accx'] <= (self.max / 5) * 3)
        elif (c == 3):
            m = (array['accx'] > (self.max / 5) * 3) & (
                    array['accx'] <= (self.max / 5) * 4)
        elif (c == 4):
            m = (array['accx'] > (self.max / 5) * 4)
        return m

    def makeProportionAdded(self):
        RawDataset.loadDataset(self)
        RawDataset.normalizeDataset(self, 'zero_one', self.dataset, 3)
        self.dataset_normalized_added = pd.DataFrame(columns=self.dataset_normalized.columns)
        qtd_data = [130, 130, 130, 130, 130]
        # qtd_data = [3862, 2474, 813, 850, 1088]
        for c in self.classes:
            m = {}
            if (c == 0):
                m = (self.dataset_normalized['accx'] > 0) & (self.dataset_normalized['accx'] <= (self.max / 5))
            elif (c == 1):
                m = (self.dataset_normalized['accx'] > self.max / 5) & (
                            self.dataset_normalized['accx'] <= (self.max / 5) * 2)
            elif (c == 2):
                m = (self.dataset_normalized['accx'] > (self.max / 5) * 2) & (
                            self.dataset_normalized['accx'] <= (self.max / 5) * 3)
            elif (c == 3):
                m = (self.dataset_normalized['accx'] > (self.max / 5) * 3) & (
                            self.dataset_normalized['accx'] <= (self.max / 5) * 4)
            elif (c == 4):
                m = (self.dataset_normalized['accx'] > (self.max / 5) * 4)
            dataset_current_class = self.dataset_normalized[m]

            ab = 0
            # if(len(dataset_current_class) < self.quantity):
            #     quantity = len(dataset_current_class)
            #     ab = random.sample(range(quantity), quantity)
            # else:
            #     ab = random.sample(range(self.quantity), self.quantity)
            if not len(dataset_current_class) < qtd_data[c]:
                ab = random.sample(range(qtd_data[c]), qtd_data[c])
            else:
                ab = random.sample(range(len(dataset_current_class)), len(dataset_current_class))
            print(len(ab), len(dataset_current_class))
            for sample in ab:
                self.dataset_normalized_added = self.dataset_normalized_added.append(dataset_current_class.iloc[sample])

        self.dataset_normalized_added.reset_index(drop=True, inplace=True)
        self.printProportionAllClasses(self.dataset_normalized_added)
        RawDataset.saveDatasetProportion(self, self.dataset_normalized_added, original=False)

    def makeProportion(self):
        RawDataset.loadDataset(self)
        RawDataset.normalizeDataset(self, 'zero_one', self.dataset, 3)
        # print(self.getProportionByClass(self.dataset_normalized, 4, False))
        # self.printProportionAllClasses(self.dataset_normalized)
        # os.system("pause")
        # print(self.dataset_normalized.columns)
        # self.dataset_normalized_added = pd.DataFrame(columns=self.dataset_normalized.columns)
        # print(len(self.dataset_normalized_added))
        # print(self.dataset_normalized.iloc[0])
        # self.dataset_normalized_added = self.dataset_normalized_added.append(self.dataset_normalized.iloc[0])
        # print(self.dataset_normalized_added)
        # os.system('pause')

        for c in self.classes:
            # print(self.getProportionByClass(self.dataset_normalized_added, c, False))
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
                # increment
                add_ = False
                if add_:
                    self.dataset_normalized_added = self.dataset_normalized_added.append(self.dataset_normalized.iloc[val])
                    self.dataset_normalized_added.reset_index(drop=True, inplace=True)
                    rem = a.index[val]
                    self.dataset_normalized = self.dataset_normalized.drop([rem])
                    self.dataset_normalized.reset_index(drop=True, inplace=True)
                else:
                    rem = a.index[val]
                    self.dataset_normalized = self.dataset_normalized.drop([rem])
                    self.dataset_normalized.reset_index(drop=True, inplace=True)
                print("Classe {0} -> {1}".format(c, self.getProportionByClass(self.dataset_normalized, c, False)))
            print("==============================")
            self.printProportionAllClasses(self.dataset_normalized)
        RawDataset.saveDatasetProportion(self, self.dataset_normalized, original=False)

dataset_directory = "../../datasets"
dataset_name = "sidewalk_accy_all_datasets_classes_new_900"
output_dataset_name = "market_dataset_y_out_pc"
quantity_per_class = 927

proportion = Proportion(dataset_directory, dataset_name, dataset_directory, output_dataset_name, quantity_per_class)
# proportion.makeProportion()
# proportion.makeProportionAdded()
proportion.printProportional(normalize=False, print_normalized=False)
