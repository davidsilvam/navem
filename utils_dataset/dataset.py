import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset(object):

    def __init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name, joined_datasaet = False):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_output_dir = dataset_output_dir
        self.output_dataset_name = output_dataset_name
        self.dataset = None
        self.dataset_normalized = None
        self.dataset_flipped = None
        self.min_max_array = []
        self.trainingSet = None
        self.valSet = None
        self.testSet = None
        self.sets = []
        self.joined_dataset = joined_datasaet

    def loadDataset(self):
        if not self.joined_dataset:
            self.dataset = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + ".txt"),
                                       sep=" ", engine="python", encoding="ISO-8859-1",
                                       names=['img_dataset', 'img_original', 'folder', 'accx'])
        else:
            self.dataset = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + ".txt"),
                                       sep=" ", engine="python", encoding="ISO-8859-1",
                                       names=['img_dataset', 'img_original', 'folder', 'accx', 'dataset',
                                              'new_img_datset', 'flipped'])

    def normalizeDataset(self, method, data, position):
        dataset_temp = self.dataset.copy()
        if method == 'zero_one':
            dataset_temp['accx'] = (data['accx'] - data['accx'].min()) / (data['accx'].max() - data['accx'].min())
        elif method == 'std_mean':
            dataset_temp['accx'] = (data['accx'] - np.mean(data['accx']))/np.std(data['accx'])
        self.dataset_normalized = dataset_temp

    def getMinMax(self, col):
        return [self.dataset[:, col].min(), self.dataset[:, col].max()]

    def splitDataset(self, train, val, test, flipped=True):
        auxSet = None
        # print(self.dataset["accx"])
        # os.syste)
        # os.system('pause')
        if flipped:
            withFlip = (self.dataset['flipped'] > 0)
            withFlip = self.dataset[withFlip]
            withoutFlip = (self.dataset['flipped'] == 0)
            withoutFlip = self.dataset[withoutFlip]
            #experiment
            todo = self.dataset.shape[0]
            compWith = (1 - ((withFlip.shape[0]*(val + test)) / (todo * (val + test)))) * (val + test)
            compWithout = ((withFlip.shape[0]*(val + test)) / (todo * (val + test))) * (val + test)
            self.trainingSetWithFlip, self.valSetWithFlip = train_test_split(withFlip, test_size=(test + val - compWith), stratify=withFlip["accx"])
            print("Train Val".format())
            print("{0} {1:4}".format(self.trainingSetWithFlip.shape[0], self.valSetWithFlip.shape[0]))
            self.trainingSetWithoutSet, auxSetWithOut = train_test_split(withoutFlip, test_size=(test + val + compWithout), stratify=withoutFlip["accx"])
            print("Train Val".format())
            print("{0} {1:4}".format(self.trainingSetWithoutSet.shape[0], auxSetWithOut.shape[0]))
            print('-----------')
            val_todo = self.valSetWithFlip.shape[0] + auxSetWithOut.shape[0]
            perWith =  self.valSetWithFlip.shape[0] / val_todo
            print(1 - perWith)
            valComplete, testComplete = train_test_split(auxSetWithOut, test_size=(1 - perWith + 0.009), stratify=auxSetWithOut["accx"])
            print("Val Test".format())
            print("{0} {1:4}".format(str(valComplete.shape[0]), str(testComplete.shape[0])))
            self.trainingSet = self.trainingSetWithFlip.append(self.trainingSetWithoutSet)
            self.trainingSet = self.trainingSet.sort_values('img_dataset', ascending=True)
            self.valSet = self.valSetWithFlip .append(valComplete)
            self.valSet = self.valSet.sort_values('img_dataset', ascending=True)
            self.testSet = testComplete
            self.testSet = self.testSet.sort_values('img_dataset', ascending=True)
            print("Train Val Test".format())
            print("{0} {1:4} {2:4}".format(self.trainingSet.shape[0], self.valSet.shape[0], self.testSet.shape[0]))
            self.sets = [self.trainingSet, self.valSet, self.testSet]
        #experiment
        # print(train*self.dataset.shape[0])
        # print(withFlip.shape[0])
        else:
            self.trainingSet, auxSet = train_test_split(self.dataset, test_size=(test + val), stratify=self.dataset["accx"])
            self.valSet, self.testSet = train_test_split(auxSet, test_size=0.5, stratify=auxSet["accx"])
            print("Train Val Test".format())
            print("{0} {1:4} {2:4}".format(self.trainingSet.shape[0], self.valSet.shape[0], self.testSet.shape[0]))
            self.trainingSet = self.trainingSet.sort_values('img_dataset', ascending=True)
            self.valSet = self.valSet.sort_values('img_dataset', ascending=True)
            self.testSet = self.testSet.sort_values('img_dataset', ascending=True)
            self.sets = [self.trainingSet, self.valSet, self.testSet]

        # os.system('pause')

        # check quantity per class
        # unique, counts = np.unique(self.valSet["accx"], return_counts=True)
        # print(dict(zip(unique, counts)))